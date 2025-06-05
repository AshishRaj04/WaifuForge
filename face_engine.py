# ------ IMPORTING LIBRARIES ------

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    utils,
    metrics,
    optimizers,
)

# ------ CONSTANTS AND CONFIGURATION ------

IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 128
NUM_FEATURES = 64
Z_DIM = 128
LEARNING_RATE = 0.0002
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
EPOCHS = 1000
CRITIC_STEPS = 5
GP_WEIGHT = 50.0

# ------ WEIGHTS DIRECTORY ------
weights_path = "checkpoints/checkpoints_84/wgan.weights.h5"

# ------ WGAN-GP MODEL ------

class WGANGP(models.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super(WGANGP, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super(WGANGP, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_wass_loss_metric = metrics.Mean(name="c_wass_loss")
        self.c_gp_metric = metrics.Mean(name="c_gp")
        self.c_loss_metric = metrics.Mean(name="c_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as tape:
                fake_images = self.generator(
                    random_latent_vectors, training=True
                )
                fake_predictions = self.critic(fake_images, training=True)
                real_predictions = self.critic(real_images, training=True)

                c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                    real_predictions
                )
                c_gp = self.gradient_penalty(
                    batch_size, real_images, fake_images
                )
                c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


# ------ FACE GENERATOR CLASS ------

class FaceGenerator:
    def __init__(self, checkpoint_path=weights_path):
        self.Z_DIM = 128
        self.IMAGE_SIZE = 64
        self.CHANNELS = 3

        # Initialize models
        self.generator = self._build_generator()
        self.critic = self._build_critic()

        # Initialize WGAN-GP
        self.wgangp = WGANGP(
            critic=self.critic,
            generator=self.generator,
            latent_dim=self.Z_DIM,
            critic_steps=5,
            gp_weight=50.0,
        )

        # Build and load weights
        try:
            self.wgangp.build(
                input_shape=(None, self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANNELS)
            )
            self.wgangp.load_weights(checkpoint_path)
            print(f"Successfully loaded weights from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")

    def _build_generator(self):
        generator_input = layers.Input(shape=(Z_DIM,))
        x = layers.Reshape((1, 1, Z_DIM))(generator_input)

        x = layers.Conv2DTranspose(
            512, kernel_size=4, strides=1, padding="valid", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Conv2DTranspose(
            256, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)

        generator_output = layers.Conv2DTranspose(
            CHANNELS, kernel_size=4, strides=2, padding="same", activation="tanh"
        )(x)
        generator = models.Model(generator_input, generator_output)
        return generator

    def _build_critic(self):
        critic_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

        x = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_normal')(critic_input)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same" , kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same",kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Conv2D(1, kernel_size=4, strides=1, padding="valid")(x)
        critic_output = layers.GlobalAveragePooling2D()(x)

        critic = models.Model(critic_input, critic_output)
        return critic

    @tf.function
    def generate_face(self, batch_size=1):
        """
        Generate faces using the trained generator
        Args:
            batch_size: Number of images to generate
        Returns:
            numpy array of generated images in uint8 format (0-255)
        """
        try:
            # Generate random noise
            noise = tf.random.normal(shape=(batch_size, self.Z_DIM))

            # Generate images
            generated_images = self.wgangp.generator(noise, training=False)

            return generated_images

        except Exception as e:
            print(f"Error generating faces: {str(e)}")
            return None


# Initialize the generator once and reuse
try:
    face_generator = FaceGenerator()
except Exception as e:
    print(f"Failed to initialize face generator: {str(e)}")
    face_generator = None


# Generate a single face image
generated_image = face_generator.generate_face(batch_size=1)
# Display the generated image
if generated_image is not None:
    plt.imshow(generated_image[0])
    plt.axis('off')
    plt.show()
# Save the generated image
if generated_image is not None:
    output_dir = "generated_faces"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_face.png")
    plt.imsave(output_path, generated_image[0])
    print(f"Generated image saved to {output_path}")