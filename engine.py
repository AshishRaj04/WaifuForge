import tensorflow as tf

# Model architecture parameters
block_size = 8
n_emb = 128
n_neurons_l1 = 512
n_neurons_l2 = 384
n_neurons_l3 = 256
vocab_size = 67

# Initialize model variables
C = tf.Variable(tf.zeros([vocab_size, n_emb]))
W1 = tf.Variable(tf.zeros([n_emb * block_size, n_neurons_l1]))
b1 = tf.Variable(tf.zeros([n_neurons_l1]))
W2 = tf.Variable(tf.zeros([n_neurons_l1, n_neurons_l2]))
b2 = tf.Variable(tf.zeros([n_neurons_l2]))
W3 = tf.Variable(tf.zeros([n_neurons_l2, n_neurons_l3]))
b3 = tf.Variable(tf.zeros([n_neurons_l3]))
W4 = tf.Variable(tf.zeros([n_neurons_l3, vocab_size]))
b4 = tf.Variable(tf.zeros([vocab_size]))

# BatchNorm parameters
gamma_1 = tf.Variable(tf.ones([n_neurons_l1]))
beta_1 = tf.Variable(tf.zeros([n_neurons_l1]))
gamma_2 = tf.Variable(tf.ones([n_neurons_l2]))
beta_2 = tf.Variable(tf.zeros([n_neurons_l2]))
gamma_3 = tf.Variable(tf.ones([n_neurons_l3]))
beta_3 = tf.Variable(tf.zeros([n_neurons_l3]))

# Running statistics
running_mean_01 = tf.Variable(tf.zeros([1, n_neurons_l1]), trainable=False)
running_std_01 = tf.Variable(tf.ones([1, n_neurons_l1]), trainable=False)
running_mean_02 = tf.Variable(tf.zeros([1, n_neurons_l2]), trainable=False)
running_std_02 = tf.Variable(tf.ones([1, n_neurons_l2]), trainable=False)
running_mean_03 = tf.Variable(tf.zeros([1, n_neurons_l3]), trainable=False)
running_std_03 = tf.Variable(tf.ones([1, n_neurons_l3]), trainable=False)

def generate_name_sample(temperature=0.8 , load=True):
    """Generate a single name using the model"""
    context = [0] * block_size
    output = []
    
    # Load parameters if not loaded
    if load:
        load_parameters()
    
    while True:
        logits = get_logits(context)
        probs = tf.nn.softmax(logits / temperature, axis=-1)
        ix = tf.random.categorical(tf.math.log(probs), num_samples=1)[0]
        next_char = ix.numpy()[0]
        
        if next_char == 0 or len(output) > 40:  # End token or max length
            break
            
        output.append(next_char)
        context = context[1:] + [next_char]
    
    # Convert indices to characters
    name = ''.join(idx_to_str[ix] for ix in output)
    return name

def get_logits(context):
    """Get logits from the model for given context"""
    emb = tf.gather(C, tf.constant([context], dtype=tf.int32))
    x1 = tf.reshape(emb, shape=(-1, n_emb * block_size))
    x1 = x1 @ W1 + b1
    x1 = (x1 - running_mean_01) / tf.sqrt(running_std_01 + 1e-5)
    x1 = x1 * gamma_1 + beta_1
    h1 = tf.keras.activations.gelu(x1)
    
    x2 = h1 @ W2 + b2
    x2 = (x2 - running_mean_02) / tf.sqrt(running_std_02 + 1e-5)
    x2 = x2 * gamma_2 + beta_2
    h2 = tf.keras.activations.gelu(x2)
    
    x3 = h2 @ W3 + b3
    x3 = (x3 - running_mean_03) / tf.sqrt(running_std_03 + 1e-5)
    x3 = x3 * gamma_3 + beta_3
    h3 = tf.keras.activations.gelu(x3)
    
    logits = h3 @ W4 + b4
    return logits

idx_to_str = {1: '#', 2: '-', 3: '0', 4: '1', 5: '2', 6: '3', 7: '4', 8: '5', 9: '6', 10: '7', 11: '8', 12: '9', 13: '?', 14: 'A', 15: 'B', 16: 'C', 17: 'D', 18: 'E', 19: 'F', 20: 'G', 21: 'H', 22: 'I', 23: 'J', 24: 'K', 25: 'L', 26: 'M', 27: 'N', 28: 'O', 29: 'P', 30: 'Q', 31: 'R', 32: 'S', 33: 'T', 34: 'U', 35: 'V', 36: 'W', 37: 'X', 38: 'Y', 39: 'Z', 40: '_', 41: 'a', 42: 'b', 43: 'c', 44: 'd', 45: 'e', 46: 'f', 47: 'g', 48: 'h', 49: 'i', 50: 'j', 51: 'k', 52: 'l', 53: 'm', 54: 'n', 55: 'o', 56: 'p', 57: 'q', 58: 'r', 59: 's', 60: 't', 61: 'u', 62: 'v', 63: 'w', 64: 'x', 65: 'y', 66: 'z', 0: '.'}

def load_parameters():
    """Load trained model parameters from checkpoint"""
    global C, W1, b1, W2, b2, W3, b3, W4, b4
    global gamma_1, beta_1, gamma_2, beta_2, gamma_3, beta_3
    global running_mean_01, running_std_01, running_mean_02, running_std_02, running_mean_03, running_std_03
    
    
    try:
        checkpoint = tf.train.Checkpoint(
            C=C, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4,
            gamma_1=gamma_1, beta_1=beta_1, gamma_2=gamma_2, beta_2=beta_2, 
            gamma_3=gamma_3, beta_3=beta_3,
            running_mean_01=running_mean_01, running_std_01=running_std_01,
            running_mean_02=running_mean_02, running_std_02=running_std_02,
            running_mean_03=running_mean_03, running_std_03=running_std_03
        )
        
        latest_checkpoint = tf.train.latest_checkpoint("./checkpoints/checkpoints_84")
        if latest_checkpoint is None:
            raise FileNotFoundError("No checkpoint found")
            
        status = checkpoint.restore(latest_checkpoint)
        status.expect_partial()
        print("Model restored from checkpoint:", latest_checkpoint)
        return True
    except Exception as e:
        print(f"Error loading parameters: {str(e)}")
        return False
    
    
# Initialize model parameters
try:
    load_parameters()
except Exception as e:
    print(f"Warning: Failed to load model parameters: {str(e)}")