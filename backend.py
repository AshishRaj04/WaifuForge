from flask import Flask, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from engine import generate_name_sample
from face_engine import generate_face

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server is running"})

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        # Generate character name
        generated_name = generate_name_sample(temperature=0.3, load=True)
        
        # Generate face image
        face_image = generate_face()
        if face_image is None:
            raise Exception("Failed to generate face image")
            
        # Convert numpy array to base64 string
        try:
            # Convert to BGR for OpenCV
            face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            
            # Encode image to PNG
            success, buffer = cv2.imencode('.png', face_image_bgr)
            if not success:
                raise Exception("Failed to encode image")
                
            # Convert to base64
            face_base64 = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as img_error:
            raise Exception(f"Image processing error: {str(img_error)}")

        return jsonify({
            "status": "success",
            "name": generated_name,
            "face": face_base64
        })
        
    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Initialize any required models or resources here
    print("Starting server...")
    app.run(debug=True, port=5000)