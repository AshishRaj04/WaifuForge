from flask import Flask, jsonify
from flask_cors import CORS
from engine import generate_name_sample
from face_engine import generate_face

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"})


@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        generated_name = generate_name_sample(temperature=0.3, load=True)
        generated_face = generate_face()

        face_data = generated_face.tolist()

        return jsonify({"status": "success", "name": generated_name, "face": face_data})

    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True, port=5000)
