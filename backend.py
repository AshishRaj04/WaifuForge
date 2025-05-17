from flask import Flask, jsonify
from flask_cors import CORS
from engine import generate_name_sample

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server is running"})


@app.route('/api/generate-name', methods=['POST'])
def generate_name():
    try:
        data = generate_name_sample(temperature=0.3 , load=True)
        
        response = {
            "status": "success",
            "name": data  
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)