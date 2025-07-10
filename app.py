from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # CORS enabled globally

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "HDP Backend is running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Explicit preflight response with correct headers
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    # Actual prediction logic
    try:
        data = request.json
        input_features = np.array([list(data.values())])
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]
        response = jsonify({'prediction': int(prediction)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
