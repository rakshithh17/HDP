from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes and methods

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "HDP Backend is running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Preflight request
        response = jsonify({'message': 'CORS preflight passed'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    data = request.json
    input_features = np.array([list(data.values())])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    
    response = jsonify({'prediction': int(prediction)})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
