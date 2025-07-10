from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Full CORS support

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "HDP Backend is running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Preflight request handling for CORS
        return jsonify({}), 200

    data = request.json
    input_features = np.array([list(data.values())])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
