from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = np.array([list(data.values())])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
