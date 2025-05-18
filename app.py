from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://thatdelta.github.io"}})

# Load model and scaler once on server start
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json(silent=True)
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing "features" in request'}), 400

    features = data['features']
    if not isinstance(features, list) or len(features) != 21:
        return jsonify({'error': 'Invalid input features: must be a list of 21 values'}), 400

    try:
        # Convert to numpy array and reshape for scaler/model
        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]  # probability of class '1' (at risk)
        pred = int(prob > 0.5)  # binary prediction with 0.5 threshold

        return jsonify({'probability_at_risk': float(prob), 'prediction': pred})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
