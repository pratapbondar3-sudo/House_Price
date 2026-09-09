import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Locate linear.pkl relative to the project directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
if not os.path.exists(MODEL_PATH):
    # Fallback if app is running from api/ subdirectory
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'linear.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Features expected by your scikit-learn model
FEATURE_NAMES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality"
]

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "expected_features": FEATURE_NAMES
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    try:
        # Extract features in the correct order
        features = [float(data[feature]) for feature in FEATURE_NAMES]
    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {str(e)}"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "All feature values must be numeric"}), 400

    prediction = model.predict(np.array([features]))
    return jsonify({
        "prediction": float(prediction[0])
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
