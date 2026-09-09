import os
import pickle
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Safely locate linear.pkl in parent directory or same directory
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR.parent / "Linear.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = CURRENT_DIR / "Linear.pkl"

with open(MODEL_PATH, "rb") as f:import os
import pickle
from pathlib import Path

# Absolute path to where this script lives
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "Linear.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = CURRENT_DIR.parent / "Linear.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as err:
    model = None
    load_error = str(err)
    model = pickle.load(f)

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
def index():
    return jsonify({"status": "ready", "features": FEATURE_NAMES})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    try:
        features = [float(data[k]) for k in FEATURE_NAMES]
        prediction = model.predict([features])
        return jsonify({"prediction": float(prediction[0])})
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
