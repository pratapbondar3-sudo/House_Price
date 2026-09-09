import os
import pickle
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load model relative to file location
model_path = os.path.join(os.path.dirname(__file__), "linear.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Categorical mapping for Neighborhood_Quality
QUALITY_MAPPING = {
    "poor": 1.0,
    "fair": 2.0,
    "average": 3.0,
    "good": 4.0,
    "excellent": 5.0,
}


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "online",
            "neighborhood_quality_options": list(QUALITY_MAPPING.keys()),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        # Resolve Neighborhood_Quality (handles both string category or raw integer/float)
        raw_quality = data.get("Neighborhood_Quality")
        if isinstance(raw_quality, str):
            quality_val = QUALITY_MAPPING.get(raw_quality.strip().lower())
            if quality_val is None:
                return (
                    jsonify(
                        {
                            "error": f"Invalid Neighborhood_Quality. Choose from: {list(QUALITY_MAPPING.keys())}"
                        }
                    ),
                    400,
                )
        else:
            quality_val = float(raw_quality)

        features = [
            float(data["Square_Footage"]),
            float(data["Num_Bedrooms"]),
            float(data["Num_Bathrooms"]),
            float(data["Year_Built"]),
            float(data["Lot_Size"]),
            float(data["Garage_Size"]),
            quality_val,
        ]

        prediction = model.predict(np.array([features]))[0]
        return jsonify({"prediction": round(float(prediction), 2)})

    except (KeyError, TypeError, ValueError) as err:
        return jsonify({"error": f"Missing or invalid input: {str(err)}"}), 400


if __name__ == "__main__":
    app.run(debug=True)
