import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)
app = Flask(__name__)

# Dynamically load the latest/best model from the models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, "src", "models")
model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

if not model_files:
    raise FileNotFoundError("No model files found in 'models' directory.")

# Pick the latest model based on modification time
latest_model_file = max(
    model_files, key=lambda f: os.path.getmtime(os.path.join(models_dir, f))
)
model_path = os.path.join(models_dir, latest_model_file)

logger.info(f"Loading model: {model_path}")
model = joblib.load(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = data.get("features")
    if not features:
        return jsonify({"error": "No features provided"}), 400

    try:
        features_np = np.array(features).reshape(1, -1)
        prediction = model.predict(features_np)[0]
        logger.info(f"Prediction: {prediction} for input: {features}")
        return jsonify({"prediction": prediction})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
