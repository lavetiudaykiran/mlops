from flask import Flask, request, jsonify
import joblib
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)

# Load the model and log it
try:
    model = joblib.load("models/linear_model.pkl")
    logger.info("Model loaded successfully from models/linear_model.pkl")
except Exception as e:
    logger.exception("Failed to load model.")
    raise e

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        logger.info(f"Received prediction request with data: {data}")

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)

        logger.info(f"Prediction result: {prediction.tolist()}")
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(debug=True)
