# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import threading
import time
import logging
from datetime import datetime
from typing import List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # api/ folder
project_root = os.path.dirname(current_dir)  # project root folder
sys.path.insert(0, project_root)

import joblib
import numpy as np
from flask import Flask, request, jsonify
from pydantic import BaseModel, confloat, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import your logger (assumed to be at src/logger.py)
from src.logger import get_logger

logger = get_logger(__name__)
app = Flask(__name__)

# Prometheus metrics
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
LAST_RETRAIN_TIME = Gauge("model_last_retrain_timestamp", "Unix timestamp of last model retrain")
RETRAIN_COUNT = Counter("retrain_requests_total", "Total retrain requests")
# Increment every time prediction endpoint is called
REQUEST_COUNT.inc()
# Data and model directories relative to project root
DATA_DIR = os.path.join(project_root, "data")
MODEL_DIR = os.path.join(project_root, "model")  # singular 'model' as per your note

# Input validation with Pydantic
class PredictRequest(BaseModel):
    features: List[confloat()]

    @validator("features")
    def check_length(cls, v):
        if len(v) != 8:
            raise ValueError("features must contain exactly 8 items")
        return v

# Model loading utility
def get_latest_model_path():
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError(f"No .pkl model files found in {MODEL_DIR}")
    full_paths = [os.path.join(MODEL_DIR, f) for f in model_files]
    latest_model = max(full_paths, key=os.path.getmtime)
    return latest_model

def load_model():
    model_path = get_latest_model_path()
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

# Load model initially
model = None
try:
    model = load_model()
    logger.info("Model loaded successfully on startup")
    if os.path.exists(get_latest_model_path()):
        LAST_RETRAIN_TIME.set(int(os.path.getmtime(get_latest_model_path())))
except Exception as e:
    logger.error(f"Failed to load model at startup: {e}")
    model = None

# Retrain control
retrain_lock = threading.Lock()
retrain_status = {
    "running": False,
    "last_result": None,
    "start_time": None,
    "end_time": None,
    "success": None,
    "message": None
}

def retrain_model_background():
    global model, retrain_status
    with retrain_lock:
        try:
            retrain_status.update({
                "running": True,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "success": None,
                "message": "Retraining in progress..."
            })
            logger.info("Starting retraining...")

            # Run train.py as subprocess from project root
            train_script = os.path.join(project_root, "src", "train.py")
            if not os.path.exists(train_script):
                raise FileNotFoundError(f"Training script not found: {train_script}")

            # Run training script (you can modify train.py to take data/model paths as args if needed)
            result = subprocess.run(
                [sys.executable, train_script],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=project_root
            )

            if result.returncode == 0:
                logger.info("Training script completed successfully")
                logger.debug(f"Training output:\n{result.stdout}")

                try:
                    model = load_model()
                    LAST_RETRAIN_TIME.set(int(time.time()))
                    retrain_status.update({
                        "running": False,
                        "success": True,
                        "message": "Model retrained and loaded successfully",
                        "end_time": datetime.now().isoformat(),
                        "stdout": result.stdout.strip()
                    })
                    logger.info("Model reloaded after retraining")
                except Exception as e:
                    msg = f"Training succeeded but failed to load model: {e}"
                    logger.error(msg)
                    retrain_status.update({
                        "running": False,
                        "success": False,
                        "message": msg,
                        "end_time": datetime.now().isoformat()
                    })
            else:
                msg = f"Training failed with return code {result.returncode}"
                logger.error(msg)
                logger.error(f"Training stderr:\n{result.stderr}")
                retrain_status.update({
                    "running": False,
                    "success": False,
                    "message": msg,
                    "end_time": datetime.now().isoformat(),
                    "stderr": result.stderr.strip()
                })

        except subprocess.TimeoutExpired:
            msg = "Training timed out after 10 minutes"
            logger.error(msg)
            retrain_status.update({
                "running": False,
                "success": False,
                "message": msg,
                "end_time": datetime.now().isoformat()
            })

        except Exception as e:
            msg = f"Unexpected error during retraining: {e}"
            logger.exception(msg)
            retrain_status.update({
                "running": False,
                "success": False,
                "message": msg,
                "end_time": datetime.now().isoformat()
            })

@app.route("/predict", methods=["POST"])
def predict():
    PREDICTION_COUNT.inc()
    start_time = time.time()

    try:
        try:
            payload = PredictRequest(**request.get_json(force=True))
        except Exception as ve:
            PREDICTION_ERRORS.inc()
            logger.error(f"Validation error: {ve}")
            return jsonify({"error": "Invalid input", "details": str(ve)}), 400

        features = np.array(payload.features).reshape(1, -1)

        if model is None:
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "Model not loaded"}), 500

        prediction = model.predict(features)
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        logger.info(f"Prediction: {prediction.tolist()} for input: {payload.features}")
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/retrain", methods=["POST"])
def retrain():
    RETRAIN_COUNT.inc()

    if retrain_status["running"]:
        return jsonify({
            "error": "Retraining already in progress",
            "start_time": retrain_status.get("start_time")
        }), 409

    auth_token = request.headers.get("X-RETRAIN-TOKEN")
    expected_token = os.environ.get("RETRAIN_TOKEN")
    if expected_token and auth_token != expected_token:
        return jsonify({"error": "Unauthorized"}), 401

    thread = threading.Thread(target=retrain_model_background, daemon=True)
    thread.start()

    return jsonify({
        "status": "Retraining started",
        "start_time": retrain_status["start_time"]
    })

@app.route("/retrain/status", methods=["GET"])
def retrain_status_endpoint():
    last_retrain_timestamp = 0
    try:
        last_retrain_timestamp = LAST_RETRAIN_TIME._value._value if hasattr(LAST_RETRAIN_TIME._value, '_value') else 0
    except:
        last_retrain_timestamp = 0

    return jsonify({
        "retrain_running": retrain_status["running"],
        "start_time": retrain_status.get("start_time"),
        "end_time": retrain_status.get("end_time"),
        "success": retrain_status.get("success"),
        "message": retrain_status.get("message"),
        "last_retrain_timestamp": last_retrain_timestamp,
        "last_retrain_time": datetime.fromtimestamp(last_retrain_timestamp).isoformat() if last_retrain_timestamp > 0 else None,
        "last_result": retrain_status.get("last_result")
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_dir_exists": os.path.exists(MODEL_DIR),
        "model_path": get_latest_model_path() if model else None,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    logger.info(f"Starting API server...")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Model loaded: {model is not None}")

    app.run(host="0.0.0.0", port=5000)
