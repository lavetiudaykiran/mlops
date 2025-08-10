# -*- coding: utf-8 -*-
import os
import sys
import threading
import time
from datetime import datetime
from typing import List

# Make sure project root is on path so 'src' imports work
ROOT = os.path.dirname(os.path.abspath(__file__))  # when run as python -m api.app from project root
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, ".."))  # adjust if running from project root

import joblib
import numpy as np
from flask import Flask, request, jsonify
from pydantic import BaseModel, confloat, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server  # optional separate exporter
from src.logger import get_logger
from typing import List

# If you have a separate train module, import; else we'll provide fallback retrain routine
try:
    from src.train import train_and_log_models  # your existing train entrypoint
except Exception:
    train_and_log_models = None

logger = get_logger(__name__)
app = Flask(__name__)

# -------------------------
# Prometheus metrics
# -------------------------
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
LAST_RETRAIN_TIME = Gauge("model_last_retrain_timestamp", "Unix timestamp of last model retrain")

# -------------------------
# Input validation with Pydantic
# -------------------------
# California housing features: 8 floats. Use confloat for numeric constraints if needed.
class PredictRequest(BaseModel):
    features: List[confloat()]  # Just a list of floats

    @validator("features")
    def check_length(cls, v):
        if len(v) != 8:
            raise ValueError("features must contain exactly 8 items")
        return v

# -------------------------
# Model loading utilities
# -------------------------
def get_models_dir():
    # Model is stored under src/models (as you requested)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "src", "models")

def find_latest_model_file(models_dir: str):
    files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    # choose most recently modified
    return max(files, key=os.path.getmtime)

def load_model():
    models_dir = get_models_dir()
    model_path = find_latest_model_file(models_dir)
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

# load initial model on startup
model = None
try:
    model = load_model()
except Exception as e:
    logger.error(f"Failed to load model at startup: {e}")
    model = None

# -------------------------
# Retrain routine (background)
# -------------------------
retrain_lock = threading.Lock()

def retrain_and_reload_model(async_run=True):
    """
    Retrains model by invoking train_and_log_models() if available,
    otherwise runs a simple fallback training routine (if provided).
    After training completes, load latest model into memory.
    """
    def _run():
        with retrain_lock:
            logger.info("Retrain started")
            try:
                # if user-provided training function exists, call it
                if train_and_log_models is not None:
                    train_and_log_models()
                else:
                    # Fallback: implement a minimal retrain if needed (could call your train script instead)
                    logger.info("No external train function found. Please place your training logic in src.train.train_and_log_models.")
                    # Optionally: os.system("python src/train.py")
                    # os.system might block; here we suggest user to integrate their train function.

                # After training finishes, reload latest model
                global model
                try:
                    model = load_model()
                    LAST_RETRAIN_TIME.set(int(time.time()))
                    logger.info("Retrain completed and model reloaded.")
                except Exception as load_err:
                    logger.exception("Retrain finished but failed to load model: %s", load_err)

            except Exception as e:
                logger.exception("Error during retrain: %s", e)
    if async_run:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"status": "retrain started"}
    else:
        _run()
        return {"status": "retrain completed"}

# -------------------------
# Filesystem watcher (optional)
# -------------------------
WATCHER_THREAD = None
WATCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

def start_filesystem_watcher():
    """
    Starts a watchdog observer that triggers retrain when new file appears under data/raw.
    Requires watchdog package.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        logger.warning("watchdog not installed; filesystem watcher disabled. Install watchdog to enable it.")
        return

    class NewFileHandler(FileSystemEventHandler):
        def on_created(self, event):
            # Trigger on creation of new files only (not directories)
            if not event.is_directory:
                logger.info(f"Detected new file: {event.src_path} - triggering retrain")
                retrain_and_reload_model(async_run=True)

    if not os.path.exists(WATCH_DIR):
        os.makedirs(WATCH_DIR, exist_ok=True)

    observer = Observer()
    handler = NewFileHandler()
    observer.schedule(handler, WATCH_DIR, recursive=True)
    observer.daemon = True
    observer.start()
    logger.info(f"Started filesystem watcher on {WATCH_DIR}")

# -------------------------
# Flask endpoints
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    PREDICTION_COUNT.inc()
    start = time.time()
    try:
        # Validate input
        try:
            payload = PredictRequest(**request.get_json(force=True))
        except Exception as verr:
            PREDICTION_ERRORS.inc()
            logger.error(f"Validation error: {verr}")
            return jsonify({"error": "Invalid input", "details": str(verr.errors())}), 400

        features = np.array(payload.features).reshape(1, -1)

        # Make sure model is loaded
        global model
        if model is None:
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "Model not loaded"}), 500

        pred = model.predict(features)
        latency = time.time() - start
        PREDICTION_LATENCY.observe(latency)
        logger.info(f"Prediction: {pred.tolist()} for input={payload.features}")
        return jsonify({"prediction": pred.tolist()})
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/retrain", methods=["POST"])
def retrain_endpoint():
    """
    Secure retrain endpoint. In production protect this behind auth.
    POST body (optional): { "async": true }
    """
    auth = request.headers.get("X-RETRAIN-TOKEN")
    # You should set environment variable RETRAIN_TOKEN for simple protection
    token = os.environ.get("RETRAIN_TOKEN")
    if token and auth != token:
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    async_run = body.get("async", True)
    result = retrain_and_reload_model(async_run=async_run)
    return jsonify(result)

@app.route("/metrics")
def metrics():
    # Return prometheus metrics
    resp = generate_latest()
    return (resp, 200, {"Content-Type": CONTENT_TYPE_LATEST})

# -------------------------
# Startup
# -------------------------
if __name__ == "__main__":
    # Optionally start prometheus metrics server on a separate port:
    # start_http_server(8000)  # then Prometheus can scrape :8000/metrics
    # Start FS watcher
    start_filesystem_watcher()
    # Start flask
    logger.info("Starting API")
    # set LAST_RETRAIN_TIME if model exists
    if model is not None:
        LAST_RETRAIN_TIME.set(int(os.path.getmtime(find_latest_model_file(get_models_dir()))))
    app.run(host="0.0.0.0", port=5000)
