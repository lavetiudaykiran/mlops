import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib
import os

DATA_PATH = "data/raw/california_housing.csv"
MODEL_DIR = "models"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["MedHouseVal", "income_cat"])
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_models():
    # Set tracking URI before any MLflow call
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file:///{tracking_dir.replace(os.sep, '/')}")

    X_train, X_test, y_train, y_test = load_data()
    
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }

    best_model_name = None
    best_rmse = float("inf")
    best_model = None

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_model = model

    # Save best model locally
    best_model_path = os.path.join(MODEL_DIR, f"{best_model_name}.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Best model: {best_model_name} saved at {best_model_path}")

    # Register best model inside a new run
    with mlflow.start_run(run_name="BestModelRegistration"):
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            registered_model_name="CaliforniaHousingModel"
        )

if __name__ == "__main__":
    train_and_log_models()