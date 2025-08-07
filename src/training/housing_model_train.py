import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from src.data_preprocessing.housing_data_preprocessing   import load_and_preprocess_housing

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_housing()

# MLflow experiment
mlflow.set_experiment("Housing Regression")

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5)
}

best_model = None
best_score = float("inf")
best_run_id = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")

        if rmse < best_score:
            best_score = rmse
            best_model = model
            best_run_id = run.info.run_id

print(f"Best Model: {best_model.__class__.__name__}, RMSE: {best_score}")
mlflow.register_model(f"runs:/{best_run_id}/model", "BestHousingModel")
