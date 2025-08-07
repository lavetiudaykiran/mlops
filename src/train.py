import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, model_name)
        print(f"{model_name} RMSE: {rmse}")

if __name__ == "__main__":
    df = pd.read_csv("data/raw_data.csv")
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_log_model(LinearRegression(), "LinearRegression", X_train, X_test, y_train, y_test)
    train_and_log_model(DecisionTreeRegressor(), "DecisionTree", X_train, X_test, y_train, y_test)
