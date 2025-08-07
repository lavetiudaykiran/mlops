from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import logging
import pandas as pd
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from src.domain.iris_input import IrisInput



# Set up logging to file
logging.basicConfig(
    filename='logs/prediction.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

# Load model
MODEL_NAME = "BestIrisModel"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/Production")

app = FastAPI(title="ML Prediction API")

Instrumentator().instrument(app).expose(app)

@app.post("/predict")
def predict(data: IrisInput):
    try:
        input_data = [[
            data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
        ]]
        df = pd.DataFrame(input_data, columns=[
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ])
        prediction = model.predict(df)
        result = int(prediction[0])

        # Log request and response
        logging.info(f"Input: {data.dict()} | Prediction: {result}")

        return {"prediction": result}

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain_iris")
def retrain_on_new_data(file: UploadFile = File(...)):
    try:
        # Save new training data
        filepath = f"data/new_data_{datetime.now().isoformat()}.csv"
        with open(filepath, "wb") as f:
            f.write(file.file.read())

        # Trigger re-training script
        subprocess.run(["python", "src/training/iris_model_train.py"], check=True)

        return {"status": "Retraining started successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
