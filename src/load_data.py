import pandas as pd
from src.logger import get_logger
from sklearn.datasets import fetch_california_housing

logger = get_logger(__name__)

def load_and_save_data(path="data/raw_data.csv"):
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv(path, index=False)
    logger.info(f"Data saved to {path}")

if __name__ == "__main__":
    load_and_save_data()
