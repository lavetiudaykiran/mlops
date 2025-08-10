from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_and_save_data(data_path="data/raw/california_housing.csv"):
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Simple preprocessing example: add median_income category (optional)
    df["income_cat"] = pd.cut(
        df["MedInc"],
        bins=[0., 1.5, 3.0, 4.5, 6., float("inf")],
        labels=[1, 2, 3, 4, 5]
    )

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Raw data saved to {data_path}")

if __name__ == "__main__":
    load_and_save_data()