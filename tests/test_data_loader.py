# test_data_loader.py
import os
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing
from src.load_data import load_and_save_data  # change to your file name

def test_default_path_saves_file(tmp_path, monkeypatch):
    # Monkeypatch project root to tmp_path
    test_root = tmp_path / "project"
    test_root.mkdir()
    monkeypatch.chdir(test_root)

    load_and_save_data()

    expected_file = test_root / "data" / "california_housing.csv"
    assert expected_file.exists(), "Data file should be created"

    df = pd.read_csv(expected_file)
    assert "MedInc" in df.columns
    assert "income_cat" in df.columns
    assert not df.empty, "DataFrame should not be empty"


def test_custom_path_saves_file(tmp_path):
    custom_path = tmp_path / "custom_data.csv"
    load_and_save_data(data_path=str(custom_path))

    assert custom_path.exists(), "Custom path file should be created"

    df = pd.read_csv(custom_path)
    assert "MedInc" in df.columns
    assert "income_cat" in df.columns


def test_income_cat_bins():
    # Just fetch directly to check binning logic
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df["income_cat"] = pd.cut(
        df["MedInc"],
        bins=[0., 1.5, 3.0, 4.5, 6., float("inf")],
        labels=[1, 2, 3, 4, 5]
    )

    assert df["income_cat"].notnull().all(), "All rows should have an income category"
    assert set(df["income_cat"].unique()) <= {1, 2, 3, 4, 5}
