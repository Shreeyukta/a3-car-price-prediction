import pytest
import joblib
import pandas as pd
import numpy as np
import os

model = joblib.load("app/model/a3_car_price.pkl")
scaler = joblib.load("app/model/scaler.dump")

sample_input = pd.DataFrame({
    'engine': [1500],
    'max_power': [85],
    'mileage': [20],
    'year': [2017]
})


def test_model_accepts_input():
    """Test if the model accepts input and does not throw an error"""
    try:
        scaled = scaler.transform(sample_input)
        model.predict(scaled)
        passed = True
    except Exception as e:
        passed = False
    assert passed, "Model failed to accept input format"


def test_model_output_shape():
    """Test if the model output shape is (1,)"""
    scaled = scaler.transform(sample_input)
    prediction = model.predict(scaled)
    assert prediction.shape == (1,), f"Expected shape (1,), but got {prediction.shape}"
