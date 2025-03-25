import unittest
import joblib
import numpy as np
import pandas as pd
import os

# Load model and scaler
model = joblib.load("./model/a3_car_price.pkl")
scaler = joblib.load("./model/scaler.dump")

class TestCarPriceModel(unittest.TestCase):

    def setUp(self):
        self.sample_input = pd.DataFrame({
            'engine': [1500],
            'max_power': [85],
            'mileage': [20],
            'year': [2017]
        })

    def test_model_input_format(self):
        """Test if the model accepts the expected input format"""
        try:
            scaled = scaler.transform(self.sample_input)
            _ = model.predict(scaled)
            passed = True
        except Exception as e:
            passed = False
        self.assertTrue(passed, "Model failed to accept input format")

    def test_model_output_shape(self):
        """Test if the output of the model has shape (1,)"""
        scaled = scaler.transform(self.sample_input)
        pred = model.predict(scaled)
        self.assertEqual(pred.shape, (1,), "Model output shape is not (1,)")

if __name__ == '__main__':
    unittest.main()
