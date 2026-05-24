"""
predict.py
----------
A thin helper that loads the trained model once and makes discount predictions.
Used by the FastAPI app's /predict_discount endpoint.
"""

import os
import joblib
import pandas as pd

from data_prep import FEATURE_NUMERIC, FEATURE_CATEGORICAL

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "models", "discount_model.joblib")


class DiscountPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run 'python src/train_model.py' first."
            )
        self.model = joblib.load(model_path)

    def predict(self, actual_price, rating, rating_count, main_category):
        """Return the predicted discount percentage for one product."""
        row = pd.DataFrame([{
            "actual_price_num": float(actual_price),
            "rating_num": float(rating),
            "rating_count_num": float(rating_count),
            "main_category": str(main_category),
        }])
        # Keep only the columns the model expects, in the right order
        row = row[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
        pred = float(self.model.predict(row)[0])
        # Clamp to a sensible 0–80% range
        return max(0.0, min(80.0, round(pred, 2)))
