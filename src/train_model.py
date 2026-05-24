"""
train_model.py
--------------
Trains the PREDICTIVE MODEL that forecasts a product's discount percentage.

This covers the assignment's "predictive modeling" requirement:
  - We use a Random Forest regressor (a solid, beginner-friendly model).
  - We measure it with RMSE, MAE and R2 (the metrics the assignment asks for).
  - We save the trained model to models/discount_model.joblib so the API can
    load it later without retraining.

Run:  python src/train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_prep import (
    load_and_clean, FEATURE_NUMERIC, FEATURE_CATEGORICAL, TARGET,
)
from monitoring import save_baseline

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "amazon_sample.csv")
MODEL_DIR = os.path.join(HERE, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "discount_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def build_pipeline():
    """
    A 'pipeline' bundles data-transforming steps + the model into ONE object.
    Benefit: we preprocess training and live data exactly the same way, so we
    can never accidentally treat them differently.
    """
    # Numbers: put them on a comparable scale.
    # Category text: turn each category into 0/1 columns (one-hot encoding),
    # because models need numbers, not words.
    preprocess = ColumnTransformer(transformers=[
        ("num", StandardScaler(), FEATURE_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CATEGORICAL),
    ])

    model = RandomForestRegressor(
        n_estimators=200,   # number of decision trees in the "forest"
        max_depth=12,
        random_state=42,    # makes results repeatable
        n_jobs=-1,          # use all CPU cores
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def main():
    print("Loading and cleaning data...")
    df = load_and_clean(DATA_PATH)

    X = df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y = df[TARGET]

    # Split: train on 80% of products, test how good we are on the unseen 20%.
    # Testing on UNSEEN data is the only honest way to measure a model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} products, testing on {len(X_test)}...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Predict on the test set and score it
    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {"rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 3),
               "n_train": len(X_train), "n_test": len(X_test)}

    print("\n=== Model performance on unseen test data ===")
    print(f"RMSE (avg error, lower is better): {metrics['rmse']}")
    print(f"MAE  (avg absolute error)        : {metrics['mae']}")
    print(f"R2   (1.0 = perfect, 0 = useless): {metrics['r2']}")

    # Save the trained model + the metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")

    # Save training-data averages so the API can detect "drift" later.
    save_baseline(df, FEATURE_NUMERIC)
    print("Saved drift-detection baseline.")


if __name__ == "__main__":
    main()
