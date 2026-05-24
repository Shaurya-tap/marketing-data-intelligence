"""
data_prep.py
------------
Loads the raw e-commerce CSV and CLEANS it so a machine learning model can use it.

Why we need this: the raw data stores numbers as messy text like "₹1,798" and
"56.6%". A model can only do math on real numbers, so we strip out the symbols
and convert everything to proper numeric columns.

This file is imported by train_model.py and app.py — it is the single place
where "what the data looks like" is defined.
"""

import re
import pandas as pd


def _to_number(value):
    """Turn a messy string like '₹1,798' or '56.6%' into a float 1798.0 / 56.6."""
    if pd.isna(value):
        return None
    # Keep only digits and the decimal point, drop ₹ , % spaces etc.
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    if cleaned == "":
        return None
    return float(cleaned)


def load_and_clean(csv_path):
    """
    Read the CSV and return a clean pandas DataFrame.

    Returns a DataFrame with these numeric columns added:
      - actual_price_num
      - discounted_price_num
      - discount_percentage_num   (this is what we PREDICT)
      - rating_num
      - rating_count_num
      - main_category             (just the top-level category, e.g. 'Electronics')
    """
    df = pd.read_csv(csv_path)

    # Convert the messy text columns into clean numbers
    df["actual_price_num"] = df["actual_price"].apply(_to_number)
    df["discounted_price_num"] = df["discounted_price"].apply(_to_number)
    df["discount_percentage_num"] = df["discount_percentage"].apply(_to_number)
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count_num"] = df["rating_count"].apply(_to_number)

    # The category column looks like "Electronics|Headphones|In-Ear".
    # The first piece is the broad category — useful and simple for the model.
    df["main_category"] = df["category"].astype(str).str.split("|").str[0]

    # Drop rows where the important numbers are missing
    needed = [
        "actual_price_num", "discount_percentage_num",
        "rating_num", "rating_count_num", "main_category",
    ]
    df = df.dropna(subset=needed).reset_index(drop=True)

    return df


# The columns the model will use to make its prediction (the "features"),
# and the column it is trying to predict (the "target").
FEATURE_NUMERIC = ["actual_price_num", "rating_num", "rating_count_num"]
FEATURE_CATEGORICAL = ["main_category"]
TARGET = "discount_percentage_num"


if __name__ == "__main__":
    # Quick self-test when run directly
    import os
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "data", "amazon_sample.csv")
    d = load_and_clean(path)
    print(f"Clean rows: {len(d)}")
    print(d[FEATURE_NUMERIC + FEATURE_CATEGORICAL + [TARGET]].head())
