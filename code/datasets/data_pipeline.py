"""
Stage 1: Data Engineering
=========================
Loads raw used-car listing data, cleans it (missing values + outliers),
and splits it into train/test sets.

Expected raw dataset: CarDekho "Used Car Price Prediction" style CSV placed at
data/raw/used_cars.csv with (at least) these columns:

    name, year, selling_price, km_driven, fuel, seller_type,
    transmission, owner

(This is the common Kaggle "Vehicle dataset" / CarDekho schema. If your
CSV has different column names, adjust COLUMN_MAP below.)

Usage:
    python code/datasets/data_pipeline.py
"""

import os
import re
import pandas as pd
import numpy as np

RAW_PATH = os.getenv("RAW_DATA_PATH", "data/raw/used_cars.csv")
PROCESSED_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

TARGET_COL = "selling_price"
TEST_SIZE = 0.2
RANDOM_STATE = 42

COLUMN_MAP = {
    "fuel_type": "fuel",
    "transmission_type": "transmission",
    "vehicle_age": "car_age",
}


def load_data(path: str) -> pd.DataFrame:
    print(f"[data_pipeline] Loading raw data from {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    print(f"[data_pipeline] Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def extract_brand(name_series: pd.Series) -> pd.Series:
    return name_series.astype(str).apply(lambda x: re.split(r"\s+", x.strip())[0] if x else "unknown")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only columns we actually use downstream; ignore anything extra.
    expected_cols = [
        "brand", "car_age", "selling_price", "km_driven",
        "fuel", "seller_type", "transmission",
    ]
    missing_expected = [c for c in expected_cols if c not in df.columns]
    if missing_expected:
        raise ValueError(
            f"[data_pipeline] Raw data is missing expected columns: {missing_expected}. "
            f"Update COLUMN_MAP in code/datasets/data_pipeline.py to match your CSV."
        )
    df = df[expected_cols]

    # --- Missing values ---
    # Numeric columns: impute with median
    for col in ["car_age", "selling_price", "km_driven"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns: impute with mode
    for col in ["fuel", "seller_type", "transmission", "brand"]:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Drop rows with non-positive price or mileage (invalid entries)
    df = df[(df["selling_price"] > 0) & (df["km_driven"] >= 0)]

    # --- Outlier removal (IQR method on numeric columns) ---
    for col in ["selling_price", "km_driven", "car_age"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - len(df)
        if removed:
            print(f"[data_pipeline] Removed {removed} outlier rows based on '{col}'")

    df = df.reset_index(drop=True)
    print(f"[data_pipeline] Cleaned data: {len(df)} rows remaining")
    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cutoff = int(len(shuffled) * (1 - test_size))
    train_df, test_df = shuffled.iloc[:cutoff], shuffled.iloc[cutoff:]
    print(f"[data_pipeline] Split into train={len(train_df)} / test={len(test_df)}")
    return train_df, test_df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_data(RAW_PATH)
    df = clean_data(df)
    train_df, test_df = split_data(df)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"[data_pipeline] Saved train -> {TRAIN_PATH}")
    print(f"[data_pipeline] Saved test  -> {TEST_PATH}")


if __name__ == "__main__":
    main()
