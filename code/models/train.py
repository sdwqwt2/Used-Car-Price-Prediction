"""
Stage 2: Model Engineering
===========================
Loads train/test data produced by Stage 1, builds features, trains a
RandomForestRegressor, evaluates it, logs metrics + model to MLflow,
and packages the fitted pipeline (preprocessing + model) to a single
joblib file for the deployment stage.

Usage:
    python code/models/train.py
"""

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

PROCESSED_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

TARGET_COL = "selling_price"
NUMERIC_FEATURES = ["km_driven", "car_age"]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "brand"]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "used-car-price-prediction")


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[train] Loading processed data from {PROCESSED_DIR}")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET_COL]
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df[TARGET_COL]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        pipeline = build_pipeline()
        print("[train] Fitting RandomForestRegressor pipeline...")
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"[train] MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 2,
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        joblib.dump(pipeline, MODEL_PATH)
        print(f"[train] Saved packaged model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
