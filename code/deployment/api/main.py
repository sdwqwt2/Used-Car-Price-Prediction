"""
Stage 3: Deployment — Model API
================================
FastAPI service that loads the packaged sklearn pipeline and exposes
a /predict endpoint for the Streamlit app to call.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")

app = FastAPI(title="Used Car Price Prediction API")

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


class CarFeatures(BaseModel):
    km_driven: float = Field(..., ge=0, example=45000)
    car_age: float = Field(..., ge=0, example=5)
    fuel: str = Field(..., example="Petrol")
    seller_type: str = Field(..., example="Individual")
    transmission: str = Field(..., example="Manual")
    brand: str = Field(..., example="Maruti")


class PredictionResponse(BaseModel):
    predicted_price: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CarFeatures):
    try:
        model = get_model()
        row = pd.DataFrame([features.dict()])
        pred = model.predict(row)[0]
        return PredictionResponse(predicted_price=float(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
