import json
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Mobile Money Fraud Detection API")

# Load model from JSON format (version-independent)
booster = xgb.Booster()
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "model.json")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

booster.load_model(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURES  = metadata["features"]
THRESHOLD = metadata.get("threshold", 0.5)

print(f"Model loaded ✓")
print(f"Features  : {len(FEATURES)}")
print(f"Threshold : {THRESHOLD}")


class Transaction(BaseModel):
    type: str = Field(..., description="CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER")
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float


def build_features(t: Transaction) -> pd.DataFrame:
    # FIX: use model_dump() instead of dict() for Pydantic v2
    d = t.model_dump()

    d['balance_error_orig']      = d['oldbalanceOrg'] - d['amount'] - d['newbalanceOrig']
    d['balance_error_dest']      = d['oldbalanceDest'] + d['amount'] - d['newbalanceDest']
    d['orig_balance_emptied']    = int(d['newbalanceOrig'] == 0)
    d['dest_balance_was_zero']   = int(d['oldbalanceDest'] == 0)
    d['amount_to_balance_ratio'] = d['amount'] / (d['oldbalanceOrg'] + 1)

    # FIX: guard against negative amounts before log transform
    d['log_amount'] = np.log1p(max(d['amount'], 0))

    d['is_risky_type'] = int(d['type'] in ['CASH_OUT', 'TRANSFER'])

    for t_name in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
        d[f'type_{t_name}'] = int(d['type'] == t_name)

    return pd.DataFrame([d])[FEATURES]


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "threshold": THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    features = build_features(transaction)
    dmatrix  = xgb.DMatrix(features)
    fraud_probability = float(booster.predict(dmatrix)[0])
    is_fraud = fraud_probability >= THRESHOLD

    return PredictionResponse(
        is_fraud          = is_fraud,
        fraud_probability = round(fraud_probability, 4),
        threshold_used    = THRESHOLD
    )