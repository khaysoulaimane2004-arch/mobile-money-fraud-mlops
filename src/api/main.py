"""
API FastAPI de scoring pour la détection de fraude mobile money.

À FAIRE (semaine 2) :
- Charger le modèle entraîné (models/model.joblib ou via MLflow registry)
- Endpoint POST /predict qui prend une transaction et retourne un score de fraude
- Endpoint GET /health pour les health checks Cloud Run
- Logger chaque prédiction (utile plus tard pour le monitoring de drift)
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Mobile Money Fraud Detection API")

# TODO: charger le vrai modèle au démarrage
# model = joblib.load("models/model.joblib")


class Transaction(BaseModel):
    type: str = Field(..., description="CASH-IN, CASH-OUT, DEBIT, PAYMENT ou TRANSFER")
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    # TODO: appliquer le même feature engineering que dans src/training/features.py
    # puis model.predict_proba(...)
    # Placeholder en attendant le vrai modèle :
    dummy_probability = 0.01
    return PredictionResponse(
        is_fraud=dummy_probability > 0.5,
        fraud_probability=dummy_probability,
    )
