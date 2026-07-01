"""Tests de base pour l'API de scoring. À étendre au fur et à mesure."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_schema():
    payload = {
        "type": "TRANSFER",
        "amount": 5000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "is_fraud" in body
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
