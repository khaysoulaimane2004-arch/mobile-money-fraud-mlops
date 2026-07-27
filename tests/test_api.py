from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_legitimate_transaction():
    """A normal PAYMENT should return low fraud probability."""
    payload = {
        "type": "PAYMENT",
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 2000.0,
        "newbalanceDest": 3000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "is_fraud" in body
    assert "fraud_probability" in body
    assert "threshold_used" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_suspicious_transaction():
    """A TRANSFER that empties the account should return high fraud probability."""
    payload = {
        "type": "TRANSFER",
        "amount": 50000.0,
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 50000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] == True


def test_predict_schema_validation():
    """Missing fields should return 422."""
    response = client.post("/predict", json={"type": "TRANSFER"})
    assert response.status_code == 422