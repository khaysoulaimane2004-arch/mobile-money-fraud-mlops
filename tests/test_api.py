import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# Mock the model loading before importing the app
mock_booster = MagicMock()
mock_booster.predict.return_value = np.array([0.95])

mock_metadata = {
    "features": [
        "amount", "log_amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "balance_error_orig", "balance_error_dest",
        "orig_balance_emptied", "dest_balance_was_zero",
        "amount_to_balance_ratio", "is_risky_type",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
        "type_PAYMENT", "type_TRANSFER"
    ],
    "threshold": 0.5
}

import xgboost as xgb

with patch.object(xgb.Booster, "load_model", return_value=None), \
     patch("builtins.open", MagicMock()), \
     patch("json.load", return_value=mock_metadata):

    import src.api.main as main_module
    main_module.booster = mock_booster
    main_module.FEATURES = mock_metadata["features"]
    main_module.THRESHOLD = mock_metadata["threshold"]

from fastapi.testclient import TestClient
client = TestClient(main_module.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_schema():
    payload = {
        "type": "TRANSFER",
        "amount": 50000,
        "oldbalanceOrg": 50000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 50000
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "is_fraud" in body
    assert "fraud_probability" in body
    assert "threshold_used" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_schema_validation():
    response = client.post("/predict", json={"type": "TRANSFER"})
    assert response.status_code == 422


def test_predict_all_transaction_types():
    for t in ["PAYMENT", "CASH_IN", "CASH_OUT", "DEBIT", "TRANSFER"]:
        payload = {
            "type": t,
            "amount": 1000,
            "oldbalanceOrg": 5000,
            "newbalanceOrig": 4000,
            "oldbalanceDest": 2000,
            "newbalanceDest": 3000
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200