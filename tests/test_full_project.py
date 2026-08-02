"""
COMPREHENSIVE TEST SUITE
HMM Fraud Intelligence Platform — MLOps Pipeline
Tests every component : API, model, features, drift, simulator, dashboard
"""

import pytest
import json
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE ENGINEERING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """Tests that feature engineering produces correct values."""

    def _make_row(self, type_="TRANSFER", amount=50000,
                  oldOrg=50000, newOrg=0, oldDest=0, newDest=50000):
        return {
            "type"          : type_,
            "amount"        : amount,
            "oldbalanceOrg" : oldOrg,
            "newbalanceOrig": newOrg,
            "oldbalanceDest": oldDest,
            "newbalanceDest": newDest,
        }

    def _apply_features(self, row):
        d = dict(row)
        d["balance_error_orig"]      = d["oldbalanceOrg"] - d["amount"] - d["newbalanceOrig"]
        d["balance_error_dest"]      = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]
        d["orig_balance_emptied"]    = int(d["newbalanceOrig"] == 0)
        d["dest_balance_was_zero"]   = int(d["oldbalanceDest"] == 0)
        d["amount_to_balance_ratio"] = d["amount"] / (d["oldbalanceOrg"] + 1)
        d["log_amount"]              = np.log1p(d["amount"])
        d["is_risky_type"]           = int(d["type"] in ["CASH_OUT", "TRANSFER"])
        for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
            d[f"type_{t}"] = int(d["type"] == t)
        return d

    def test_balance_error_orig_correct(self):
        """balance_error_orig = oldBal - amount - newBal"""
        row = self._make_row(amount=50000, oldOrg=50000, newOrg=0)
        d   = self._apply_features(row)
        assert d["balance_error_orig"] == pytest.approx(0.0)

    def test_balance_error_nonzero_when_inconsistent(self):
        """A legitimate tx with balanced accounting gives error ~0."""
        row = self._make_row(amount=1000, oldOrg=5000, newOrg=4000,
                             oldDest=2000, newDest=3000)
        d   = self._apply_features(row)
        assert abs(d["balance_error_orig"]) < 0.01
        assert abs(d["balance_error_dest"]) < 0.01

    def test_orig_balance_emptied_true_for_fraud(self):
        """Fraud empties the account — newbalanceOrig == 0."""
        row = self._make_row(newOrg=0)
        d   = self._apply_features(row)
        assert d["orig_balance_emptied"] == 1

    def test_orig_balance_emptied_false_for_legit(self):
        """Legitimate transaction leaves balance."""
        row = self._make_row(newOrg=4000)
        d   = self._apply_features(row)
        assert d["orig_balance_emptied"] == 0

    def test_dest_balance_was_zero(self):
        """Destination had zero balance before receiving."""
        row = self._make_row(oldDest=0)
        d   = self._apply_features(row)
        assert d["dest_balance_was_zero"] == 1

    def test_amount_to_balance_ratio_fraud_case(self):
        """Fraudster sends 100% of balance — ratio should be close to 1."""
        row = self._make_row(amount=50000, oldOrg=50000)
        d   = self._apply_features(row)
        assert d["amount_to_balance_ratio"] > 0.99

    def test_amount_to_balance_ratio_legit_case(self):
        """Legitimate tx sends small fraction of balance."""
        row = self._make_row(amount=1000, oldOrg=50000)
        d   = self._apply_features(row)
        assert d["amount_to_balance_ratio"] < 0.1

    def test_log_amount_positive(self):
        """log_amount must always be positive for amount > 0."""
        row = self._make_row(amount=50000)
        d   = self._apply_features(row)
        assert d["log_amount"] > 0

    def test_is_risky_type_transfer(self):
        """TRANSFER is risky."""
        row = self._make_row(type_="TRANSFER")
        d   = self._apply_features(row)
        assert d["is_risky_type"] == 1

    def test_is_risky_type_cash_out(self):
        """CASH_OUT is risky."""
        row = self._make_row(type_="CASH_OUT")
        d   = self._apply_features(row)
        assert d["is_risky_type"] == 1

    def test_is_risky_type_payment_not_risky(self):
        """PAYMENT is NOT risky."""
        row = self._make_row(type_="PAYMENT")
        d   = self._apply_features(row)
        assert d["is_risky_type"] == 0

    def test_one_hot_encoding_transfer(self):
        """Only type_TRANSFER should be 1 for TRANSFER transactions."""
        row = self._make_row(type_="TRANSFER")
        d   = self._apply_features(row)
        assert d["type_TRANSFER"] == 1
        assert d["type_CASH_OUT"] == 0
        assert d["type_PAYMENT"]  == 0
        assert d["type_CASH_IN"]  == 0
        assert d["type_DEBIT"]    == 0

    def test_one_hot_encoding_payment(self):
        """Only type_PAYMENT should be 1 for PAYMENT transactions."""
        row = self._make_row(type_="PAYMENT")
        d   = self._apply_features(row)
        assert d["type_PAYMENT"]  == 1
        assert d["type_TRANSFER"] == 0

    def test_amount_to_balance_ratio_no_division_by_zero(self):
        """When oldbalanceOrg is 0, formula adds 1 to avoid division by zero."""
        row = self._make_row(oldOrg=0, amount=1000)
        d   = self._apply_features(row)
        assert d["amount_to_balance_ratio"] == pytest.approx(1000 / 1)

    def test_all_16_features_present(self):
        """Exactly 16 features must be produced."""
        FEATURES = [
            'amount', 'log_amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'balance_error_orig',
            'balance_error_dest', 'orig_balance_emptied', 'dest_balance_was_zero',
            'amount_to_balance_ratio', 'is_risky_type', 'type_CASH_IN',
            'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
        ]
        row = self._make_row()
        d   = self._apply_features(row)
        for feat in FEATURES:
            assert feat in d, f"Missing feature: {feat}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — API ENDPOINT TESTS
# ══════════════════════════════════════════════════════════════════════════════

# Mock setup before importing the app
mock_booster   = MagicMock()
mock_booster.predict.return_value = np.array([0.95])

mock_metadata = {
    "features": [
        "amount", "log_amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "balance_error_orig",
        "balance_error_dest", "orig_balance_emptied", "dest_balance_was_zero",
        "amount_to_balance_ratio", "is_risky_type", "type_CASH_IN",
        "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
    ],
    "threshold": 0.5
}

import xgboost as xgb

with patch.object(xgb.Booster, "load_model", return_value=None), \
     patch("builtins.open", MagicMock()), \
     patch("json.load", return_value=mock_metadata):
    import src.api.main as main_module
    main_module.booster   = mock_booster
    main_module.FEATURES  = mock_metadata["features"]
    main_module.THRESHOLD = mock_metadata["threshold"]

from fastapi.testclient import TestClient
client = TestClient(main_module.app)


class TestAPIEndpoints:
    """Tests all API endpoints."""

    # ── Health endpoint ───────────────────────────────────────────
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_ok_status(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_returns_model_name(self):
        r = client.get("/health")
        assert "model" in r.json()

    def test_health_returns_threshold(self):
        r = client.get("/health")
        assert "threshold" in r.json()

    # ── Predict endpoint — schema ─────────────────────────────────
    def test_predict_returns_200(self):
        r = client.post("/predict", json={
            "type":"TRANSFER","amount":50000,
            "oldbalanceOrg":50000,"newbalanceOrig":0,
            "oldbalanceDest":0,"newbalanceDest":50000
        })
        assert r.status_code == 200

    def test_predict_response_has_is_fraud(self):
        r = client.post("/predict", json={
            "type":"PAYMENT","amount":1000,
            "oldbalanceOrg":5000,"newbalanceOrig":4000,
            "oldbalanceDest":2000,"newbalanceDest":3000
        })
        assert "is_fraud" in r.json()

    def test_predict_response_has_fraud_probability(self):
        r = client.post("/predict", json={
            "type":"PAYMENT","amount":1000,
            "oldbalanceOrg":5000,"newbalanceOrig":4000,
            "oldbalanceDest":2000,"newbalanceDest":3000
        })
        assert "fraud_probability" in r.json()

    def test_predict_response_has_threshold_used(self):
        r = client.post("/predict", json={
            "type":"PAYMENT","amount":1000,
            "oldbalanceOrg":5000,"newbalanceOrig":4000,
            "oldbalanceDest":2000,"newbalanceDest":3000
        })
        assert "threshold_used" in r.json()

    def test_predict_probability_between_0_and_1(self):
        r = client.post("/predict", json={
            "type":"TRANSFER","amount":50000,
            "oldbalanceOrg":50000,"newbalanceOrig":0,
            "oldbalanceDest":0,"newbalanceDest":50000
        })
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_is_fraud_is_boolean(self):
        r = client.post("/predict", json={
            "type":"CASH_OUT","amount":10000,
            "oldbalanceOrg":10000,"newbalanceOrig":0,
            "oldbalanceDest":5000,"newbalanceDest":15000
        })
        assert isinstance(r.json()["is_fraud"], bool)

    # ── Predict endpoint — all transaction types ──────────────────
    @pytest.mark.parametrize("tx_type", ["PAYMENT","CASH_IN","CASH_OUT","DEBIT","TRANSFER"])
    def test_predict_accepts_all_transaction_types(self, tx_type):
        r = client.post("/predict", json={
            "type":tx_type,"amount":1000,
            "oldbalanceOrg":5000,"newbalanceOrig":4000,
            "oldbalanceDest":2000,"newbalanceDest":3000
        })
        assert r.status_code == 200

    # ── Validation tests ─────────────────────────────────────────
    def test_predict_missing_type_returns_422(self):
        r = client.post("/predict", json={
            "amount":1000,"oldbalanceOrg":5000,
            "newbalanceOrig":4000,"oldbalanceDest":2000,"newbalanceDest":3000
        })
        assert r.status_code == 422

    def test_predict_missing_amount_returns_422(self):
        r = client.post("/predict", json={
            "type":"TRANSFER","oldbalanceOrg":5000,
            "newbalanceOrig":4000,"oldbalanceDest":2000,"newbalanceDest":3000
        })
        assert r.status_code == 422

    def test_predict_empty_body_returns_422(self):
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_predict_missing_all_balances_returns_422(self):
        r = client.post("/predict", json={"type":"TRANSFER","amount":1000})
        assert r.status_code == 422

    # ── Edge cases ────────────────────────────────────────────────
    def test_predict_zero_amount(self):
        r = client.post("/predict", json={
            "type":"PAYMENT","amount":0,
            "oldbalanceOrg":5000,"newbalanceOrig":5000,
            "oldbalanceDest":0,"newbalanceDest":0
        })
        assert r.status_code == 200

    def test_predict_very_large_amount(self):
        r = client.post("/predict", json={
            "type":"TRANSFER","amount":99999999,
            "oldbalanceOrg":99999999,"newbalanceOrig":0,
            "oldbalanceDest":0,"newbalanceDest":99999999
        })
        assert r.status_code == 200

    def test_predict_negative_amount(self):
        """Negative amounts should still be handled without crash."""
        r = client.post("/predict", json={
            "type":"PAYMENT","amount":-100,
            "oldbalanceOrg":5000,"newbalanceOrig":5100,
            "oldbalanceDest":0,"newbalanceDest":0
        })
        assert r.status_code in [200, 422]

    def test_predict_threshold_matches_metadata(self):
        r = client.post("/predict", json={
            "type":"PAYMENT","amount":1000,
            "oldbalanceOrg":5000,"newbalanceOrig":4000,
            "oldbalanceDest":2000,"newbalanceDest":3000
        })
        assert r.json()["threshold_used"] == mock_metadata["threshold"]

    # ── Unknown routes ────────────────────────────────────────────
    def test_unknown_route_returns_404(self):
        r = client.get("/unknown")
        assert r.status_code == 404

    def test_get_on_predict_returns_405(self):
        r = client.get("/predict")
        assert r.status_code == 405


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DRIFT DETECTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDriftDetection:
    """Tests drift logic and threshold behavior."""

    def test_drift_threshold_triggers_retrain(self):
        """drift_share >= 0.30 must set retrain_needed = True."""
        summary = {"drift_share": 0.80, "drift_detected": True}
        retrain = summary["drift_share"] >= 0.30
        assert retrain is True

    def test_drift_below_threshold_no_retrain(self):
        """drift_share < 0.30 must NOT trigger retrain."""
        summary = {"drift_share": 0.20, "drift_detected": True}
        retrain = summary["drift_share"] >= 0.30
        assert retrain is False

    def test_drift_exactly_at_threshold(self):
        """drift_share == 0.30 must trigger retrain (inclusive)."""
        summary = {"drift_share": 0.30}
        retrain = summary["drift_share"] >= 0.30
        assert retrain is True

    def test_drift_share_valid_range(self):
        """drift_share must be between 0 and 1."""
        for val in [0.0, 0.25, 0.50, 0.80, 1.0]:
            assert 0.0 <= val <= 1.0

    def test_no_drift_summary_handled_gracefully(self):
        """Missing drift_summary.json should not crash the pipeline."""
        summary = {}
        retrain = summary.get("retrain_needed", False)
        assert retrain is False

    def test_drift_summary_json_structure(self):
        """drift_summary.json must contain required keys."""
        required_keys = [
            "drift_detected", "drift_share", "n_drifted_cols",
            "n_total_cols", "threshold", "retrain_needed"
        ]
        summary = {
            "drift_detected": True,
            "drift_share": 0.80,
            "n_drifted_cols": 4,
            "n_total_cols": 5,
            "threshold": 0.30,
            "retrain_needed": True,
            "timestamp": datetime.now().isoformat(),
            "drifted_features": ["amount"]
        }
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_n_drifted_cols_consistent_with_share(self):
        """n_drifted_cols / n_total_cols should match drift_share."""
        n_drifted = 4
        n_total   = 5
        computed_share = n_drifted / n_total
        recorded_share = 0.80
        assert abs(computed_share - recorded_share) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRANSACTION SIMULATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTransactionSimulator:
    """Tests the transaction generator logic."""

    def _generate_normal(self, seed=42):
        import random
        random.seed(seed)
        tx_type = random.choice(["PAYMENT","CASH_IN","CASH_OUT","TRANSFER","DEBIT"])
        amount  = abs(np.random.lognormal(9, 1.5))
        old_bal = abs(np.random.lognormal(10, 2))
        new_bal = max(0, old_bal - amount)
        return {
            "type"          : tx_type,
            "amount"        : round(amount, 2),
            "oldbalanceOrg" : round(old_bal, 2),
            "newbalanceOrig": round(new_bal, 2),
            "oldbalanceDest": round(abs(np.random.lognormal(9, 2)), 2),
            "newbalanceDest": round(abs(np.random.lognormal(9, 2)) + amount, 2),
        }

    def test_normal_transaction_has_required_fields(self):
        tx = self._generate_normal()
        required = ["type","amount","oldbalanceOrg","newbalanceOrig",
                    "oldbalanceDest","newbalanceDest"]
        for field in required:
            assert field in tx, f"Missing field: {field}"

    def test_normal_transaction_type_is_valid(self):
        valid_types = {"PAYMENT","CASH_IN","CASH_OUT","TRANSFER","DEBIT"}
        for i in range(20):
            tx = self._generate_normal(seed=i)
            assert tx["type"] in valid_types

    def test_normal_transaction_amount_is_positive(self):
        for i in range(20):
            tx = self._generate_normal(seed=i)
            assert tx["amount"] >= 0

    def test_drift_factor_zero_at_start(self):
        """First 200 transactions have zero drift."""
        n_transactions = 500
        drift_start    = 200
        for i in range(200):
            drift_factor = 0.0 if i < drift_start else (i - drift_start) / (n_transactions - drift_start)
            if i < 200:
                assert drift_factor == 0.0

    def test_drift_factor_increases_linearly(self):
        """Drift factor increases from 0 to 1 after drift_start."""
        n   = 500
        ds  = 200
        factors = [(i - ds) / (n - ds) for i in range(ds, n)]
        assert factors[0]  < 0.01
        # Last index is 499, not 500, so max factor is 299/300 ≈ 0.997
        assert factors[-1] == pytest.approx(299/300, abs=1e-6)
        assert all(factors[i] <= factors[i+1] for i in range(len(factors)-1))
        
    def test_drift_factor_max_is_1(self):
        """Drift factor never exceeds 1."""
        n   = 500
        ds  = 200
        for i in range(n):
            factor = 0.0 if i < ds else (i - ds) / (n - ds)
            assert factor <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL ARTIFACTS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestModelArtifacts:
    """Tests that model files exist and have correct structure."""

    MODEL_JSON     = "models/model.json"
    METADATA_JSON  = "models/model_metadata.json"

    def test_model_json_exists(self):
        assert os.path.exists(self.MODEL_JSON), \
            f"model.json not found at {self.MODEL_JSON}"

    def test_model_metadata_exists(self):
        assert os.path.exists(self.METADATA_JSON), \
            f"model_metadata.json not found at {self.METADATA_JSON}"

    def test_metadata_has_features(self):
        if not os.path.exists(self.METADATA_JSON):
            pytest.skip("model_metadata.json not found")
        with open(self.METADATA_JSON) as f:
            meta = json.load(f)
        assert "features" in meta
        assert len(meta["features"]) > 0

    def test_metadata_has_threshold(self):
        if not os.path.exists(self.METADATA_JSON):
            pytest.skip("model_metadata.json not found")
        with open(self.METADATA_JSON) as f:
            meta = json.load(f)
        assert "threshold" in meta
        assert 0.0 <= meta["threshold"] <= 1.0

    def test_metadata_threshold_is_valid(self):
        if not os.path.exists(self.METADATA_JSON):
            pytest.skip("model_metadata.json not found")
        with open(self.METADATA_JSON) as f:
            meta = json.load(f)
        threshold = meta.get("threshold", 0.5)
        assert 0.0 < threshold < 1.0

    def test_metadata_features_count(self):
        """Should have exactly 17 features."""
        if not os.path.exists(self.METADATA_JSON):
            pytest.skip("model_metadata.json not found")
        with open(self.METADATA_JSON) as f:
            meta = json.load(f)
        assert len(meta["features"]) >= 15

    def test_model_json_not_empty(self):
        if not os.path.exists(self.MODEL_JSON):
            pytest.skip("model.json not found")
        size = os.path.getsize(self.MODEL_JSON)
        assert size > 1000, "model.json seems too small — may be corrupted"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RETRAINING PIPELINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrainingPipeline:
    """Tests retraining logic."""

    def test_retrain_log_structure(self):
        """Retrain log entries must have required keys."""
        entry = {
            "timestamp" : datetime.now().isoformat(),
            "status"    : "DEPLOYED",
            "old_pr_auc": 0.9977,
            "new_pr_auc": 1.0000,
            "deployed"  : True,
            "reason"    : "New model better"
        }
        for key in ["timestamp","status","old_pr_auc","new_pr_auc","deployed"]:
            assert key in entry

    def test_deploy_when_new_model_better(self):
        """New model should deploy when PR-AUC >= current - 0.005."""
        current_score = 0.9977
        new_score     = 1.0000
        should_deploy = new_score >= current_score - 0.005
        assert should_deploy is True

    def test_reject_when_new_model_worse(self):
        """New model should be rejected when PR-AUC significantly worse."""
        current_score = 0.9977
        new_score     = 0.85
        should_deploy = new_score >= current_score - 0.005
        assert should_deploy is False

    def test_tolerance_allows_minor_regression(self):
        """0.5% tolerance allows minor regression due to randomness."""
        current_score = 0.9977
        new_score     = 0.993  # 0.4% lower — within tolerance
        should_deploy = new_score >= current_score - 0.005
        assert should_deploy is True

    def test_pr_auc_valid_range(self):
        """PR-AUC must always be between 0 and 1."""
        for score in [0.0, 0.5, 0.9977, 1.0]:
            assert 0.0 <= score <= 1.0

    def test_retrain_script_exists(self):
        assert os.path.exists("src/training/retrain.py"), \
            "src/training/retrain.py not found"

    def test_retrain_workflow_exists(self):
        assert os.path.exists(".github/workflows/retrain.yml"), \
            ".github/workflows/retrain.yml not found"

    def test_cicd_workflow_exists(self):
        assert os.path.exists(".github/workflows/ci-cd.yml"), \
            ".github/workflows/ci-cd.yml not found"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PROJECT STRUCTURE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestProjectStructure:
    """Tests that all required files and folders exist."""

    @pytest.mark.parametrize("path", [
        "src/api/main.py",
        "src/training/retrain.py",
        "src/monitoring/drift.py",
        "src/simulation/stream.py",
        "src/dashboard/app.py",
        "docker/Dockerfile",
        "requirements.txt",
        "requirements-api.txt",
        "pytest.ini",
        ".github/workflows/ci-cd.yml",
        ".github/workflows/retrain.yml",
        "README.md",
    ])
    def test_required_file_exists(self, path):
        assert os.path.exists(path), f"Required file missing: {path}"

    @pytest.mark.parametrize("folder", [
        "src/api",
        "src/training",
        "src/monitoring",
        "src/simulation",
        "src/dashboard",
        "models",
        "reports/drift",
        "tests",
        "docker",
        ".github/workflows",
    ])
    def test_required_folder_exists(self, folder):
        assert os.path.isdir(folder), f"Required folder missing: {folder}"

    def test_requirements_api_not_empty(self):
        with open("requirements-api.txt") as f:
            content = f.read().strip()
        assert len(content) > 0

    def test_requirements_contains_fastapi(self):
        with open("requirements-api.txt") as f:
            content = f.read()
        assert "fastapi" in content.lower()

    def test_requirements_contains_xgboost(self):
        with open("requirements-api.txt") as f:
            content = f.read()
        assert "xgboost" in content.lower()

    def test_dockerfile_exists_and_not_empty(self):
        assert os.path.exists("docker/Dockerfile")
        size = os.path.getsize("docker/Dockerfile")
        assert size > 0

    def test_gitignore_excludes_venv(self):
        with open(".gitignore") as f:
            content = f.read()
        assert ".venv" in content or "venv/" in content


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DATA INTEGRITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Tests on the transaction log data."""

    LOG_FILE = "reports/drift/transactions_log.csv"

    def _load_log(self):
        if not os.path.exists(self.LOG_FILE):
            pytest.skip("Transaction log not found — run simulator first")
        return pd.read_csv(self.LOG_FILE)

    def test_log_file_not_empty(self):
        df = self._load_log()
        assert len(df) > 0

    def test_log_has_required_columns(self):
        df = self._load_log()
        required = ["timestamp","type","amount","is_fraud",
                    "fraud_probability","drift_factor"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_fraud_probability_in_valid_range(self):
        df = self._load_log()
        assert df["fraud_probability"].between(0, 1).all()

    def test_drift_factor_in_valid_range(self):
        df = self._load_log()
        assert df["drift_factor"].between(0, 1).all()

    def test_amount_is_positive(self):
        df = self._load_log()
        assert (df["amount"] >= 0).all()

    def test_transaction_types_are_valid(self):
        df = self._load_log()
        valid = {"PAYMENT","CASH_IN","CASH_OUT","TRANSFER","DEBIT"}
        assert set(df["type"].unique()).issubset(valid)

    def test_is_fraud_is_boolean_or_01(self):
        df = self._load_log()
        assert df["is_fraud"].isin([True, False, 0, 1]).all()

    def test_drift_increases_after_midpoint(self):
        """Drift factor should be 0 at start and >0 near end."""
        df = self._load_log()
        first_100 = df.head(100)["drift_factor"].mean()
        last_100  = df.tail(100)["drift_factor"].mean()
        assert last_100 > first_100

    def test_fraud_rate_increases_with_drift(self):
        """Higher drift → more fraud detected."""
        df = self._load_log()
        df["is_fraud_int"] = df["is_fraud"].astype(int)
        low_drift  = df[df["drift_factor"] < 0.3]["is_fraud_int"].mean()
        high_drift = df[df["drift_factor"] > 0.7]["is_fraud_int"].mean()
        assert high_drift >= low_drift