"""
Automatic Retraining Script
Triggered by GitHub Actions when drift is detected.
Retrains XGBoost, compares with current model,
and saves the new model only if it is better.
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE

# ─── Paths ────────────────────────────────────────────────────────
DRIFT_SUMMARY  = "reports/drift/drift_summary.json"
MODEL_JSON     = "models/model.json"
METADATA_JSON  = "models/model_metadata.json"
RETRAIN_LOG    = "reports/drift/retrain_log.json"

# ─── Features ─────────────────────────────────────────────────────
FEATURES = [
    'amount', 'log_amount',
    'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balance_error_orig', 'balance_error_dest',
    'orig_balance_emptied', 'dest_balance_was_zero',
    'amount_to_balance_ratio', 'is_risky_type',
    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
    'type_PAYMENT', 'type_TRANSFER'
]


def generate_training_data(n=150000):
    """
    Generate fresh training data.
    In production: replace this with a real data pull
    from your data warehouse or feature store.
    """
    print("Generating training data...")
    np.random.seed(int(datetime.now().timestamp()) % 10000)

    types = np.random.choice(
        ['PAYMENT', 'CASH_OUT', 'CASH_IN', 'TRANSFER', 'DEBIT'],
        n, p=[0.339, 0.352, 0.220, 0.083, 0.006]
    )
    amount       = np.random.lognormal(11, 1.5, n)
    oldbalOrg    = np.random.lognormal(12, 2, n)
    newbalOrg    = np.maximum(oldbalOrg - amount, 0)
    oldbalDest   = np.random.lognormal(11, 2, n)
    newbalDest   = oldbalDest + amount
    is_fraud     = np.zeros(n, dtype=int)

    # Inject fraud only in CASH_OUT and TRANSFER
    risky_idx  = np.where(np.isin(types, ['CASH_OUT', 'TRANSFER']))[0]
    fraud_idx  = np.random.choice(risky_idx, size=int(n * 0.0013), replace=False)
    is_fraud[fraud_idx] = 1
    newbalOrg[fraud_idx] = 0
    amount[fraud_idx]    = oldbalOrg[fraud_idx]

    df = pd.DataFrame({
        'type'          : types,
        'amount'        : amount,
        'oldbalanceOrg' : oldbalOrg,
        'newbalanceOrig': newbalOrg,
        'oldbalanceDest': oldbalDest,
        'newbalanceDest': newbalDest,
        'isFraud'       : is_fraud,
    })

    print(f"Data generated : {n:,} rows | {is_fraud.sum():,} frauds")
    return df


def engineer_features(df):
    """Apply same feature engineering as original training."""
    df = df.copy()
    df['balance_error_orig']      = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['balance_error_dest']      = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['orig_balance_emptied']    = (df['newbalanceOrig'] == 0).astype(int)
    df['dest_balance_was_zero']   = (df['oldbalanceDest'] == 0).astype(int)
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['log_amount']              = np.log1p(df['amount'])
    df['is_risky_type']           = df['type'].isin(['CASH_OUT', 'TRANSFER']).astype(int)
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)

    # Ensure all type columns exist
    for t in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
        col = f'type_{t}'
        if col not in df.columns:
            df[col] = 0

    return df


def get_current_model_score():
    """Load current model metadata to get its PR-AUC."""
    if not os.path.exists(METADATA_JSON):
        print("No current model metadata found — baseline is 0")
        return 0.0
    with open(METADATA_JSON) as f:
        meta = json.load(f)
    score = meta.get("metrics", {}).get("pr_auc", 0.0)
    print(f"Current model PR-AUC : {score:.4f}")
    return score


def train_new_model(X_train, y_train):
    """Train a new XGBoost model."""
    print("Training new XGBoost model...")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = 'aucpr',
        random_state      = 42,
        n_jobs            = -1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return PR-AUC."""
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc  = average_precision_score(y_test, y_proba)
    return pr_auc


def save_model(model, pr_auc, features):
    """Save model in JSON format + metadata."""
    model.get_booster().save_model(MODEL_JSON)

    metadata = {
        "features"   : features,
        "threshold"  : 0.5,
        "metrics"    : {"pr_auc": pr_auc},
        "xgb_version": xgb.__version__,
        "retrained_at": datetime.now().isoformat()
    }
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"New model saved : PR-AUC {pr_auc:.4f}")


def log_retrain_result(status, old_score, new_score, reason=""):
    """Append retraining result to log file."""
    log = []
    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG) as f:
            try:
                log = json.load(f)
            except Exception:
                log = []

    log.append({
        "timestamp" : datetime.now().isoformat(),
        "status"    : status,
        "old_pr_auc": round(old_score, 4),
        "new_pr_auc": round(new_score, 4),
        "deployed"  : status == "DEPLOYED",
        "reason"    : reason
    })

    with open(RETRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2)


def check_drift_requires_retraining():
    """Check if drift summary says retraining is needed."""
    if not os.path.exists(DRIFT_SUMMARY):
        print("No drift summary found — running retraining anyway")
        return True
    with open(DRIFT_SUMMARY) as f:
        summary = json.load(f)
    needed = summary.get("retrain_needed", False)
    drift  = summary.get("drift_share", 0) * 100
    print(f"Drift level : {drift:.0f}% | Retraining needed : {needed}")
    return needed


def main():
    print("=" * 60)
    print("AUTO-RETRAINING PIPELINE")
    print(f"Started at : {datetime.now().isoformat()}")
    print("=" * 60)

    # Step 1: Check if retraining is needed
    if not check_drift_requires_retraining():
        print("Drift below threshold — retraining not needed")
        sys.exit(0)

    # Step 2: Get current model score
    current_score = get_current_model_score()

    # Step 3: Generate / load fresh data
    df = generate_training_data(n=150000)
    df = engineer_features(df)

    X = df[FEATURES].astype(float)
    y = df['isFraud'].astype(int)

    # Step 4: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 5: Handle imbalance with SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE : {len(X_train_res):,} samples")

    # Step 6: Train new model
    new_model = train_new_model(X_train_res, y_train_res)

    # Step 7: Evaluate new model
    new_score = evaluate_model(new_model, X_test, y_test)
    print(f"New model PR-AUC : {new_score:.4f}")

    # Step 8: Compare and deploy if better
    print("-" * 60)
    if new_score >= current_score - 0.005:
        # Accept new model if within 0.5% of current
        # (small tolerance for randomness in data generation)
        save_model(new_model, new_score, FEATURES)
        log_retrain_result("DEPLOYED", current_score, new_score,
                           f"New model ({new_score:.4f}) >= current ({current_score:.4f})")
        print(f"NEW MODEL DEPLOYED ✓ (PR-AUC: {new_score:.4f})")

        # Write flag for GitHub Actions to detect
        with open("reports/drift/model_updated.flag", "w") as f:
            f.write(f"updated={datetime.now().isoformat()}")

    else:
        log_retrain_result("REJECTED", current_score, new_score,
                           f"New model ({new_score:.4f}) < current ({current_score:.4f})")
        print(f"NEW MODEL REJECTED — keeping current model")
        print(f"Current: {current_score:.4f} | New: {new_score:.4f}")
        sys.exit(1)

    print("=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()