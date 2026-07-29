"""
Drift Detection using Evidently AI v0.7+
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

try:
    # Evidently v0.7+
    from evidently import Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    from evidently import Report
    EVIDENTLY_VERSION = "new"
except ImportError:
    try:
        # Evidently v0.4.x
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        EVIDENTLY_VERSION = "old"
    except ImportError:
        EVIDENTLY_VERSION = None


MONITORING_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

LOG_FILE        = "reports/drift/transactions_log.csv"
REPORT_FILE     = "reports/drift/drift_report.html"
SUMMARY_FILE    = "reports/drift/drift_summary.json"
DRIFT_THRESHOLD = 0.30


def generate_reference_data(n=1000, random_state=42):
    np.random.seed(random_state)
    return pd.DataFrame({
        "amount"        : np.random.lognormal(9, 1.5, n),
        "oldbalanceOrg" : np.random.lognormal(10, 2, n),
        "newbalanceOrig": np.random.lognormal(9, 2, n),
        "oldbalanceDest": np.random.lognormal(9, 2, n),
        "newbalanceDest": np.random.lognormal(9, 2, n),
    })


def load_production_data():
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(
            f"Log file not found: {LOG_FILE}\n"
            "Run the simulator first: python src/simulation/stream.py"
        )
    df = pd.read_csv(LOG_FILE)
    print(f"Production data loaded : {len(df)} transactions")
    return df[MONITORING_FEATURES]


def run_drift_detection():
    print("=" * 55)
    print("DRIFT DETECTION — Evidently AI")
    print("=" * 55)

    if EVIDENTLY_VERSION is None:
        print("ERROR: Evidently not installed")
        return None

    print(f"Evidently version mode : {EVIDENTLY_VERSION}")

    reference_df  = generate_reference_data()
    production_df = load_production_data()

    print(f"Reference data  : {len(reference_df)} samples")
    print(f"Production data : {len(production_df)} samples")

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

    if EVIDENTLY_VERSION == "new":
        data_definition    = DataDefinition(numerical_columns=MONITORING_FEATURES)
        reference_dataset  = Dataset.from_pandas(reference_df, data_definition=data_definition)
        production_dataset = Dataset.from_pandas(production_df, data_definition=data_definition)
        report   = Report(metrics=[DataDriftPreset()])
        my_eval  = report.run(reference_data=reference_dataset, current_data=production_dataset)
        my_eval.save_html(REPORT_FILE)
        result_dict = my_eval.dict()
        metrics     = result_dict.get("metrics", [])
        n_drifted   = 0
        drift_share = 0.0
        drifted_features = []
        for metric in metrics:
            metric_name = metric.get("metric_name", "")
            value       = metric.get("value", {})
            if "DriftedColumnsCount" in metric_name and isinstance(value, dict):
                n_drifted   = int(value.get("count", 0))
                drift_share = float(value.get("share", 0.0))
            if "ValueDrift" in metric_name and isinstance(value, dict):
                if value.get("drift_detected", False):
                    col = metric.get("config", {}).get("column", "")
                    if col:
                        drifted_features.append(col)

    else:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=production_df)
        report.save_html(REPORT_FILE)
        result      = report.as_dict()
        dataset_drift = result["metrics"][1]["result"]
        drift_share   = dataset_drift["share_of_drifted_columns"]
        n_drifted     = dataset_drift["number_of_drifted_columns"]
        drifted_features = []

    total_features = len(MONITORING_FEATURES)
    summary = {
        "timestamp"        : datetime.now().isoformat(),
        "drift_detected"   : n_drifted > 0,
        "drift_share"      : round(drift_share, 4),
        "n_drifted_cols"   : n_drifted,
        "n_total_cols"     : total_features,
        "threshold"        : DRIFT_THRESHOLD,
        "retrain_needed"   : drift_share >= DRIFT_THRESHOLD,
        "drifted_features" : drifted_features
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nHTML report saved : {REPORT_FILE}")
    print()
    print("=" * 55)
    print("DRIFT REPORT SUMMARY")
    print("=" * 55)
    print(f"Drift detected    : {summary['drift_detected']}")
    print(f"Drifted features  : {n_drifted} / {total_features} ({drift_share*100:.1f}%)")
    print(f"Threshold         : {DRIFT_THRESHOLD*100:.0f}%")
    print(f"Retraining needed : {summary['retrain_needed']}")
    if drifted_features:
        print(f"Drifted columns   : {drifted_features}")
    print()
    if summary["retrain_needed"]:
        print("DRIFT EXCEEDS THRESHOLD — RETRAINING SHOULD BE TRIGGERED")
    else:
        print("No significant drift detected — model is stable")

    return summary


if __name__ == "__main__":
    summary = run_drift_detection()