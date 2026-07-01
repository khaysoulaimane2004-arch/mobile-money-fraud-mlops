"""
Détection de drift avec Evidently AI.

À FAIRE (semaine 3) :
- Comparer un jeu de référence (données d'entraînement) à un jeu de production
  (transactions récentes simulées dans src/simulation/stream.py)
- Générer un rapport Evidently (HTML pour le dashboard, JSON pour le pipeline
  de décision automatique)
- Définir un seuil de drift au-delà duquel on déclenche le ré-entraînement
  (voir .github/workflows/retrain.yml)
"""

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def generate_drift_report(reference_df, current_df, output_path: str) -> dict:
    """Génère un rapport de drift et retourne un résumé exploitable par le pipeline CI."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(output_path)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    return {"drift_detected": drift_detected, "drift_share": drift_share}


def should_trigger_retraining(drift_summary: dict, threshold: float = 0.3) -> bool:
    """Décide si le drift est suffisant pour déclencher un ré-entraînement."""
    return drift_summary["drift_detected"] and drift_summary["drift_share"] >= threshold


if __name__ == "__main__":
    import pandas as pd

    reference = pd.read_csv("data/processed/paysim_features.csv").sample(5000)
    # TODO: remplacer par les vraies données de production loggées par l'API
    current = pd.read_csv("data/processed/paysim_features.csv").sample(5000)

    summary = generate_drift_report(reference, current, "reports/drift/latest_report.html")
    print(summary)
    print("Ré-entraînement nécessaire :", should_trigger_retraining(summary))
