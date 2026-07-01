"""
Entraînement du modèle de détection de fraude avec tracking MLflow.

À FAIRE (semaine 1) :
- Charger data/processed/paysim_features.csv
- Split train/test stratifié (important vu le déséquilibre)
- Entraîner plusieurs modèles candidats (XGBoost, RandomForest, LogisticRegression
  en baseline)
- Logger chaque run dans MLflow : hyperparamètres, métriques (PR-AUC en priorité,
  pas l'accuracy qui ne veut rien dire sur ces données), matrice de confusion
- Sélectionner le meilleur modèle et le sauvegarder dans models/
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_error_orig",
    "balance_error_dest",
    "orig_balance_emptied",
    "amount_to_balance_ratio",
    # TODO: ajouter les colonnes encodées de 'type'
]
TARGET_COLUMN = "isFraud"


def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_xgboost(X_train, y_train, **params) -> XGBClassifier:
    """Entraîne un XGBoost avec class_weight adapté au déséquilibre."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        **params,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    pr_auc = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {"pr_auc": pr_auc, "report": report}


def run_experiment(data_path: str, experiment_name: str = "fraud-mobile-money"):
    mlflow.set_experiment(experiment_name)
    df = load_features(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    with mlflow.start_run():
        params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
        mlflow.log_params(params)

        model = train_xgboost(X_train, y_train, **params)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_metric("pr_auc", metrics["pr_auc"])
        mlflow.sklearn.log_model(model, "model")

        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        return model, metrics


if __name__ == "__main__":
    run_experiment("data/processed/paysim_features.csv")
