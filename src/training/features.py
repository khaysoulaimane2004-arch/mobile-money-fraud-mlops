"""
Feature engineering pour le dataset PaySim (transactions de mobile money).

Colonnes brutes attendues :
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

À FAIRE (semaine 1) :
- Échantillonner le dataset (6.3M lignes -> taille gérable sur Kaggle, ex: 1-2M
  en gardant TOUTES les fraudes + un sous-échantillon des transactions légitimes)
- Créer des features dérivées pertinentes pour la fraude mobile money, ex :
    - erreur_solde_orig = oldbalanceOrg - amount - newbalanceOrig (devrait être ~0)
    - erreur_solde_dest = oldbalanceDest + amount - newbalanceDest
    - solde_orig_vide_apres = (newbalanceOrig == 0)
    - ratio_montant_solde = amount / (oldbalanceOrg + 1)
- Encoder la variable 'type' (one-hot ou target encoding)
- Gérer le déséquilibre de classes (SMOTE ou class_weight, à comparer)
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Charge le CSV brut PaySim."""
    return pd.read_csv(path)


def sample_dataset(
    df: pd.DataFrame,
    fraud_col: str = "isFraud",
    legit_sample_frac: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Garde toutes les fraudes (minoritaires) et échantillonne les transactions
    légitimes pour réduire la taille du dataset à un niveau gérable sur Kaggle.
    """
    fraud = df[df[fraud_col] == 1]
    legit = df[df[fraud_col] == 0].sample(frac=legit_sample_frac, random_state=random_state)
    return pd.concat([fraud, legit]).sample(frac=1, random_state=random_state).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features dérivées. À compléter."""
    df = df.copy()
    df["balance_error_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["balance_error_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["orig_balance_emptied"] = (df["newbalanceOrig"] == 0).astype(int)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    # TODO: encodage de 'type', features temporelles à partir de 'step', etc.
    return df


if __name__ == "__main__":
    # Exemple d'utilisation, à adapter au chemin réel sur Kaggle
    raw = load_raw_data("data/raw/paysim.csv")
    sampled = sample_dataset(raw)
    features = engineer_features(sampled)
    features.to_csv("data/processed/paysim_features.csv", index=False)
