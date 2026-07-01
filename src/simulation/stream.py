"""
Simulateur de flux de transactions de mobile money avec drift progressif.

À FAIRE (semaine 3) :
- Rejouer un échantillon de PaySim comme un flux "temps réel" (transaction par
  transaction ou par lot)
- Injecter un drift progressif et contrôlé, par exemple :
    - augmenter progressivement le montant moyen des transactions
    - changer la distribution des types de transaction (plus de TRANSFER par ex.)
    - faire évoluer le pattern des comptes fraudeurs
- Envoyer chaque transaction à l'API de scoring déployée et logger la réponse
- Sauvegarder un échantillon glissant (référence vs production) pour
  src/monitoring/drift.py
"""

import time

import numpy as np
import pandas as pd
import requests


def inject_drift(df: pd.DataFrame, drift_factor: float) -> pd.DataFrame:
    """
    Modifie artificiellement la distribution des montants pour simuler un
    changement de comportement des utilisateurs/fraudeurs dans le temps.
    drift_factor: 0 = pas de drift, augmente progressivement au fil du temps.
    """
    df = df.copy()
    df["amount"] = df["amount"] * (1 + drift_factor)
    return df


def stream_transactions(
    df: pd.DataFrame,
    api_url: str,
    batch_size: int = 50,
    delay_seconds: float = 1.0,
    max_drift: float = 0.5,
):
    """Rejoue le dataset en simulant un drift croissant au fil du flux."""
    n_batches = len(df) // batch_size
    for i in range(n_batches):
        drift_factor = max_drift * (i / n_batches)  # drift linéaire croissant
        batch = df.iloc[i * batch_size : (i + 1) * batch_size]
        batch = inject_drift(batch, drift_factor)

        for _, row in batch.iterrows():
            payload = row.to_dict()
            # TODO: gérer les erreurs réseau / retries
            requests.post(f"{api_url}/predict", json=payload, timeout=5)

        time.sleep(delay_seconds)


if __name__ == "__main__":
    data = pd.read_csv("data/processed/paysim_features.csv")
    stream_transactions(data, api_url="http://localhost:8080")
