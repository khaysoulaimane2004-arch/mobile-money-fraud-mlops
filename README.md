# Détection de fraude sur transactions de mobile money — pipeline MLOps

Pipeline MLOps de bout en bout pour la détection de fraude sur des transactions de mobile money (dépôt, retrait, transfert), avec entraînement traçable, déploiement conteneurisé, CI/CD, monitoring de drift et ré-entraînement automatique.

## Contexte métier

Le mobile money (Orange Money, MTN MoMo, M-Pesa, etc.) est un mode de paiement majeur dans plusieurs marchés, notamment en Afrique. La fraude y représente un risque financier direct pour les opérateurs et leurs utilisateurs. Ce projet simule un système de scoring de fraude en temps quasi-réel capable de détecter une dérive du comportement des fraudeurs dans le temps et de déclencher automatiquement un ré-entraînement du modèle.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — simulateur de transactions de mobile money basé sur un échantillon réel d'un mois de logs financiers d'un opérateur africain. 6,3 millions de transactions, 5 types (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER), taux de fraude extrêmement déséquilibré (~0,13%).

## Architecture

```
Kaggle Notebook (entraînement + MLflow)
        |
        v
Image Docker (modèle + API FastAPI)
        |
        v
GitHub Actions (build, test, déploiement)
        |
        v
Cloud Run (API de scoring en production)
        |
        v
Flux de transactions simulé --> Evidently AI (détection drift)
        |
        v
Dashboard Streamlit <-- déclenchement ré-entraînement si drift
        |
        v
(retour vers l'entraînement)
```

## Structure du repo

```
.
├── notebooks/              # exploration et entraînement (Kaggle)
├── src/
│   ├── training/            # scripts d'entraînement, feature engineering
│   ├── api/                 # API FastAPI de scoring
│   ├── monitoring/           # détection de drift (Evidently AI)
│   ├── simulation/           # générateur de flux de transactions avec drift
│   └── dashboard/             # dashboard Streamlit
├── tests/                  # tests unitaires et d'intégration
├── docker/                 # Dockerfile(s)
├── .github/workflows/        # pipelines CI/CD
├── data/
│   ├── raw/                  # données brutes (non versionnées, voir .gitignore)
│   └── processed/             # données nettoyées/échantillonnées
├── models/                 # artefacts de modèles entraînés (non versionnés)
├── reports/drift/             # rapports de drift générés
└── docs/                    # documentation complémentaire
```

## Stack technique

- **Entraînement** : Python, scikit-learn, XGBoost, MLflow
- **Service** : FastAPI, Docker
- **Cloud** : Google Cloud Run (free tier)
- **CI/CD** : GitHub Actions
- **Monitoring** : Evidently AI
- **Dashboard** : Streamlit

## Avancement

- [ ] Semaine 1 — Exploration des données, feature engineering, entraînement, tracking MLflow
- [ ] Semaine 2 — API FastAPI, conteneurisation Docker, déploiement manuel sur Cloud Run
- [ ] Semaine 3 — CI/CD GitHub Actions, intégration Evidently AI
- [ ] Semaine 4 — Boucle de ré-entraînement automatique, dashboard Streamlit, documentation finale

## Lancer le projet en local

Instructions à compléter au fur et à mesure de l'avancement (voir `docs/setup.md`).
