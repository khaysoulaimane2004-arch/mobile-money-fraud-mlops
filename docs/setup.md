# Mise en route

## 1. Environnement local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Dataset

Télécharger PaySim depuis Kaggle (`ealaxi/paysim1`) et placer le CSV dans `data/raw/`.

Sur Kaggle Notebooks, le dataset peut être ajouté directement comme source de données — pas besoin de le télécharger en local.

## 3. Entraînement (semaine 1)

```bash
python src/training/features.py
python src/training/train.py
```

Pour visualiser les runs MLflow en local :
```bash
mlflow ui
```

## 4. API en local (semaine 2)

```bash
uvicorn src.api.main:app --reload --port 8080
```

Build et test de l'image Docker :
```bash
docker build -f docker/Dockerfile -t fraud-api .
docker run -p 8080:8080 fraud-api
```

## 5. Déploiement manuel sur Cloud Run (semaine 2, avant le CI/CD)

Prérequis : `gcloud` CLI installé et authentifié (`gcloud auth login`), projet GCP configuré.

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/fraud-api -f docker/Dockerfile .
gcloud run deploy fraud-mobile-money-api \
  --image gcr.io/PROJECT_ID/fraud-api \
  --region europe-west1 \
  --allow-unauthenticated
```

## 6. Secrets GitHub Actions à configurer (semaine 3)

Dans les paramètres du repo GitHub (`Settings > Secrets and variables > Actions`) :
- `GCP_PROJECT_ID` : l'identifiant du projet GCP
- `GCP_SA_KEY` : clé JSON d'un compte de service avec les droits Cloud Run + Artifact Registry

## 7. Dashboard (semaine 4)

```bash
streamlit run src/dashboard/app.py
```
