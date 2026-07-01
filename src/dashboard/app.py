"""
Dashboard Streamlit — vue d'ensemble du système de détection de fraude.

À FAIRE (semaine 4) :
- Afficher le volume de transactions traitées et le taux de fraude détecté
- Afficher l'historique du drift (graphique dans le temps)
- Afficher l'historique des ré-entraînements déclenchés (date, raison, métriques
  avant/après)
- Lien vers le dernier rapport Evidently complet (reports/drift/latest_report.html)
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraude mobile money — monitoring", layout="wide")

st.title("Monitoring — détection de fraude mobile money")

col1, col2, col3 = st.columns(3)
# TODO: remplacer par les vraies métriques issues des logs de l'API
col1.metric("Transactions scorées (24h)", "—")
col2.metric("Taux de fraude détecté", "—")
col3.metric("Dernier drift mesuré", "—")

st.subheader("Évolution du drift dans le temps")
st.info("Graphique à connecter à reports/drift/ une fois le monitoring en place.")

st.subheader("Historique des ré-entraînements")
st.info("Table à connecter aux runs MLflow une fois la boucle automatique en place.")
