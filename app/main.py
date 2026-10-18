"""Point d'entrée : routeur des pages (pas de page d'accueil).  streamlit run app/main.py"""
import streamlit as st
from utils import api_call, api_url, ensure_data

st.set_page_config(page_title="Classification explicable", page_icon="🎓", layout="wide")
ensure_data()

pages = [
    st.Page("pages/1_Donnees.py", title="1 · Données", icon="📊", url_path="donnees", default=True),
    st.Page("pages/2_Preparation.py", title="2 · Préparation", icon="🧹", url_path="preparation"),
    st.Page("pages/3_Modelisation.py", title="3 · Modélisation", icon="🤖", url_path="modelisation"),
    st.Page("pages/4_Evaluation.py", title="4 · Évaluation", icon="📈", url_path="evaluation"),
    st.Page("pages/5_Explicabilite.py", title="5 · Explicabilité", icon="🔍", url_path="explicabilite"),
    st.Page("pages/6_Prediction_API.py", title="6 · Prédiction (API)", icon="🔌", url_path="prediction"),
]
page = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption("État de la session")
    health, _ = api_call("GET", "/health", timeout=2)
    st.write("Préparation :", "✅" if "X_train" in st.session_state else "❌")
    st.write("Modèles :", "✅" if "fitted" in st.session_state else "❌")
    st.write("API FastAPI :", "✅" if health else "❌")
    st.caption(api_url())

page.run()
