"""Fonctions communes à toutes les pages de l'application."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:          # xgboost non installé
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent.parent
CSV_LOCAL = ROOT / "data" / "students_dropout.csv"
DEFAULT_TARGET = "Dropout"

# Variables « codes » du jeu décrochage étudiant (entiers qui sont en fait des catégories)
CAT_DEFAULT = ["Marital Status", "Application mode", "Course", "Previous qualification",
               "Nationality", "Mother's qualification", "Father's qualification",
               "Mother's occupation", "Father's occupation"]


# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Téléchargement du jeu de données...")
def load_default_dataset():
    """Jeu UCI « Décrochage étudiant » : copie locale si elle existe, sinon téléchargement."""
    if CSV_LOCAL.exists():
        return pd.read_csv(CSV_LOCAL), "copie locale (data/students_dropout.csv)"
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=697)
    df = data.data.features.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Nacionality": "Nationality"})
    df = df.drop(columns=["Educational special needs"], errors="ignore")
    df[DEFAULT_TARGET] = (data.data.targets.iloc[:, 0] == "Dropout").astype(int)
    return df, "téléchargement automatique UCI (id=697)"


def ensure_data():
    """Charge le jeu par défaut si aucune donnée n'est encore en mémoire."""
    if "df" not in st.session_state:
        df, origine = load_default_dataset()
        st.session_state["df"] = df
        st.session_state["target"] = DEFAULT_TARGET
        st.session_state["source_info"] = origine


def require(keys, message):
    """Arrête la page si une étape précédente n'a pas été faite."""
    if any(k not in st.session_state for k in keys):
        st.warning(message)
        st.stop()


# ----------------------------------------------------------------------------
# Préparation
# ----------------------------------------------------------------------------
def detect_types(df, features):
    """Classe les variables en catégorielles / binaires / numériques (détection automatique)."""
    cat = [c for c in features
           if c in CAT_DEFAULT or df[c].dtype == "object" or str(df[c].dtype) in ("category", "bool")]
    rest = [c for c in features if c not in cat]
    binary = [c for c in rest if df[c].nunique() == 2]
    num = [c for c in rest if c not in binary]
    return cat, binary, num


def build_prep(cfg, columns):
    """Préprocesseur (imputation, encodage, normalisation) limité aux colonnes données."""
    cat = [c for c in cfg["cat"] if c in columns]
    num = [c for c in cfg["num"] if c in columns]
    binary = [c for c in cfg["bin"] if c in columns]
    scalers = {"Standardisation (StandardScaler)": StandardScaler(),
               "Min-Max (0 à 1)": MinMaxScaler(), "Aucune": "passthrough"}
    return ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                min_frequency=cfg["min_freq"],
                                                sparse_output=False))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy=cfg["impute"])),
                          ("sc", scalers[cfg["scaling"]])]), num),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ])


# ----------------------------------------------------------------------------
# Modèles
# ----------------------------------------------------------------------------
def model_names():
    names = ["Régression logistique", "Arbre de décision", "KNN", "Random Forest"]
    return names + (["XGBoost"] if HAS_XGB else [])


def build_estimator(name, p, balance=True, spw=1.0, seed=42):
    cw = "balanced" if balance else None
    if name == "Régression logistique":
        return LogisticRegression(C=p.get("C", 0.25), max_iter=3000, class_weight=cw)
    if name == "Arbre de décision":
        return DecisionTreeClassifier(max_depth=p.get("max_depth", 8),
                                      min_samples_leaf=p.get("min_samples_leaf", 20),
                                      class_weight=cw, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 15),
                                    weights=p.get("weights", "distance"), p=p.get("p", 1))
    if name == "Random Forest":
        depth = p.get("max_depth", 0)
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 200),
                                      max_depth=None if depth == 0 else depth,
                                      min_samples_leaf=p.get("min_samples_leaf", 1),
                                      class_weight=cw, random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=p.get("n_estimators", 300),
                             max_depth=p.get("max_depth", 3),
                             learning_rate=p.get("learning_rate", 0.05),
                             subsample=p.get("subsample", 0.8),
                             colsample_bytree=p.get("colsample_bytree", 0.8),
                             scale_pos_weight=spw if balance else 1.0,
                             eval_metric="logloss", random_state=seed, n_jobs=-1)
    raise ValueError(name)


# Grilles pour l'optimisation automatique (RandomizedSearchCV)
SEARCH_GRIDS = {
    "Régression logistique": {"clf__C": loguniform(1e-3, 1e1)},
    "Arbre de décision": {"clf__max_depth": [3, 4, 5, 6, 8, 10],
                          "clf__min_samples_leaf": [1, 5, 10, 20, 40]},
    "KNN": {"clf__n_neighbors": list(range(3, 40, 2)),
            "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
    "Random Forest": {"clf__n_estimators": [100, 200, 300],
                      "clf__max_depth": [5, 8, 12, None],
                      "clf__min_samples_leaf": [1, 3, 5, 10]},
    "XGBoost": {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [2, 3, 4, 5],
                "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0]},
}


# ----------------------------------------------------------------------------
# Métriques
# ----------------------------------------------------------------------------
def metrics_at(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "AUC-ROC": auc,
        "PR-AUC": average_precision_score(y_true, proba),
        "Précision": precision_score(y_true, pred, zero_division=0),
        "Rappel": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Gini": 2 * auc - 1,
        "KS": float(np.max(tpr - fpr)),
    }


# ----------------------------------------------------------------------------
# Appels à l'API FastAPI
# ----------------------------------------------------------------------------
import os
import requests


def api_url():
    """URL de l'API : variable d'environnement API_URL, sinon secrets Streamlit, sinon local."""
    url = os.getenv("API_URL")
    if not url:
        try:
            url = st.secrets.get("API_URL")
        except Exception:
            url = None
    return (url or "http://127.0.0.1:8000").rstrip("/")


def api_call(method, path, timeout=30, **kwargs):
    """Appelle l'API. Renvoie (données, message_erreur) : l'un des deux est None."""
    try:
        r = requests.request(method, f"{api_url()}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.RequestException:
        return None, f"API injoignable à l'adresse {api_url()}"
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return None, f"Erreur {r.status_code} : {detail}"
    return r.json(), None
