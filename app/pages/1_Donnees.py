import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import DEFAULT_TARGET, ensure_data, load_default_dataset

st.title("📊 Module 1 · Données")

# ------------------------------------------------------------------ source
source = st.radio("Source des données",
                  ["Jeu UCI « Décrochage étudiant » (téléchargement automatique)",
                   "Importer mon propre fichier CSV"], horizontal=True)

if source.startswith("Jeu UCI"):
    df, origine = load_default_dataset()
    st.session_state.update(df=df, target=DEFAULT_TARGET, source_info=origine)
    st.caption(f"Source : {origine}")
else:
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is None:
        ensure_data()
        st.info("En attente d'un fichier. En attendant, le jeu par défaut reste chargé.")
    else:
        raw = pd.read_csv(up)
        col1, col2 = st.columns(2)
        target = col1.selectbox("Variable cible", raw.columns, index=len(raw.columns) - 1)
        values = raw[target].dropna().unique().tolist()
        pos = col2.selectbox("Valeur de la classe d'intérêt (positive)", values)
        df = raw.dropna(subset=[target]).copy()
        df[target] = (df[target] == pos).astype(int)
        st.session_state.update(df=df, target=target, source_info=f"fichier importé ({up.name})")
        for k in ["X_train", "fitted"]:       # une nouvelle donnée invalide la suite
            st.session_state.pop(k, None)

ensure_data()
df, target = st.session_state["df"], st.session_state["target"]
y = df[target]
X = df.drop(columns=target)

# ------------------------------------------------------------------ aperçu
st.subheader("Aperçu")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(df):,}".replace(",", " "))
c2.metric("Variables explicatives", X.shape[1])
c3.metric("Valeurs manquantes", int(df.isna().sum().sum()))
c4.metric("Classe positive", f"{y.mean():.1%}")

n = st.slider("Nombre de lignes à afficher", 5, 50, 10)
st.dataframe(df.head(n))

tab_stats, tab_na, tab_cls = st.tabs(["Statistiques descriptives", "Valeurs manquantes", "Classes"])
with tab_stats:
    st.dataframe(df.describe().T)
with tab_na:
    na = df.isna().sum().rename("Manquantes").to_frame()
    na["Part (%)"] = (na["Manquantes"] / len(df) * 100).round(2)
    st.dataframe(na[na["Manquantes"] > 0] if na["Manquantes"].sum() else na.head(0))
    if na["Manquantes"].sum() == 0:
        st.success("Aucune valeur manquante.")
with tab_cls:
    counts = y.map({0: "Classe 0 (négative)", 1: "Classe 1 (positive)"}).value_counts()
    st.plotly_chart(px.pie(names=counts.index, values=counts.values,
                           title="Répartition de la variable cible"))
    if y.mean() < 0.4 or y.mean() > 0.6:
        st.info(f"Classes déséquilibrées ({y.mean():.1%} de positifs) : privilégier "
                "AUC-ROC, PR-AUC, rappel et F1 plutôt que l'accuracy.")

# ------------------------------------------------------------------ exploration
st.subheader("Visualisations exploratoires")
num_cols = X.select_dtypes("number").columns.tolist()
tab_dist, tab_corr, tab_rate = st.tabs(["Distributions", "Corrélations", "Taux de la classe positive"])

with tab_dist:
    if num_cols:
        col = st.selectbox("Variable", num_cols, key="dist")
        d = df.assign(Classe=y.map({0: "Négative", 1: "Positive"}))
        st.plotly_chart(px.histogram(d, x=col, color="Classe", barmode="overlay",
                                     opacity=0.6, title=f"Distribution de {col} selon la classe"))
    else:
        st.info("Pas de variable numérique.")

with tab_corr:
    if len(num_cols) > 1:
        corr_t = df[num_cols + [target]].corr()[target].drop(target).sort_values()
        st.plotly_chart(px.bar(corr_t, orientation="h", height=max(400, 18 * len(corr_t)),
                               title="Corrélation de chaque variable avec la cible",
                               labels={"value": "Corrélation", "index": "Variable"}))
        if st.checkbox("Afficher la matrice de corrélation complète"):
            st.plotly_chart(px.imshow(X[num_cols].corr(), zmin=-1, zmax=1, height=800,
                                      color_continuous_scale="RdBu_r"))
    else:
        st.info("Pas assez de variables numériques.")

with tab_rate:
    col = st.selectbox("Variable", X.columns.tolist(), key="rate")
    s = X[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:
        s = pd.qcut(s, q=10, duplicates="drop").astype(str)
    g = pd.DataFrame({"mod": s.astype(str), "y": y}).groupby("mod")["y"].agg(["mean", "count"]).reset_index()
    fig = px.bar(g, x="mod", y="mean", hover_data=["count"],
                 labels={"mod": col, "mean": "Taux de classe positive"},
                 title=f"Taux de classe positive selon : {col}")
    fig.add_hline(y=y.mean(), line_dash="dash", annotation_text="Taux global")
    fig.update_xaxes(type="category")
    st.plotly_chart(fig)
