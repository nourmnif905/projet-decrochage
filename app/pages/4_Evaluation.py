import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from utils import metrics_at, require

st.title("📈 Module 4 · Évaluation")
require(["fitted"], "Entraîne d'abord des modèles dans la page **3 · Modélisation**.")
s = st.session_state
y_test, probas = s["y_test"], s["probas"]

# ------------------------------------------------------------------ seuil
st.subheader("Seuil de décision")
c1, c2 = st.columns([2, 1])
threshold = c1.slider("Seuil : on signale un étudiant si sa probabilité dépasse ce seuil",
                      0.05, 0.95, 0.50, 0.01)
credit = c2.checkbox("Cas de scoring de crédit (ajouter Gini et KS)", False)
st.caption("Baisser le seuil attrape plus de positifs (rappel ↑) mais crée plus de fausses alertes (précision ↓).")

rows = {n: metrics_at(y_test, p, threshold) for n, p in probas.items()}
tab = pd.DataFrame(rows).T
cols = ["AUC-ROC", "PR-AUC", "Précision", "Rappel", "F1"] + (["Gini", "KS"] if credit else [])
st.dataframe(tab[cols].round(3).sort_values("AUC-ROC", ascending=False))

# ------------------------------------------------------------------ modèle détaillé
model = st.selectbox("Modèle à analyser en détail", list(probas), index=list(probas).index(s["table"].index[0]))
m = rows[model]
a, b, c, d = st.columns(4)
a.metric("Rappel", f"{m['Rappel']:.1%}")
b.metric("Précision", f"{m['Précision']:.1%}")
c.metric("F1", f"{m['F1']:.1%}")
pred = probas[model] >= threshold
d.metric("Étudiants signalés", f"{int(pred.sum())} / {len(pred)}")

# ------------------------------------------------------------------ courbes
left, right = st.columns(2)
with left:
    fig = go.Figure()
    for n, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{n} (AUC={roc_auc_score(y_test, p):.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title="Courbes ROC superposées", xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig)
with right:
    fig = go.Figure()
    for n, p in probas.items():
        pr, rc, _ = precision_recall_curve(y_test, p)
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=n))
    prec_t, rec_t = m["Précision"], m["Rappel"]
    fig.add_trace(go.Scatter(x=[rec_t], y=[prec_t], mode="markers", name=f"Seuil {threshold:.2f}",
                             marker=dict(size=14, color="red", symbol="x")))
    fig.add_hline(y=float(y_test.mean()), line_dash="dash", line_color="gray",
                  annotation_text="Hasard")
    fig.update_layout(title="Courbes précision-rappel", xaxis_title="Rappel",
                      yaxis_title="Précision", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig)

# ------------------------------------------------------------------ confusion + seuils
left, right = st.columns(2)
with left:
    cm = confusion_matrix(y_test, pred.astype(int))
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                              x=["Prédit : négatif", "Prédit : positif"],
                              y=["Réel : négatif", "Réel : positif"],
                              title=f"Matrice de confusion : {model} (seuil {threshold:.2f})"))
    tn, fp, fn, tp = cm.ravel()
    st.caption(f"{tp} positifs détectés · {fn} positifs manqués · {fp} fausses alertes · {tn} vrais négatifs.")
with right:
    grid = np.arange(0.05, 0.96, 0.05)
    curve = pd.DataFrame([{"Seuil": t, **{k: metrics_at(y_test, probas[model], t)[k]
                                          for k in ["Précision", "Rappel", "F1"]}} for t in grid])
    fig = px.line(curve, x="Seuil", y=["Précision", "Rappel", "F1"], markers=True,
                  title=f"Effet du seuil : {model}")
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="seuil choisi")
    st.plotly_chart(fig)

st.info("Le jeu de test ne sert qu'à l'évaluation. Les modules 5 (explicabilité) et 6 (API) viendront ensuite.")
