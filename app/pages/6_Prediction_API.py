import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import (api_call, api_url, load_default_dataset, model_selector, require_api,
                   student_form)

st.title("🔌 Module 6 · Prédiction via l'API FastAPI")
st.caption("Cette page n'effectue aucun calcul de modèle : elle envoie l'étudiant à l'API, qui renvoie la probabilité.")

models = require_api()
c1, c2, c3 = st.columns(3)
c1.metric("État de l'API", "En ligne ✅")
c2.metric("Modèles disponibles", len(models))
c3.markdown(f"[Documentation Swagger]({api_url()}/docs)")

tab_pred, tab_models = st.tabs(["Prédire pour un étudiant", "Modèles de l'API"])

with tab_models:
    tbl = pd.DataFrame([{"Identifiant": m["id"], "Modèle": m["name"], "Scénario": m["scenario"],
                         "Seuil": m["threshold"], "Variables": m["n_features"], **m["metrics"]} for m in models])
    st.dataframe(tbl)
    st.caption("Les métriques proviennent des notebooks (jeu de test). "
               "Les modèles « Toutes les variables » de modelisation.ipynb utilisent le seuil 0,5.")

with tab_pred:
    model = model_selector(models, key="model6")
    model_id = model["id"]
    specs, err = api_call("GET", "/features", params={"model_id": model_id})
    if err:
        st.error(err)
        st.stop()

    df, _ = load_default_dataset()
    pool = st.session_state["X_test"] if "X_test" in st.session_state else df.drop(columns="Dropout")
    values = student_form(specs, df, pool, f"p6_{model_id}")

    threshold = st.slider("Seuil de décision", 0.05, 0.95, float(model["threshold"]), 0.01,
                          help="Au-dessus de ce seuil, l'étudiant est signalé comme à risque.")

    if st.button("🔮 Prédire via l'API", type="primary"):
        res, err = api_call("POST", "/predict",
                            json={"model_id": model_id, "features": values, "threshold": threshold})
        if err:
            st.error(err)
        else:
            st.session_state["last_pred"] = res

    res = st.session_state.get("last_pred")
    if res and res["model_id"] == model_id:
        a, b = st.columns([1, 1])
        with a:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=res["probability"] * 100,
                number={"suffix": " %"}, title={"text": "Probabilité de décrochage"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#c0392b" if res["prediction"] else "#27ae60"},
                       "threshold": {"line": {"color": "black", "width": 4}, "value": res["threshold"] * 100}}))
            fig.update_layout(height=300, margin=dict(t=60, b=10))
            st.plotly_chart(fig)
        with b:
            (st.error if res["prediction"] else st.success)(f"**{res['decision']}**")
            st.write(f"Seuil utilisé : **{res['threshold']:.2f}** · Modèle : **{res['model_name']}** "
                     f"({res['scenario']})")
            if res["ignored_features"]:
                st.caption("Variables ignorées par ce modèle : " + ", ".join(res["ignored_features"]))
            with st.expander("Réponse JSON de l'API"):
                st.json(res)
