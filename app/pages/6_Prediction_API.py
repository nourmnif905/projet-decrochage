import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import api_call, api_url, ensure_data, load_default_dataset

st.title("🔌 Module 6 · Prédiction via l'API FastAPI")
st.caption("Cette page n'effectue aucun calcul de modèle : elle envoie l'étudiant à l'API, qui renvoie la probabilité.")

# ------------------------------------------------------------------ état de l'API
health, err = api_call("GET", "/health", timeout=3)
if err:
    st.error(err)
    st.code("uvicorn api.main:app --port 8000", language="bash")
    st.info("Lance cette commande dans un second terminal (venv activé, depuis la racine du projet), puis recharge la page.")
    st.stop()
if health["n_models"] == 0:
    st.warning("L'API ne trouve aucun modèle dans `models/`. Exécute `modelisation.ipynb` puis `evaluation_finale.ipynb`.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("État de l'API", "En ligne ✅")
c2.metric("Modèles disponibles", health["n_models"])
c3.markdown(f"[Documentation Swagger]({api_url()}/docs)")

tab_pred, tab_models = st.tabs(["Prédire pour un étudiant", "Modèles de l'API"])

models, err = api_call("GET", "/models")
if err:
    st.error(err)
    st.stop()

# ------------------------------------------------------------------ liste des modèles
with tab_models:
    tbl = pd.DataFrame([{"Identifiant": m["id"], "Modèle": m["name"], "Scénario": m["scenario"],
                         "Seuil": m["threshold"], "Variables": m["n_features"], **m["metrics"]} for m in models])
    st.dataframe(tbl)
    st.caption("Les métriques proviennent des notebooks (jeu de test). "
               "Les modèles « Toutes les variables » de modelisation.ipynb utilisent le seuil 0,5.")

# ------------------------------------------------------------------ prédiction
with tab_pred:
    labels = {m["id"]: f"{m['name']} · {m['scenario']} (seuil {m['threshold']})" for m in models}
    ids = list(labels)
    default = next((i for i in ids if "__" in i and i.startswith("xgboost") and "toutes" in i), ids[0])
    model_id = st.selectbox("Modèle", ids, index=ids.index(default), format_func=labels.get)
    model = next(m for m in models if m["id"] == model_id)

    specs, err = api_call("GET", "/features", params={"model_id": model_id})
    if err:
        st.error(err)
        st.stop()

    # étudiant de départ
    df, _ = load_default_dataset()
    pool = st.session_state["X_test"] if "X_test" in st.session_state else df.drop(columns="Dropout")
    start = st.radio("Point de départ", ["Un étudiant réel du jeu de test", "Un étudiant « typique »"],
                     horizontal=True)
    if start.startswith("Un étudiant réel"):
        idx = st.selectbox("Étudiant (index)", list(pool.index[:300]))
        base = pool.loc[idx].to_dict()
        real = df.loc[idx, "Dropout"] if idx in df.index else None
        if real is not None:
            statut = "a décroché" if real == 1 else "n'a pas décroché"
            st.caption(f"Statut réel connu : {statut}.")
        tag = f"idx{idx}"
    else:
        base, tag = {}, "typique"

    st.markdown("**Variables de l'étudiant** (modifiables)")
    values = {}
    cols = st.columns(3)
    for i, sp in enumerate(specs):
        name, kind = sp["name"], sp["type"]
        default_v = base.get(name, sp.get("default"))
        key = f"{model_id}_{tag}_{name}"
        with cols[i % 3]:
            if kind in ("categorical", "binary"):
                opts = sp["values"]
                if default_v not in opts:
                    default_v = opts[0]
                values[name] = st.selectbox(name, opts, index=opts.index(default_v), key=key)
            elif kind == "numeric":
                conv = int if sp.get("integer") else float
                values[name] = st.number_input(name, min_value=conv(sp["min"]), max_value=conv(sp["max"]),
                                               value=conv(min(max(default_v, sp["min"]), sp["max"])),
                                               step=conv(1) if sp.get("integer") else 0.1, key=key)
            else:
                values[name] = default_v
    values = {k: (v.item() if hasattr(v, "item") else v) for k, v in values.items()}

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
