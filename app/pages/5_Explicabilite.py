import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import (api_call, api_url, load_default_dataset, model_selector, require_api,
                   student_form)

RED, GREEN, GREY = "#c0392b", "#27ae60", "#9aa0a6"

st.title("🔍 Module 5 · Explainable AI")
st.caption("Pourquoi le modèle prédit-il ce qu'il prédit ? Les explications sont calculées par l'API "
           "(SHAP, LIME, permutation) et dessinées ici.")

models = require_api()
model = model_selector(models, key="model5")
model_id = model["id"]
st.caption(f"Modèle analysé : **{model['name']}** · scénario : **{model['scenario']}** · "
           f"documentation de l'API : {api_url()}/docs")

tab_g, tab_l = st.tabs(["🌍 Explications globales (le modèle)", "👤 Explication locale (un étudiant)"])


def labels_of(items):
    return {it["variable"]: it["label"] for it in items}


# =============================================================================
# EXPLICATIONS GLOBALES
# =============================================================================
with tab_g:
    top_n = st.slider("Nombre de variables affichées", 5, 25, 12, key="topn")

    # ---- 1. importance propre au modèle
    st.subheader("1. Importance propre au modèle")
    nat, err = api_call("GET", "/explain/native", params={"model_id": model_id})
    if err:
        st.info(err.replace("Erreur 400 : ", ""))
    else:
        df_n = pd.DataFrame(nat["importances"]).head(top_n).iloc[::-1]
        colors = [GREY if pd.isna(c) else (RED if c > 0 else GREEN) for c in df_n.get("signed_coefficient", [None] * len(df_n))]
        fig = go.Figure(go.Bar(x=df_n["importance"], y=df_n["label"], orientation="h", marker_color=colors))
        fig.update_layout(title=f"{nat['model_name']} : {nat['kind']}", height=max(350, 28 * len(df_n)),
                          xaxis_title="Importance", margin=dict(l=10))
        st.plotly_chart(fig)
        if "signed_coefficient" in df_n and df_n["signed_coefficient"].notna().any():
            st.caption("Rouge : coefficient positif (augmente le risque) · vert : négatif (le réduit) · "
                       "gris : variable catégorielle (plusieurs colonnes regroupées, sans signe unique).")
        else:
            st.caption("L'importance par impureté indique l'usage de la variable par les arbres, "
                       "mais pas le sens de son effet (voir SHAP plus bas).")

    # ---- 2. permutation
    st.subheader("2. Importance par permutation (comparable entre modèles)")
    st.caption("On mélange une variable et on mesure la **baisse d'AUC** : plus elle est forte, plus le modèle "
               "dépend de cette variable. La mesure est la même pour tous les modèles, donc comparable.")
    ids = {m["id"]: f"{m['name']} · {m['scenario']}" for m in models}
    same_scen = [i for i, m in zip(ids, models) if m["scenario"] == model["scenario"]]
    chosen = st.multiselect("Modèles à comparer", list(ids), default=[model_id], format_func=ids.get, key="perm_models")
    if chosen:
        rows = {}
        with st.spinner("Calcul de l'importance par permutation (quelques secondes la première fois)..."):
            for mid in chosen:
                res, err = api_call("GET", "/explain/permutation", params={"model_id": mid}, timeout=300)
                if err:
                    st.error(f"{ids[mid]} : {err}")
                else:
                    rows[mid] = pd.DataFrame(res["importances"]).set_index("label")
        if rows:
            mean_all = pd.concat({k: v["mean"] for k, v in rows.items()}, axis=1).mean(axis=1)
            order = mean_all.sort_values(ascending=False).head(top_n).index[::-1]
            fig = go.Figure()
            for mid, d in rows.items():
                d = d.reindex(order)
                fig.add_trace(go.Bar(y=order, x=d["mean"], orientation="h", name=ids[mid],
                                     error_x=dict(type="data", array=d["std"])))
            fig.update_layout(barmode="group", title="Baisse d'AUC-ROC quand la variable est mélangée",
                              height=max(380, 34 * len(order) * max(1, len(rows) * 0.6)),
                              xaxis_title="Baisse d'AUC", legend=dict(orientation="h", y=-0.15), margin=dict(l=10))
            st.plotly_chart(fig)

    # ---- 3. SHAP beeswarm
    st.subheader("3. SHAP summary (beeswarm)")
    st.caption("Chaque point est un étudiant. **À droite de 0** : la variable pousse vers le décrochage ; "
               "**à gauche** : elle en protège. La couleur indique la valeur de la variable (rouge = élevée).")
    shp, err = api_call("GET", "/explain/shap/global", params={"model_id": model_id, "top": 15}, timeout=300)
    if err:
        st.info(err.replace("Erreur 400 : ", ""))
    else:
        vars_ = shp["variables"][:top_n]
        rng = np.random.default_rng(0)
        fig = go.Figure()
        for pos, v in enumerate(reversed(vars_)):
            n = len(v["shap"])
            y = pos + rng.uniform(-0.3, 0.3, n)
            if v["color"] is not None:
                marker = dict(color=v["color"], colorscale="RdBu_r", cmin=0, cmax=1, size=5, opacity=0.75)
            else:
                marker = dict(color=GREY, size=5, opacity=0.6)
            hover = [f"{v['label']} = {x:g}<br>SHAP = {s:+.3f}" for x, s in zip(v["values"], v["shap"])]
            fig.add_trace(go.Scatter(x=v["shap"], y=y, mode="markers", marker=marker, showlegend=False,
                                     hovertext=hover, hoverinfo="text"))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(height=max(400, 34 * len(vars_)), margin=dict(l=10),
                          xaxis_title=f"Valeur SHAP ({shp['unit']}) : impact sur la prédiction",
                          yaxis=dict(tickmode="array", tickvals=list(range(len(vars_))),
                                     ticktext=[v["label"] for v in reversed(vars_)]))
        st.plotly_chart(fig)
        st.caption("Variables triées par impact moyen. Points gris : variables catégorielles (codes, sans ordre). "
                   f"Calcul sur {shp['n_samples']} étudiants du jeu de test.")

        # ---- 4. dépendance
        st.subheader("4. Graphiques de dépendance SHAP")
        st.caption("Comment l'effet d'une variable change selon sa valeur.")
        names = {v["label"]: v for v in shp["variables"]}
        pick = st.selectbox("Variable", list(names), key="dep")
        v = names[pick]
        d = pd.DataFrame({"valeur": v["values"], "shap": v["shap"]})
        if v["type"] in ("categorical", "binary"):
            d["valeur"] = d["valeur"].astype(int).astype(str)
            fig = px.strip(d, x="valeur", y="shap", color_discrete_sequence=[RED])
            fig.update_xaxes(type="category")
        else:
            fig = px.scatter(d, x="valeur", y="shap", opacity=0.6, color_discrete_sequence=[RED])
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(title=f"Dépendance SHAP : {pick}", xaxis_title=pick,
                          yaxis_title=f"Valeur SHAP ({shp['unit']})")
        st.plotly_chart(fig)


# =============================================================================
# EXPLICATION LOCALE
# =============================================================================
with tab_l:
    st.markdown("Choisis un étudiant du jeu de test, ou modifie ses variables pour créer un **étudiant fictif**.")
    specs, err = api_call("GET", "/features", params={"model_id": model_id})
    if err:
        st.error(err)
        st.stop()
    df, _ = load_default_dataset()
    pool = st.session_state["X_test"] if "X_test" in st.session_state else df.drop(columns="Dropout")
    values = student_form(specs, df, pool, f"p5_{model_id}")
    threshold = st.slider("Seuil de décision", 0.05, 0.95, float(model["threshold"]), 0.01, key="thr5")

    if st.button("🔎 Expliquer cet étudiant", type="primary"):
        with st.spinner("SHAP et LIME en cours de calcul..."):
            res, err = api_call("POST", "/explain/compare", timeout=300,
                                json={"model_id": model_id, "features": values, "threshold": threshold})
        if err:
            st.error(err.replace("Erreur 400 : ", ""))
        else:
            st.session_state["explain_res"] = res

    res = st.session_state.get("explain_res")
    if res and res["shap"]["model_id"] == model_id:
        s, l, ag = res["shap"], res["lime"], res["agreement"]

        # ---- prédiction + phrase de synthèse
        a, b, c = st.columns(3)
        a.metric("Probabilité de décrochage", f"{s['probability']:.1%}")
        b.metric("Seuil de décision", f"{s['threshold']:.2f}")
        c.metric("Décision", "À risque" if s["prediction"] else "Risque faible")
        (st.error if s["prediction"] else st.success)(f"**Synthèse :** {res['summary']}")

        # ---- SHAP waterfall
        st.subheader("SHAP : contribution de chaque variable")
        contribs = s["contributions"]
        top = contribs[:10]
        rest = contribs[10:]
        labels = [f"{c['label']} = {c['display']}" for c in top]
        vals = [c["shap"] for c in top]
        if rest:
            labels.append(f"Autres variables ({len(rest)})")
            vals.append(sum(c["shap"] for c in rest))
        x = [s["base_value"]] + vals + [s["output_value"]]
        y = ["Valeur moyenne (référence)"] + labels + ["Prédiction de l'étudiant"]
        measure = ["absolute"] + ["relative"] * len(vals) + ["total"]
        fig = go.Figure(go.Waterfall(
            orientation="h", measure=measure, x=x, y=y,
            increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=GREEN)),
            totals=dict(marker=dict(color="#34495e")), connector=dict(line=dict(color="#bbb")),
            text=[f"{v:+.2f}" if m == "relative" else f"{v:.2f}" for v, m in zip(x, measure)],
            textposition="outside"))
        fig.update_layout(height=max(420, 36 * len(y)), margin=dict(l=10, r=40),
                          xaxis_title=f"Sortie du modèle ({s['unit']})")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig)
        extra = (" (en log-odds : probabilité = 1 / (1 + e^-x))" if s["unit"] == "log-odds" else "")
        st.caption(f"Rouge : pousse vers le décrochage · vert : en protège. On part de la valeur moyenne "
                   f"(probabilité de référence {s['base_probability']:.0%}) pour arriver à la prédiction "
                   f"({s['probability']:.0%}){extra}.")

        # ---- LIME
        st.subheader("LIME : explication par un modèle linéaire local")
        dl = pd.DataFrame(l["contributions"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=dl["weight"], y=dl["condition"], orientation="h",
                               marker_color=[RED if w > 0 else GREEN for w in dl["weight"]]))
        fig.update_layout(height=max(350, 34 * len(dl)), margin=dict(l=10),
                          xaxis_title="Poids LIME (positif = pousse vers le décrochage)")
        st.plotly_chart(fig)
        st.caption(f"Fidélité du modèle local (R²) : **{l['local_model_r2']:.2f}**. "
                   "Plus elle est proche de 1, plus l'approximation locale est fiable.")

        # ---- comparaison
        st.subheader("Comparaison SHAP / LIME")
        sd = {c["variable"]: c for c in contribs}
        rows = []
        for it in l["contributions"]:
            v = it["variable"]
            sv = sd[v]["shap"] if v in sd else None
            rows.append({"Variable": it["label"], "Condition LIME": it["condition"],
                         "Poids LIME": round(it["weight"], 3),
                         "Valeur SHAP": None if sv is None else round(sv, 3),
                         "Même sens ?": "—" if sv is None else ("✅" if (sv > 0) == (it["weight"] > 0) else "❌")})
        st.dataframe(pd.DataFrame(rows))
        k, cm = ag["top_k"], ag["common_in_top_k"]
        pct = ag["sign_agreement"]
        st.write(f"- Variables communes dans le **top {k}** de SHAP et de LIME : **{cm} / {k}**")
        if pct is not None:
            st.write(f"- Variables comparées avec le **même sens** : **{ag['same_sign']} / {ag['variables_compared']}** ({pct:.0%})")
        if pct is not None and pct >= 0.8 and cm >= 3:
            st.success("Les deux méthodes sont **cohérentes** : elles retiennent les mêmes facteurs principaux, dans le même sens.")
        elif pct is not None and pct >= 0.6:
            st.warning("Les deux méthodes sont **partiellement cohérentes** : les facteurs principaux se recoupent, "
                       "mais certains détails diffèrent.")
        else:
            st.error("Les deux méthodes **divergent**. Cela arrive : LIME approche le modèle par un modèle linéaire "
                     "sur des données perturbées (résultat un peu aléatoire, dépend de la fidélité locale), "
                     "alors que SHAP répartit exactement la prédiction entre les variables.")
        with st.expander("Comment lire ces explications ?"):
            st.markdown(
                "- **SHAP** répartit la prédiction entre toutes les variables de façon additive et cohérente : "
                "la somme des contributions donne exactement l'écart avec la valeur moyenne.\n"
                "- **LIME** ajuste un petit modèle linéaire autour de cet étudiant (en perturbant ses valeurs) : "
                "il décrit le comportement local, mais son résultat peut légèrement varier.\n"
                "- Une explication est **fiable** si SHAP et LIME s'accordent, et si elle est **plausible** "
                "pour un expert (frais impayés, peu de matières validées → plus de risque).")
