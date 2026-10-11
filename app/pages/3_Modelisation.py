import time

import pandas as pd
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from utils import (SEARCH_GRIDS, build_estimator, build_prep, metrics_at, model_names,
                   require)

st.title("🤖 Module 3 · Modélisation et comparaison")
require(["X_train"], "Termine d'abord la page **2 · Préparation** (bouton « Valider »).")
s = st.session_state

# ------------------------------------------------------------------ choix
chosen = st.multiselect("Modèles à entraîner", model_names(),
                        default=["Régression logistique", "Random Forest"] if "XGBoost" not in model_names()
                        else ["Régression logistique", "Random Forest", "XGBoost"])
balance = st.checkbox("Compenser le déséquilibre des classes (class_weight / scale_pos_weight)", True)
mode = st.radio("Hyperparamètres",
                ["Manuels (réglés ci-dessous)", "Optimisation automatique (recherche aléatoire + validation croisée)"],
                horizontal=True)
auto = mode.startswith("Optimisation")

params = {}
if auto:
    c1, c2 = st.columns(2)
    n_iter = c1.slider("Combinaisons testées par modèle", 5, 60, 15)
    n_folds = c2.slider("Plis de validation croisée", 3, 10, 5)
    st.caption("Critère optimisé : AUC-ROC. La recherche n'utilise que le jeu d'entraînement.")
else:
    for name in chosen:
        with st.expander(f"Hyperparamètres : {name}", expanded=False):
            p = {}
            if name == "Régression logistique":
                p["C"] = st.select_slider("C (inverse de la régularisation)",
                                          [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0], 0.25, key=f"{name}C")
            elif name == "Arbre de décision":
                p["max_depth"] = st.slider("Profondeur maximale", 2, 20, 8, key=f"{name}d")
                p["min_samples_leaf"] = st.slider("Effectif minimal par feuille", 1, 60, 20, key=f"{name}l")
            elif name == "KNN":
                p["n_neighbors"] = st.slider("Nombre de voisins", 3, 51, 15, step=2, key=f"{name}k")
                p["weights"] = st.selectbox("Pondération", ["distance", "uniform"], key=f"{name}w")
                p["p"] = st.selectbox("Distance (1 = Manhattan, 2 = euclidienne)", [1, 2], key=f"{name}p")
            elif name == "Random Forest":
                p["n_estimators"] = st.slider("Nombre d'arbres", 50, 500, 200, step=50, key=f"{name}n")
                p["max_depth"] = st.slider("Profondeur maximale (0 = illimitée)", 0, 25, 0, key=f"{name}d")
                p["min_samples_leaf"] = st.slider("Effectif minimal par feuille", 1, 20, 3, key=f"{name}l")
            elif name == "XGBoost":
                p["n_estimators"] = st.slider("Nombre d'arbres", 50, 500, 300, step=50, key=f"{name}n")
                p["max_depth"] = st.slider("Profondeur maximale", 2, 8, 3, key=f"{name}d")
                p["learning_rate"] = st.select_slider("Taux d'apprentissage",
                                                      [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.05, key=f"{name}lr")
                p["subsample"] = st.slider("Part des lignes par arbre", 0.5, 1.0, 0.8, step=0.05, key=f"{name}s")
                p["colsample_bytree"] = st.slider("Part des colonnes par arbre", 0.5, 1.0, 0.8, step=0.05,
                                                  key=f"{name}c")
            params[name] = p

# ------------------------------------------------------------------ entraînement
if st.button("🚀 Entraîner les modèles", type="primary", disabled=not chosen):
    X_tr, y_tr = s["X_train"], s["y_train"]
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    fitted, best_params, cv_auc, times = {}, {}, {}, {}
    bar = st.progress(0.0)
    for i, name in enumerate(chosen):
        t0 = time.time()
        with st.spinner(f"Entraînement : {name}..."):
            est = build_estimator(name, params.get(name, {}), balance, spw, s["seed"])
            pipe = Pipeline([("prep", build_prep(s["prep_cfg"], X_tr.columns)), ("clf", est)])
            if auto:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s["seed"])
                search = RandomizedSearchCV(pipe, SEARCH_GRIDS[name], n_iter=n_iter, scoring="roc_auc",
                                            cv=cv, random_state=s["seed"], n_jobs=-1)
                search.fit(X_tr, y_tr)
                fitted[name] = search.best_estimator_
                best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                cv_auc[name] = search.best_score_
            else:
                pipe.fit(X_tr, y_tr)
                fitted[name] = pipe
                best_params[name] = params.get(name, {})
        times[name] = time.time() - t0
        bar.progress((i + 1) / len(chosen))

    probas = {n: m.predict_proba(s["X_test"])[:, 1] for n, m in fitted.items()}
    table = pd.DataFrame({n: metrics_at(s["y_test"], p) for n, p in probas.items()}).T
    if cv_auc:
        table.insert(0, "AUC (CV train)", pd.Series(cv_auc))
    table["Temps (s)"] = pd.Series(times)
    s.update(fitted=fitted, probas=probas, table=table.sort_values("AUC-ROC", ascending=False),
             best_params=best_params)

# ------------------------------------------------------------------ résultats
if "fitted" in s:
    st.subheader("Tableau comparatif (jeu de test, seuil 0,5)")
    show = s["table"].drop(columns=["Gini", "KS"], errors="ignore").round(3)
    st.dataframe(show.style.highlight_max(subset=["AUC-ROC", "PR-AUC", "F1"], color="#1f6f4a"))
    best = show.index[0]
    st.success(f"Meilleur modèle selon l'AUC-ROC : **{best}** ({show.loc[best, 'AUC-ROC']:.3f})")
    with st.expander("Hyperparamètres utilisés"):
        st.json({k: {a: (b if isinstance(b, (int, float, str, type(None))) else float(b))
                     for a, b in v.items()} for k, v in s["best_params"].items()})
    st.write("Étape suivante : page **4 · Évaluation**.")
