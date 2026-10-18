"""Routes d'explicabilité : importance native, permutation, SHAP (global / local), LIME.

L'API calcule les explications et renvoie des données (JSON) ; l'application Streamlit les dessine.
Toutes les explications sont données **par variable d'origine** (les colonnes one-hot d'une même
variable sont regroupées : leurs contributions SHAP s'additionnent).
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

TARGET = "Dropout"

LABELS = {
    "Curricular units 1st sem (approved)": "Matières validées au S1",
    "Curricular units 1st sem (grade)": "Note moyenne au S1",
    "Curricular units 1st sem (enrolled)": "Matières inscrites au S1",
    "Curricular units 1st sem (evaluations)": "Évaluations passées au S1",
    "Curricular units 1st sem (credited)": "Matières créditées au S1",
    "Curricular units 1st sem (without evaluations)": "Matières sans évaluation au S1",
    "Curricular units 2nd sem (approved)": "Matières validées au S2",
    "Curricular units 2nd sem (grade)": "Note moyenne au S2",
    "Curricular units 2nd sem (enrolled)": "Matières inscrites au S2",
    "Curricular units 2nd sem (evaluations)": "Évaluations passées au S2",
    "Curricular units 2nd sem (credited)": "Matières créditées au S2",
    "Curricular units 2nd sem (without evaluations)": "Matières sans évaluation au S2",
    "Tuition fees up to date": "Frais de scolarité à jour",
    "Debtor": "Endetté",
    "Scholarship holder": "Boursier",
    "Age at enrollment": "Âge à l'inscription",
    "Admission grade": "Note d'admission",
    "Previous qualification (grade)": "Note de la qualification précédente",
    "Previous qualification": "Qualification précédente",
    "Gender": "Genre",
    "Course": "Formation",
    "Application mode": "Mode de candidature",
    "Application order": "Ordre de candidature",
    "Marital Status": "État civil",
    "Displaced": "Étudiant déplacé",
    "Daytime/evening attendance": "Cours de jour",
    "International": "Étudiant international",
    "Nationality": "Nationalité",
    "Mother's qualification": "Études de la mère",
    "Father's qualification": "Études du père",
    "Mother's occupation": "Profession de la mère",
    "Father's occupation": "Profession du père",
    "Unemployment rate": "Taux de chômage",
    "Inflation rate": "Taux d'inflation",
    "GDP": "PIB",
}


def label(v: str) -> str:
    return LABELS.get(v, v)


class ExplainRequest(BaseModel):
    model_id: str = Field(..., description="Identifiant du modèle (voir GET /models)")
    features: Dict[str, Any] = Field(..., description="Valeurs des variables de l'étudiant "
                                     "(les absentes reçoivent une valeur typique)")
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Seuil de décision")

    model_config = {"json_schema_extra": {"example": {
        "model_id": "xgboost", "features": {"Debtor": 1, "Tuition fees up to date": 0}, "threshold": 0.35}}}


# --------------------------------------------------------------------------- utilitaires
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _family(clf) -> Optional[str]:
    n = type(clf).__name__
    return {"LogisticRegression": "linear", "XGBClassifier": "xgb",
            "DecisionTreeClassifier": "tree", "RandomForestClassifier": "tree"}.get(n)


def _owner(name: str, cols: List[str]) -> str:
    """Retrouve la variable d'origine d'une colonne transformée (ex. cat__Course_9254 -> Course)."""
    kind, _, rest = name.partition("__")
    if kind == "cat":
        cands = [c for c in cols if rest.startswith(c + "_")]
        if cands:
            return max(cands, key=len)
    return rest


def _fmt(spec: dict, v) -> str:
    t = spec.get("type")
    if t == "binary":
        return "oui" if float(v) == 1 else "non"
    if t == "categorical":
        return f"code {int(float(v))}"
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:.1f}"


def build_sentence(contribs: List[dict], proba: float, threshold: float) -> str:
    """Phrase de synthèse en français à partir des contributions SHAP."""
    def d(c):
        return f"{c['label']} ({c['display']})"
    pos = [c for c in contribs if c["shap"] > 0][:3]     # poussent vers le décrochage
    neg = [c for c in contribs if c["shap"] < 0][:3]     # protègent
    join = lambda L: ", ".join(d(c) for c in L[:-1]) + (" et " if len(L) > 1 else "") + d(L[-1])
    if proba >= threshold:
        s = f"Le risque de décrochage est élevé ({proba:.0%}, seuil {threshold:.0%})"
        if pos:
            s += f", principalement à cause de : {join(pos)}"
        if neg:
            s += f" ; il est atténué par : {join(neg[:2])}"
    else:
        s = f"Le risque de décrochage est faible ({proba:.0%}, seuil {threshold:.0%})"
        if neg:
            s += f", principalement grâce à : {join(neg)}"
        if pos:
            s += f" ; les principaux facteurs de risque restent : {join(pos[:2])}"
    return s + "."


def build_router(registry, to_frame, decision_label) -> APIRouter:
    router = APIRouter(prefix="/explain", tags=["Explicabilité"])

    # ---------------------------------------------------------------- accès aux données
    def split():
        if registry.df is None:
            raise HTTPException(503, "Jeu de données introuvable (data/students_dropout.csv) : "
                                     "les explications en ont besoin.")
        if registry._split is None:
            X, y = registry.df.drop(columns=TARGET), registry.df[TARGET]
            registry._split = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        return registry._split

    def cached(key, fn):
        if key not in registry.cache:
            registry.cache[key] = fn()
        return registry.cache[key]

    def need_shap():
        try:
            import shap  # noqa: F401
            return shap
        except ImportError:
            raise HTTPException(501, "La bibliothèque « shap » n'est pas installée sur le serveur.")

    # ---------------------------------------------------------------- SHAP (calcul)
    def shap_by_variable(m: dict, X: pd.DataFrame):
        """Renvoie (matrice (n × variables), noms des variables, valeur de base, unité)."""
        pipe = m["model"]
        prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
        fam = _family(clf)
        if fam is None:
            raise HTTPException(400, f"SHAP n'est pas proposé pour le modèle « {m['name']} » "
                                     "(KNN : calcul trop lent). Choisis un autre modèle.")
        cols = m["columns"]
        Xt = np.asarray(prep.transform(X[cols]), dtype=float)
        names = list(prep.get_feature_names_out())
        unit = "log-odds" if fam in ("linear", "xgb") else "probabilité"

        if fam == "linear":
            shap = need_shap()
            X_train = split()[0]
            bg = np.asarray(prep.transform(X_train[cols].sample(min(100, len(X_train)), random_state=0)), dtype=float)
            ex = shap.LinearExplainer(clf, bg)
            sv, base = np.asarray(ex.shap_values(Xt)), float(np.ravel(ex.expected_value)[0])
        else:
            sv = base = None
            try:
                shap = need_shap()
                ex = shap.TreeExplainer(clf)
                raw = ex.shap_values(Xt, check_additivity=False)
                ev = np.ravel(ex.expected_value)
                sv = raw[1] if isinstance(raw, list) else np.asarray(raw)
                base = float(ev[-1] if len(ev) > 1 else ev[0])
            except HTTPException:
                raise
            except Exception:
                if fam != "xgb":
                    raise
            if sv is None and fam == "xgb":          # repli : TreeSHAP natif de XGBoost
                import xgboost
                contrib = clf.get_booster().predict(xgboost.DMatrix(Xt), pred_contribs=True)
                sv, base = contrib[:, :-1], float(contrib[0, -1])
            if sv.ndim == 3:
                sv = sv[:, :, 1]

        owners = [_owner(n, cols) for n in names]
        variables = [c for c in cols if c in set(owners)]
        agg = np.zeros((sv.shape[0], len(variables)))
        for j, o in enumerate(owners):
            agg[:, variables.index(o)] += sv[:, j]
        return agg, variables, base, unit

    def out_proba(base, total, unit):
        return _sigmoid(base + total) if unit == "log-odds" else base + total

    def local_shap(m, features, threshold):
        X, filled, _ = to_frame(m, [features])
        agg, variables, base, unit = shap_by_variable(m, X)
        proba = float(m["model"].predict_proba(X)[0, 1])
        t = threshold if threshold is not None else m["threshold"]
        contribs = []
        for j, v in enumerate(variables):
            spec = registry.specs.get(v, {"type": "numeric"})
            raw = X[v].iloc[0]
            contribs.append({"variable": v, "label": label(v), "value": float(raw),
                             "display": _fmt(spec, raw), "shap": float(agg[0, j])})
        contribs.sort(key=lambda c: -abs(c["shap"]))
        return {"model_id": m["id"], "model_name": m["name"], "scenario": m["scenario"],
                "probability": round(proba, 4), "threshold": t, "prediction": int(proba >= t),
                "decision": decision_label(proba, t), "unit": unit,
                "base_value": base, "output_value": base + float(agg[0].sum()),
                "base_probability": out_proba(base, 0.0, unit),
                "contributions": contribs, "filled_with_defaults": filled,
                "summary": build_sentence(contribs, proba, t)}

    # ---------------------------------------------------------------- LIME (calcul)
    def lime_explainer(m):
        def make():
            from lime.lime_tabular import LimeTabularExplainer
            cols = m["columns"]
            X_train = split()[0][cols]
            cat_idx = [i for i, c in enumerate(cols)
                       if registry.specs.get(c, {}).get("type") in ("categorical", "binary")]
            return LimeTabularExplainer(X_train.values.astype(float), feature_names=cols,
                                        categorical_features=cat_idx,
                                        class_names=["Pas de décrochage", "Décrochage"],
                                        discretize_continuous=True, mode="classification",
                                        random_state=0)
        return cached(("lime_explainer", m["id"]), make)

    def local_lime(m, features, threshold, num_features=10, num_samples=2000):
        try:
            import lime  # noqa: F401
        except ImportError:
            raise HTTPException(501, "La bibliothèque « lime » n'est pas installée sur le serveur.")
        X, filled, _ = to_frame(m, [features])
        cols = m["columns"]
        pipe = m["model"]

        def predict_fn(arr):
            return pipe.predict_proba(pd.DataFrame(arr, columns=cols))

        exp = lime_explainer(m).explain_instance(X[cols].iloc[0].values.astype(float), predict_fn,
                                                 num_features=num_features, num_samples=num_samples,
                                                 labels=(1,))
        proba = float(pipe.predict_proba(X)[0, 1])
        t = threshold if threshold is not None else m["threshold"]
        items = []
        for (idx, w), (desc, _) in zip(exp.local_exp[1], exp.as_list(label=1)):
            v = cols[idx]
            items.append({"variable": v, "label": label(v), "condition": desc, "weight": float(w)})
        return {"model_id": m["id"], "model_name": m["name"], "probability": round(proba, 4),
                "threshold": t, "prediction": int(proba >= t),
                "local_prediction": float(exp.local_pred[0]) if np.ndim(exp.local_pred) else float(exp.local_pred),
                "local_model_r2": float(exp.score), "contributions": items,
                "note": "Un poids positif pousse vers le décrochage, un poids négatif en protège."}

    def agreement(shap_c, lime_c, k=5):
        sd = {c["variable"]: c["shap"] for c in shap_c}
        ld = {c["variable"]: c["weight"] for c in lime_c}
        top_s = [c["variable"] for c in shap_c[:k]]
        top_l = [c["variable"] for c in sorted(lime_c, key=lambda c: -abs(c["weight"]))[:k]]
        common = [v for v in top_s if v in top_l]
        inter = [v for v in ld if v in sd]
        same = [v for v in inter if (sd[v] > 0) == (ld[v] > 0)]
        return {"top_k": k, "common_in_top_k": len(common), "common_variables": common,
                "variables_compared": len(inter), "same_sign": len(same),
                "sign_agreement": (len(same) / len(inter)) if inter else None}

    # ---------------------------------------------------------------- routes globales
    @router.get("/native", summary="Importance propre au modèle (coefficients ou impureté)")
    def native(model_id: str = Query(...)):
        m = registry.get(model_id)

        def compute():
            pipe = m["model"]
            prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
            cols = m["columns"]
            names = list(prep.get_feature_names_out())
            if hasattr(clf, "coef_"):
                vals, kind = np.abs(np.ravel(clf.coef_)), "|coefficient| (variables standardisées)"
            elif hasattr(clf, "feature_importances_"):
                vals, kind = np.asarray(clf.feature_importances_), "importance par impureté / gain"
            else:
                raise HTTPException(400, f"Pas d'importance propre pour « {m['name']} » (KNN) : "
                                         "utiliser l'importance par permutation.")
            agg: Dict[str, float] = {}
            for n, v in zip(names, vals):
                o = _owner(n, cols)
                agg[o] = agg.get(o, 0.0) + float(v)
            items = [{"variable": k, "label": label(k), "importance": v}
                     for k, v in sorted(agg.items(), key=lambda kv: -kv[1])]
            extra = {}
            if hasattr(clf, "coef_"):      # signe du coefficient pour les variables numériques / binaires
                coef = dict(zip(names, np.ravel(clf.coef_)))
                extra = {_owner(n, cols): float(c) for n, c in coef.items() if not n.startswith("cat__")}
            for it in items:
                it["signed_coefficient"] = extra.get(it["variable"])
            return {"model_id": m["id"], "model_name": m["name"], "kind": kind, "importances": items}
        return cached(("native", model_id), compute)

    @router.get("/permutation", summary="Importance par permutation (comparable entre modèles)")
    def permutation(model_id: str = Query(...), n_repeats: int = Query(5, ge=1, le=20)):
        m = registry.get(model_id)

        def compute():
            _, X_test, _, y_test = split()
            cols = m["columns"]
            r = permutation_importance(m["model"], X_test[cols], y_test, scoring="roc_auc",
                                       n_repeats=n_repeats, random_state=0, n_jobs=1)
            items = [{"variable": c, "label": label(c), "mean": float(r.importances_mean[i]),
                      "std": float(r.importances_std[i])} for i, c in enumerate(cols)]
            items.sort(key=lambda d: -d["mean"])
            return {"model_id": m["id"], "model_name": m["name"], "metric": "baisse d'AUC-ROC",
                    "n_repeats": n_repeats, "importances": items}
        return cached(("perm", model_id, n_repeats), compute)

    @router.get("/shap/global", summary="Données du graphique SHAP summary (beeswarm) et de dépendance")
    def shap_global(model_id: str = Query(...), n_samples: int = Query(300, ge=50, le=800),
                    top: int = Query(15, ge=3, le=30)):
        m = registry.get(model_id)

        def compute():
            _, X_test, _, _ = split()
            cols = m["columns"]
            Xs = X_test[cols].sample(min(n_samples, len(X_test)), random_state=0)
            agg, variables, base, unit = shap_by_variable(m, Xs)
            order = np.argsort(-np.abs(agg).mean(axis=0))[:top]
            out = []
            for j in order:
                v = variables[j]
                spec = registry.specs.get(v, {"type": "numeric"})
                x = Xs[v].astype(float)
                color = None
                if spec.get("type") in ("numeric", "binary") and x.nunique() > 1:
                    color = [float(c) for c in x.rank(pct=True).values]
                out.append({"variable": v, "label": label(v), "type": spec.get("type"),
                            "mean_abs_shap": float(np.abs(agg[:, j]).mean()),
                            "values": [float(a) for a in x.values],
                            "shap": [float(a) for a in agg[:, j]], "color": color})
            return {"model_id": m["id"], "model_name": m["name"], "unit": unit,
                    "n_samples": int(len(Xs)), "variables": out}
        return cached(("shap_global", model_id, n_samples, top), compute)

    # ---------------------------------------------------------------- routes locales
    @router.post("/shap", summary="SHAP local : contribution de chaque variable pour un étudiant")
    def shap_local(req: ExplainRequest):
        return local_shap(registry.get(req.model_id), req.features, req.threshold)

    @router.post("/lime", summary="LIME local : explication d'un étudiant par un modèle linéaire local")
    def lime_local(req: ExplainRequest):
        return local_lime(registry.get(req.model_id), req.features, req.threshold)

    @router.post("/compare", summary="SHAP + LIME + phrase de synthèse + cohérence, pour un étudiant")
    def compare(req: ExplainRequest):
        m = registry.get(req.model_id)
        s = local_shap(m, req.features, req.threshold)
        l = local_lime(m, req.features, req.threshold)
        return {"shap": s, "lime": l, "agreement": agreement(s["contributions"], l["contributions"]),
                "summary": s["summary"]}

    return router
