"""API d'interprétation : prédiction (et, plus tard, explications SHAP / LIME).

Lancer :  uvicorn api.main:app --reload --port 8000
Docs   :  http://127.0.0.1:8000/docs   (Swagger, généré automatiquement)
"""
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DATA_CSV = Path(os.getenv("DATA_CSV", ROOT / "data" / "students_dropout.csv"))
TARGET = "Dropout"

CAT_COLS = ["Marital Status", "Application mode", "Course", "Previous qualification",
            "Nationality", "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation"]
STEM_NAMES = {"regression_logistique": "Régression logistique", "arbre_de_decision": "Arbre de décision",
              "knn": "KNN", "random_forest": "Random Forest", "xgboost": "XGBoost"}

app = FastAPI(
    title="API de prédiction du décrochage étudiant",
    description=("Service appelé par l'application Streamlit. Il charge les modèles entraînés "
                 "(dossier `models/`) et renvoie la probabilité de décrochage d'un étudiant. "
                 "Les routes d'explication (SHAP, LIME) seront ajoutées avec le Module 5."),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- schémas
class PredictRequest(BaseModel):
    model_id: str = Field(..., description="Identifiant du modèle (voir GET /models)")
    features: Dict[str, Any] = Field(
        ..., description="Valeurs des variables de l'étudiant. Les variables absentes "
                         "reçoivent leur valeur typique (médiane ou modalité la plus fréquente).")
    threshold: Optional[float] = Field(None, ge=0, le=1,
                                       description="Seuil de décision. Par défaut : seuil du modèle.")

    model_config = {"json_schema_extra": {"example": {
        "model_id": "xgboost",
        "features": {"Age at enrollment": 22, "Debtor": 1, "Tuition fees up to date": 0,
                     "Curricular units 1st sem (approved)": 2},
        "threshold": 0.35}}}


class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    scenario: str
    probability: float = Field(..., description="Probabilité de décrochage (classe positive)")
    threshold: float
    prediction: int = Field(..., description="1 si probabilité >= seuil, sinon 0")
    decision: str
    filled_with_defaults: List[str] = Field(..., description="Variables non fournies, remplacées par une valeur typique")
    ignored_features: List[str] = Field(..., description="Variables fournies mais inconnues du modèle")


class BatchRequest(BaseModel):
    model_id: str
    rows: List[Dict[str, Any]]
    threshold: Optional[float] = Field(None, ge=0, le=1)


# --------------------------------------------------------------------------- chargement
def slug(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


class Registry:
    def __init__(self):
        self.models: Dict[str, dict] = {}
        self.specs: Dict[str, dict] = {}
        self.load()

    def load(self):
        self.models.clear()
        metrics_base, metrics_final = {}, {}
        p = MODELS_DIR / "comparatif_modeles.csv"
        if p.exists():
            metrics_base = pd.read_csv(p, index_col=0).to_dict("index")
        p = MODELS_DIR / "resultats_finaux.csv"
        if p.exists():
            for r in pd.read_csv(p).to_dict("records"):
                metrics_final[(r["Scénario"], r["Modèle"])] = r

        # 1) modèles de modelisation.ipynb (toutes les variables, seuil 0,5)
        for f in sorted(MODELS_DIR.glob("*.joblib")):
            if f.stem in ("meta", "modeles_finaux"):
                continue
            name = STEM_NAMES.get(f.stem, f.stem)
            model = joblib.load(f)
            cols = list(getattr(model, "feature_names_in_", []))
            self.models[f.stem] = {"id": f.stem, "name": name, "scenario": "Toutes les variables",
                                   "model": model, "columns": cols, "threshold": 0.5,
                                   "metrics": self._clean(metrics_base.get(name, {}))}

        # 2) modèles de evaluation_finale.ipynb (par scénario, seuil choisi)
        f = MODELS_DIR / "modeles_finaux.joblib"
        if f.exists():
            for (scen, name), d in joblib.load(f).items():
                mid = f"{slug(name)}__{slug(scen)}"
                self.models[mid] = {"id": mid, "name": name, "scenario": scen, "model": d["model"],
                                    "columns": list(d["columns"]), "threshold": float(d["threshold"]),
                                    "metrics": self._clean(metrics_final.get((scen, name), {}))}
        self._build_specs()

    @staticmethod
    def _clean(d):
        keep = ["AUC-ROC", "PR-AUC", "Précision", "Rappel", "F1"]
        return {k: round(float(d[k]), 3) for k in keep if k in d and pd.notna(d[k])}

    def _build_specs(self):
        """Description de chaque variable (type, bornes, valeurs possibles, valeur par défaut)."""
        self.specs = {}
        if not DATA_CSV.exists():
            return
        df = pd.read_csv(DATA_CSV)
        for c in df.columns:
            if c == TARGET:
                continue
            s = df[c]
            if c in CAT_COLS:
                vals = sorted(s.unique().tolist())
                self.specs[c] = {"name": c, "type": "categorical", "values": vals,
                                 "default": s.mode().iloc[0].item()}
            elif s.nunique() == 2:
                self.specs[c] = {"name": c, "type": "binary", "values": sorted(s.unique().tolist()),
                                 "default": s.mode().iloc[0].item()}
            else:
                is_int = bool(np.all(np.mod(s, 1) == 0))
                conv = int if is_int else float
                self.specs[c] = {"name": c, "type": "numeric", "integer": is_int,
                                 "min": conv(s.min()), "max": conv(s.max()),
                                 "default": conv(round(s.median()) if is_int else s.median())}

    def get(self, model_id: str) -> dict:
        if model_id not in self.models:
            raise HTTPException(404, f"Modèle inconnu : {model_id}. Voir GET /models.")
        return self.models[model_id]


registry = Registry()


# --------------------------------------------------------------------------- prédiction
def to_frame(m: dict, rows: List[Dict[str, Any]]):
    cols = m["columns"]
    filled, ignored = set(), set()
    records = []
    for features in rows:
        rec = {}
        for c in cols:
            if c in features and features[c] is not None:
                rec[c] = features[c]
            elif c in registry.specs:
                rec[c] = registry.specs[c]["default"]
                filled.add(c)
            else:
                raise HTTPException(422, f"Variable manquante sans valeur par défaut : {c}")
        ignored |= set(features) - set(cols)
        records.append(rec)
    df = pd.DataFrame(records, columns=cols)
    for c in cols:
        if registry.specs.get(c, {}).get("type") == "numeric" or \
                registry.specs.get(c, {}).get("type") == "binary":
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                raise HTTPException(422, f"La variable « {c} » doit être numérique.")
    return df, sorted(filled), sorted(ignored)


def decision_label(p: float, t: float) -> str:
    return "Risque ÉLEVÉ de décrochage" if p >= t else "Risque faible de décrochage"


@app.get("/", include_in_schema=False)
def root():
    return {"message": "API de prédiction du décrochage étudiant", "docs": "/docs"}


@app.get("/health", tags=["Service"], summary="État du service")
def health():
    return {"status": "ok", "n_models": len(registry.models),
            "dataset_specs": len(registry.specs)}


@app.post("/reload", tags=["Service"], summary="Recharger les modèles depuis le disque")
def reload_models():
    registry.load()
    return {"status": "reloaded", "n_models": len(registry.models)}


@app.get("/models", tags=["Modèles"], summary="Liste des modèles disponibles")
def list_models():
    return [{"id": m["id"], "name": m["name"], "scenario": m["scenario"],
             "threshold": m["threshold"], "n_features": len(m["columns"]),
             "metrics": m["metrics"]} for m in registry.models.values()]


@app.get("/features", tags=["Modèles"], summary="Variables attendues par un modèle")
def features(model_id: str = Query(..., description="Identifiant du modèle")):
    m = registry.get(model_id)
    return [registry.specs.get(c, {"name": c, "type": "unknown"}) for c in m["columns"]]


@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"],
          summary="Probabilité de décrochage d'un étudiant")
def predict(req: PredictRequest):
    m = registry.get(req.model_id)
    X, filled, ignored = to_frame(m, [req.features])
    proba = float(m["model"].predict_proba(X)[0, 1])
    t = req.threshold if req.threshold is not None else m["threshold"]
    return PredictResponse(model_id=m["id"], model_name=m["name"], scenario=m["scenario"],
                           probability=round(proba, 4), threshold=t, prediction=int(proba >= t),
                           decision=decision_label(proba, t), filled_with_defaults=filled,
                           ignored_features=ignored)


@app.post("/predict/batch", tags=["Prédiction"], summary="Probabilités pour plusieurs étudiants")
def predict_batch(req: BatchRequest):
    m = registry.get(req.model_id)
    if not req.rows:
        raise HTTPException(422, "La liste « rows » est vide.")
    X, _, _ = to_frame(m, req.rows)
    proba = m["model"].predict_proba(X)[:, 1]
    t = req.threshold if req.threshold is not None else m["threshold"]
    return {"model_id": m["id"], "threshold": t,
            "probabilities": [round(float(p), 4) for p in proba],
            "predictions": [int(p >= t) for p in proba]}
