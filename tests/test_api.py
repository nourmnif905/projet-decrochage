"""Tests de l'API :  pytest tests/   (les modèles doivent exister dans models/)"""
from fastapi.testclient import TestClient
from api.main import app, registry

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_models_and_predict():
    models = client.get("/models").json()
    assert len(models) > 0
    mid = models[0]["id"]
    r = client.post("/predict", json={"model_id": mid, "features": {"Age at enrollment": 30}})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["prediction"] in (0, 1)


def test_threshold_changes_decision():
    mid = client.get("/models").json()[0]["id"]
    low = client.post("/predict", json={"model_id": mid, "features": {}, "threshold": 0.0}).json()
    high = client.post("/predict", json={"model_id": mid, "features": {}, "threshold": 1.0}).json()
    assert low["prediction"] == 1 and high["prediction"] == 0


def test_unknown_model_404():
    assert client.post("/predict", json={"model_id": "nope", "features": {}}).status_code == 404


def test_bad_value_422():
    mid = client.get("/models").json()[0]["id"]
    r = client.post("/predict", json={"model_id": mid, "features": {"Age at enrollment": "abc"}})
    assert r.status_code == 422


# ----------------------------------------------------------------------------- explicabilité
import math
import pytest

pytest.importorskip("shap")
pytest.importorskip("lime")


def _model_id():
    ids = [m["id"] for m in client.get("/models").json()]
    return next((i for i in ids if i.startswith("xgboost")), ids[0])


def test_native_and_permutation():
    mid = _model_id()
    assert client.get("/explain/native", params={"model_id": mid}).status_code == 200
    r = client.get("/explain/permutation", params={"model_id": mid, "n_repeats": 2})
    assert r.status_code == 200 and len(r.json()["importances"]) > 5


def test_shap_global():
    r = client.get("/explain/shap/global", params={"model_id": _model_id(), "n_samples": 60, "top": 5})
    assert r.status_code == 200
    v = r.json()["variables"][0]
    assert len(v["values"]) == len(v["shap"]) == 60


def test_shap_local_is_additive():
    """base + somme des contributions doit redonner la probabilité prédite."""
    body = {"model_id": _model_id(), "features": {"Debtor": 1, "Age at enrollment": 30}}
    j = client.post("/explain/shap", json=body).json()
    total = j["base_value"] + sum(c["shap"] for c in j["contributions"])
    p = 1 / (1 + math.exp(-total)) if j["unit"] == "log-odds" else total
    assert abs(p - j["probability"]) < 1e-3
    assert j["summary"].startswith("Le risque de décrochage")


def test_lime_and_compare():
    body = {"model_id": _model_id(), "features": {"Debtor": 1}}
    assert client.post("/explain/lime", json=body).status_code == 200
    r = client.post("/explain/compare", json=body).json()
    assert {"shap", "lime", "agreement", "summary"} <= set(r)


def test_knn_shap_refused():
    ids = [m["id"] for m in client.get("/models").json()]
    if "knn" in ids:
        assert client.post("/explain/shap", json={"model_id": "knn", "features": {}}).status_code == 400
