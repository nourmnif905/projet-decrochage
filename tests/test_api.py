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
