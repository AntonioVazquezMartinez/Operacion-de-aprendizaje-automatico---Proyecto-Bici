"""Pruebas de la API con el cliente de pruebas de FastAPI (requiere fastapi y httpx)."""

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(bundle_dir, monkeypatch_module):
    monkeypatch_module.setenv("MODEL_DIR", str(bundle_dir))
    from src.serving.app import app

    with TestClient(app) as c:  # el contexto ejecuta el lifespan (carga del modelo)
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def valid_instance(client, bundle_dir):
    window = client.get("/model-info").json()["predictable_window"]
    station = int(pd.read_csv(bundle_dir / "stations.csv")["id"].iloc[0])
    return {"station_id": station, "fecha_hora": window["hasta"]}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "1", "detail": None}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_uri"] == "models:/mibici-test/1"
    assert "predictable_window" in body


def test_predict_ok(client, valid_instance):
    r = client.post("/predict", json={"instances": [valid_instance, valid_instance]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_predictions"] == 2
    assert body["model_version"] == "1"
    assert body["predictions"][0]["demand_pred"] >= 0


def test_predict_invalid_payload_returns_422(client):
    assert client.post("/predict", json={"instances": [{"station_id": "abc"}]}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_predict_unknown_station_returns_422(client, valid_instance):
    r = client.post("/predict", json={"instances": [{**valid_instance, "station_id": 999999}]})
    assert r.status_code == 422
    assert "desconocidas" in r.json()["detail"]


def test_openapi_documents_predict(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "PredictRequest" in schema["components"]["schemas"]
