"""Pruebas del servicio FastAPI (src/serving).

Se entrena un modelo pequeño con los datos sintéticos de tests/conftest_data.py y se exporta a una carpeta
temporal con el mismo formato que `python -m src.modeling.export` (modelo MLflow + stations.csv + metadatos),
así que las pruebas no dependen de models/champion. La última prueba usa el modelo real si existe.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import PROJ_ROOT, load_config
from src.features import MiBiciFeatureEngineer, model_input_columns, temporal_split
from src.serving.app import (
    build_model_inputs,
    check_schema_compatibility,
    create_app,
    parse_utc_offset,
    sha256_file,
    to_local_hour,
)
from src.serving.schemas import LAG_FIELDS, PredictionInstance
from tests.conftest_data import make_raw_fixture, make_test_config

GDL = timezone(timedelta(hours=-6))


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _sin_variables_de_entorno(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("MODEL_METADATA", raising=False)


@pytest.fixture(scope="module")
def artefactos(tmp_path_factory):
    """Datos sintéticos, pipeline entrenado y carpeta exportada con el formato de src.modeling.export."""
    import mlflow.sklearn
    from mlflow.models import infer_signature

    from src.dataset import MiBiciDataCleaner
    from src.modeling.pipeline import build_pipeline

    tmp = tmp_path_factory.mktemp("serving")
    config = make_test_config(tmp)
    make_raw_fixture(config.path("raw_dir"))
    MiBiciDataCleaner(config).clean()
    df = MiBiciFeatureEngineer(config).build_features(
        pd.read_csv(config.file("interim_dir", "interim_file"), parse_dates=["Inicio_del_viaje"]),
        pd.read_csv(config.file("raw_dir", "stations_file")),
    )
    cols = model_input_columns(config)
    split = temporal_split(df, config)
    train = df[split == "train"]
    pipeline = build_pipeline(config, "hist_gradient_boosting").fit(train[cols], train[config["features"]["target"]])

    model_dir = tmp / "models" / "champion"
    signature = infer_signature(train[cols].head(50), pipeline.predict(train[cols].head(50)))
    mlflow.sklearn.save_model(pipeline, str(model_dir), signature=signature, serialization_format="cloudpickle")
    stations = pd.read_csv(config.file("raw_dir", "stations_file"))[["id", "latitude", "longitude", "capacity"]]
    stations.to_csv(model_dir / "stations.csv", index=False)
    metadata = {
        "registered_model_name": "mibici-demand-forecaster", "version": 7, "alias": "champion",
        "model_uri": "models:/mibici-demand-forecaster/7", "algorithm": "hist_gradient_boosting",
        "model_file": "model.pkl", "model_sha256": sha256_file(model_dir / "model.pkl"),
        "stations_sha256": sha256_file(model_dir / "stations.csv"), "num_stations": len(stations),
        "metrics": {"test_mae": 1.0}, "input_columns": cols, "target": "demand_count",
        "library_versions": {"python": "x"},
    }
    metadata_path = tmp / "models" / "champion_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    config["serving"].update({"model_dir": str(model_dir), "metadata_file": str(metadata_path),
                              "max_batch_size": 50})
    return {"config": config, "df": df, "test": df[split == "test"].reset_index(drop=True), "cols": cols,
            "pipeline": pipeline, "model_dir": model_dir, "metadata_path": metadata_path,
            "metadata": metadata, "tmp": tmp}


@pytest.fixture(scope="module")
def client(artefactos):
    with TestClient(create_app(artefactos["config"])) as c:
        yield c


def _instancia(row, **cambios):
    data = {"station_id": int(row["Origen_Id"]), "datetime": pd.Timestamp(row["fecha_hora"]).isoformat(),
            **{lag: float(row[lag]) for lag in LAG_FIELDS}}
    data.update(cambios)
    return data


def _config_con(artefactos, **serving):
    config = copy.deepcopy(artefactos["config"])
    config["serving"].update(serving)
    return config


# --------------------------------------------------------------------------- unitarias
def test_parse_utc_offset():
    assert parse_utc_offset("-06:00") == GDL
    assert parse_utc_offset("+05:30") == timezone(timedelta(hours=5, minutes=30))


def test_to_local_hour_trunca_y_convierte():
    assert to_local_hour(datetime(2025, 11, 8, 10, 45, 12), GDL) == datetime(2025, 11, 8, 10)
    assert to_local_hour(datetime(2025, 11, 8, 16, 30, tzinfo=timezone.utc), GDL) == datetime(2025, 11, 8, 10)
    assert to_local_hour(datetime(2025, 11, 8, 10, 0, tzinfo=GDL), GDL) == datetime(2025, 11, 8, 10)


def test_build_model_inputs_deriva_calendario_y_estacion(artefactos, client):
    state = client.app.state.model_state
    base = {lag: 1.0 for lag in LAG_FIELDS}
    instancias = [
        PredictionInstance(station_id=2, datetime="2024-05-04T10:20:00", **base),  # sábado
        PredictionInstance(station_id=1, datetime="2024-05-01T06:00:00", **base),  # 1 de mayo (festivo), miércoles
        PredictionInstance(station_id=3, datetime="2024-05-06T18:59:59", **base),  # lunes laborable
    ]
    X, horas = build_model_inputs(instancias, state, GDL, "Origen_Id")
    assert list(X.columns) == artefactos["cols"]
    assert horas == [datetime(2024, 5, 4, 10), datetime(2024, 5, 1, 6), datetime(2024, 5, 6, 18)]
    assert X["Origen_Id"].tolist() == [2, 1, 3]  # se conserva el orden de la petición
    assert X["is_weekend"].tolist() == [1, 0, 0]
    assert X["is_holiday"].tolist() == [0, 1, 0]
    assert X.loc[0, "sin_hour"] == np.sin(2 * np.pi * 10 / 24.0)
    assert X.loc[2, "cos_dayofweek"] == np.cos(2 * np.pi * 0 / 7.0)
    stations = pd.read_csv(artefactos["model_dir"] / "stations.csv").set_index("id")
    assert X.loc[0, "capacity"] == stations.loc[2, "capacity"]
    assert X.loc[1, "latitude"] == stations.loc[1, "latitude"]
    assert X["capacity"].dtype == np.int64 and X["latitude"].dtype == np.float64


def test_check_schema_compatibility_detecta_rezagos_distintos(artefactos):
    config = copy.deepcopy(artefactos["config"])
    check_schema_compatibility(config, artefactos["cols"], "Origen_Id")
    config["features"]["lag_hours"] = [24]
    with pytest.raises(ValueError, match="rezagos"):
        check_schema_compatibility(config, artefactos["cols"], "Origen_Id")
    otras = [c for c in artefactos["cols"] if c != "demand_lag_168h"] + ["demand_lag_1h"]
    with pytest.raises(ValueError, match="no deriva el servicio"):
        check_schema_compatibility(artefactos["config"], otras, "Origen_Id")


# --------------------------------------------------------------------------- API: operación
@pytest.mark.integration
def test_health_y_raiz(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_sha256_verified"] is True
    assert body["model_uri"] == "models:/mibici-demand-forecaster/7"
    assert client.get("/").json()["docs"] == "/docs"


@pytest.mark.integration
def test_model_info(client, artefactos):
    info = client.get("/model-info").json()
    assert info["model"] == {"name": "mibici-demand-forecaster", "version": 7, "alias": "champion",
                             "uri": "models:/mibici-demand-forecaster/7", "algorithm": "hist_gradient_boosting"}
    assert info["model_sha256"] == info["model_sha256_computed"] == artefactos["metadata"]["model_sha256"]
    assert info["stations_sha256_verified"] is True
    assert info["input_columns"] == artefactos["cols"]
    assert info["num_stations"] == 4 and info["max_batch_size"] == 50
    assert info["metrics"] == {"test_mae": 1.0}
    assert "scikit-learn" in info["runtime_library_versions"]


@pytest.mark.integration
def test_openapi_documenta_predict(client):
    spec = client.get("/openapi.json").json()
    assert {"/predict", "/predict/raw", "/health", "/model-info"} <= set(spec["paths"])
    instance = spec["components"]["schemas"]["PredictionInstance"]
    assert instance["example"]["station_id"] == 24
    assert all(instance["properties"][f].get("description") for f in instance["properties"])
    assert "404" in spec["paths"]["/predict"]["post"]["responses"]


@pytest.mark.integration
def test_ruta_inexistente_404_json(client):
    r = client.get("/no-existe")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


# --------------------------------------------------------------------------- API: predicción
@pytest.mark.integration
def test_predict_una_instancia(client, artefactos):
    row = artefactos["test"].iloc[0]
    r = client.post("/predict", json={"instances": [_instancia(row)]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_instances"] == 1
    assert body["model"]["uri"] == "models:/mibici-demand-forecaster/7"
    pred = body["predictions"][0]
    assert pred["station_id"] == int(row["Origen_Id"])
    assert pred["datetime"] == pd.Timestamp(row["fecha_hora"]).isoformat()
    assert pred["demand_pred"] >= 0 and math.isfinite(pred["demand_pred"])


@pytest.mark.integration
def test_predict_lote_coherente_con_el_pipeline(client, artefactos):
    """Coherencia entrenamiento-servicio: mismas variables de src.features -> misma predicción, bit a bit."""
    test, cols = artefactos["test"].head(40), artefactos["cols"]
    esperado = np.clip(artefactos["pipeline"].predict(test[cols]), 0, None)
    r = client.post("/predict", json={"instances": [_instancia(row) for _, row in test.iterrows()]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert [p["station_id"] for p in preds] == test["Origen_Id"].tolist()
    np.testing.assert_array_equal([p["demand_pred"] for p in preds], esperado)

    raw = client.post("/predict/raw", json={"rows": test[cols].to_dict(orient="records")})
    assert raw.status_code == 200, raw.text
    np.testing.assert_array_equal(raw.json()["predictions"], esperado)


@pytest.mark.integration
def test_predict_no_negativo(artefactos):
    class ModeloNegativo:
        def predict(self, X):
            return np.full(len(X), -3.0)

    with TestClient(create_app(artefactos["config"])) as c:
        c.app.state.model_state.model = ModeloNegativo()
        row = artefactos["test"].iloc[0]
        assert c.post("/predict", json={"instances": [_instancia(row)]}).json()["predictions"][0]["demand_pred"] == 0
        raw = c.post("/predict/raw", json={"rows": artefactos["test"][artefactos["cols"]].head(2).to_dict("records")})
        assert raw.json()["predictions"] == [0.0, 0.0]


@pytest.mark.integration
def test_predict_zona_horaria(client, artefactos):
    row = artefactos["test"].iloc[0]
    local = pd.Timestamp(row["fecha_hora"]).to_pydatetime()
    utc = (local + timedelta(hours=6, minutes=15)).isoformat() + "Z"
    a = client.post("/predict", json={"instances": [_instancia(row)]}).json()["predictions"][0]
    b = client.post("/predict", json={"instances": [_instancia(row, datetime=utc)]}).json()["predictions"][0]
    assert b["datetime"] == a["datetime"] and b["demand_pred"] == a["demand_pred"]


@pytest.mark.integration
def test_predict_estacion_desconocida_404(client, artefactos):
    row = artefactos["test"].iloc[0]
    r = client.post("/predict", json={"instances": [_instancia(row), _instancia(row, station_id=999),
                                                    _instancia(row, station_id=555)]})
    assert r.status_code == 404
    assert r.json()["error"] == "unknown_station" and r.json()["unknown_station_ids"] == [555, 999]


BASE = {"station_id": 1, "datetime": "2024-05-10T08:00:00", "demand_lag_24h": 1.0, "demand_lag_168h": 0.0,
        "demand_same_hour_mean_7d": 0.5}


@pytest.mark.integration
@pytest.mark.parametrize("cambio, tipo", [
    ({"demand_lag_24h": -1}, "greater_than_equal"),
    ({"demand_lag_168h": "abc"}, "float_type"),
    ({"demand_same_hour_mean_7d": True}, "float_type"),
    ({"demand_lag_24h": None}, "float_type"),
    ({"station_id": "1"}, "int_type"),
    ({"station_id": 1.5}, "int_type"),
    ({"station_id": 0}, "greater_than_equal"),
    ({"datetime": "2024-13-40T25:00:00"}, "datetime_from_date_parsing"),
    ({"datetime": "ayer"}, "datetime_from_date_parsing"),
    ({"datetime": 1715328000}, "value_error"),
    ({"extra": 1}, "extra_forbidden"),
])
def test_predict_422_por_campo(client, cambio, tipo):
    r = client.post("/predict", json={"instances": [{**BASE, **cambio}]})
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "validation_error"
    assert tipo in [d["type"] for d in body["details"]]


@pytest.mark.integration
def test_predict_422_campo_faltante(client):
    sin_lag = {k: v for k, v in BASE.items() if k != "demand_lag_168h"}
    r = client.post("/predict", json={"instances": [sin_lag]})
    assert r.status_code == 422 and r.json()["details"][0]["type"] == "missing"


@pytest.mark.integration
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_predict_422_no_finitos(client, literal):
    cuerpo = json.dumps({"instances": [BASE]}).replace('"demand_lag_24h": 1.0', f'"demand_lag_24h": {literal}')
    r = client.post("/predict", content=cuerpo, headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["details"][0]["type"] == "finite_number"


@pytest.mark.integration
def test_predict_422_lote_vacio_y_demasiado_grande(client):
    vacio = client.post("/predict", json={"instances": []})
    assert vacio.status_code == 422 and vacio.json()["details"][0]["type"] == "too_short"
    grande = client.post("/predict", json={"instances": [BASE] * 51})
    assert grande.status_code == 422 and grande.json()["details"][0]["type"] == "too_long"


@pytest.mark.integration
def test_predict_422_json_invalido(client):
    r = client.post("/predict", content="{no es json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


@pytest.mark.integration
def test_predict_raw_422(client, artefactos):
    fila = artefactos["test"][artefactos["cols"]].iloc[0].to_dict()
    sin_col = {k: v for k, v in fila.items() if k != "capacity"}
    r = client.post("/predict/raw", json={"rows": [sin_col]})
    assert r.status_code == 422 and r.json()["details"][0]["type"] == "columns_mismatch"
    r = client.post("/predict/raw", json={"rows": [{**fila, "Origen_Id": 1.5}]})
    assert r.status_code == 422 and r.json()["details"][0]["type"] == "int_expected"
    cuerpo = json.dumps({"rows": [{**fila, "latitude": 0.123}]}).replace("0.123", "NaN")
    r = client.post("/predict/raw", content=cuerpo, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# --------------------------------------------------------------------------- API: modelo no disponible
@pytest.mark.integration
def test_503_sin_modelo(artefactos, tmp_path):
    config = _config_con(artefactos, model_dir=str(tmp_path / "no_existe"))
    with TestClient(create_app(config)) as c:
        h = c.get("/health")
        assert h.status_code == 503 and h.json()["model_loaded"] is False and "MLmodel" in h.json()["detail"]
        r = c.post("/predict", json={"instances": [BASE]})
        assert r.status_code == 503 and r.json()["error"] == "model_unavailable"
        assert c.get("/model-info").status_code == 503
        assert c.post("/predict/raw", json={"rows": [{"a": 1.0}]}).status_code == 503


@pytest.mark.integration
def test_503_sha256_distinto(artefactos, tmp_path):
    metadata = {**artefactos["metadata"], "model_sha256": "0" * 64}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with TestClient(create_app(_config_con(artefactos, metadata_file=str(path)))) as c:
        h = c.get("/health")
        assert h.status_code == 503 and "SHA-256" in h.json()["detail"]


@pytest.mark.integration
def test_503_esquema_incompatible(artefactos):
    config = copy.deepcopy(artefactos["config"])
    config["features"]["lag_hours"] = [24, 48]
    with TestClient(create_app(config)) as c:
        h = c.get("/health")
        assert h.status_code == 503 and "rezagos" in h.json()["detail"]


@pytest.mark.integration
def test_strict_startup_detiene_el_arranque(artefactos, tmp_path):
    config = _config_con(artefactos, model_dir=str(tmp_path / "no_existe"), strict_startup=True)
    with pytest.raises(RuntimeError, match="strict_startup"):
        with TestClient(create_app(config)):
            pass


@pytest.mark.integration
def test_variables_de_entorno_sobrescriben_rutas(artefactos, tmp_path, monkeypatch):
    copia = tmp_path / "otro_modelo"
    shutil.copytree(artefactos["model_dir"], copia)
    monkeypatch.setenv("MODEL_DIR", str(copia))
    monkeypatch.setenv("MODEL_METADATA", str(artefactos["metadata_path"]))
    with TestClient(create_app(_config_con(artefactos, model_dir="no/existe", metadata_file="no/existe.json"))) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/model-info").json()["model_dir"] == str(copia)


@pytest.mark.integration
def test_500_generico_sin_traza(artefactos):
    class ModeloRoto:
        def predict(self, X):
            raise RuntimeError("detalle interno secreto")

    with TestClient(create_app(artefactos["config"]), raise_server_exceptions=False) as c:
        c.app.state.model_state.model = ModeloRoto()
        r = c.post("/predict", json={"instances": [BASE]})
        assert r.status_code == 500
        assert r.json() == {"error": "internal_error", "message": "Error interno del servidor."}
        assert "secreto" not in r.text


# --------------------------------------------------------------------------- modelo real (opcional)
@pytest.mark.integration
def test_modelo_real_reproduce_la_referencia():
    """Las 500 filas de models/reference_predictions.csv se reproducen por /predict y por /predict/raw."""
    config = load_config()
    model_dir = PROJ_ROOT / config["serving"]["model_dir"]
    reference_path = PROJ_ROOT / config["serving"]["reference_predictions_file"]
    if not (model_dir / "MLmodel").exists() or not reference_path.exists():
        pytest.skip("No está el modelo exportado en models/champion (se genera con python -m src.modeling.export)")
    reference = pd.read_csv(reference_path)
    esperado = reference["demand_pred"].to_numpy(dtype=float)
    with TestClient(create_app(config)) as c:
        assert c.get("/health").status_code == 200
        cols = c.get("/model-info").json()["input_columns"]
        instancias = [_instancia(row) for _, row in reference.iterrows()]
        r = c.post("/predict", json={"instances": instancias})
        assert r.status_code == 200, r.text
        np.testing.assert_array_equal([p["demand_pred"] for p in r.json()["predictions"]], esperado)
        raw = c.post("/predict/raw", json={"rows": reference[cols].to_dict(orient="records")})
        assert raw.status_code == 200, raw.text
        np.testing.assert_array_equal(raw.json()["predictions"], esperado)
