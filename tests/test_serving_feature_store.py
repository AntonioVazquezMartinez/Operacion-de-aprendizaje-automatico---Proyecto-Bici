"""Feature store del servicio (versión 1.1.0): rezagos opcionales en /predict y registro de predicciones.

Con los datos sintéticos de tests/conftest_data.py se entrena un modelo pequeño y se exporta con el mismo código
de `python -m src.modeling.export` para el feature store (`export_feature_store`). La prueba principal,
adaptada de `test_served_predictions_match_offline_pipeline` de la rama entrega-final (Jesús Rincón), comprueba
que no hay training-serving skew: /predict SIN rezagos devuelve exactamente los rezagos y la predicción del
pipeline fuera de línea (la matriz de características del entrenamiento).
"""

from __future__ import annotations

import copy
import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features import MiBiciFeatureEngineer, model_input_columns, temporal_split
from src.modeling.export import export_feature_store, normalize_text_files, serving_paths
from src.serving.app import create_app, sha256_file
from src.serving.feature_store import (
    FeatureStore,
    build_feature_store,
    compute_lags,
    feature_store_bytes,
    lag_offsets_hours,
    predictable_window,
    read_feature_store,
    write_feature_store,
)
from src.serving.schemas import LAG_FIELDS
from tests.conftest_data import make_raw_fixture, make_test_config

STORE_HOURS = 240  # los datos sintéticos tienen 336 h con historial completo


@pytest.fixture(autouse=True)
def _sin_variables_de_entorno(monkeypatch):
    for name in ("MODEL_DIR", "MODEL_METADATA", "PREDICTION_LOG"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def fs(tmp_path_factory):
    """Modelo pequeño exportado con feature store y su referencia (formato de src.modeling.export)."""
    import mlflow.sklearn
    from mlflow.models import infer_signature

    from src.dataset import MiBiciDataCleaner
    from src.modeling.pipeline import build_pipeline

    tmp = tmp_path_factory.mktemp("serving_fs")
    config = make_test_config(tmp)
    make_raw_fixture(config.path("raw_dir"))
    MiBiciDataCleaner(config).clean()
    raw_stations = pd.read_csv(config.file("raw_dir", "stations_file"))
    df = MiBiciFeatureEngineer(config).build_features(
        pd.read_csv(config.file("interim_dir", "interim_file"), parse_dates=["Inicio_del_viaje"]), raw_stations)
    cols = model_input_columns(config)
    train = df[temporal_split(df, config) == "train"]
    pipeline = build_pipeline(config, "hist_gradient_boosting").fit(train[cols], train[config["features"]["target"]])

    model_dir = tmp / "models" / "champion"
    signature = infer_signature(train[cols].head(50), pipeline.predict(train[cols].head(50)))
    mlflow.sklearn.save_model(pipeline, str(model_dir), signature=signature, serialization_format="cloudpickle")
    raw_stations[["id", "latitude", "longitude", "capacity"]].to_csv(model_dir / "stations.csv", index=False)
    config["serving"].update({
        "model_dir": str(model_dir), "metadata_file": str(tmp / "models" / "champion_metadata.json"),
        "feature_store_hours": STORE_HOURS,
        "feature_store_reference_file": str(tmp / "models" / "reference_feature_store_predictions.csv"),
        "feature_store_reference_rows": 60, "max_batch_size": 1000,
    })
    paths = serving_paths(config)
    fs_metadata = export_feature_store(config, pipeline, df, raw_stations, paths)
    metadata = {
        "registered_model_name": "mibici-demand-forecaster", "version": 7, "alias": "champion",
        "model_uri": "models:/mibici-demand-forecaster/7", "algorithm": "hist_gradient_boosting",
        "model_file": "model.pkl", "model_sha256": sha256_file(model_dir / "model.pkl"),
        "stations_sha256": sha256_file(model_dir / "stations.csv"), "input_columns": cols,
        "target": "demand_count", **fs_metadata,
    }
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    window = fs_metadata["feature_store"]["predictable_window"]
    return {"config": config, "df": df, "cols": cols, "pipeline": pipeline, "paths": paths, "metadata": metadata,
            "raw_stations": raw_stations, "window": window, "tmp": tmp,
            "reference": pd.read_csv(paths["feature_store_reference"])}


@pytest.fixture(scope="module")
def client(fs):
    with TestClient(create_app(fs["config"])) as c:
        yield c


def _sin_rezagos(station_id, hour) -> dict:
    return {"station_id": int(station_id), "datetime": pd.Timestamp(hour).isoformat()}


def _con_rezagos(row) -> dict:
    return {**_sin_rezagos(row["Origen_Id"], row["fecha_hora"]), **{lag: float(row[lag]) for lag in LAG_FIELDS}}


def _config_con(fs, **serving):
    config = copy.deepcopy(fs["config"])
    config["serving"].update(serving)
    return config


# --------------------------------------------------------------------------- funciones del feature store
def test_ventana_predecible():
    config = make_test_config(Path("."))
    assert sorted(set(lag_offsets_hours(config))) == [24, 48, 72, 96, 120, 144, 168]
    start, end = pd.Timestamp("2025-12-17 00:00"), pd.Timestamp("2025-12-31 23:00")
    expected = (pd.Timestamp("2025-12-24 00:00"), pd.Timestamp("2026-01-01 23:00"))
    assert predictable_window(start, end, config) == expected


def test_build_feature_store_ultimas_horas(fs):
    store = build_feature_store(fs["df"], fs["config"], STORE_HOURS)
    end = fs["df"]["fecha_hora"].max()
    assert store["fecha_hora"].min() == end - pd.Timedelta(hours=STORE_HOURS - 1)
    assert store["fecha_hora"].max() == end
    assert len(store) == STORE_HOURS * fs["df"]["Origen_Id"].nunique()
    assert list(store.columns) == ["Origen_Id", "fecha_hora", "demand_count"]
    with pytest.raises(ValueError, match="rezago más largo"):
        build_feature_store(fs["df"], fs["config"], 168)


def test_feature_store_determinista(fs, tmp_path):
    """Mismos datos -> mismos bytes: cabecera gzip sin nombre ni fecha (mtime=0) y CSV con LF."""
    store = build_feature_store(fs["df"], fs["config"], STORE_HOURS)
    a, b = feature_store_bytes(store), feature_store_bytes(store.copy())
    assert a == b
    assert a[:2] == b"\x1f\x8b" and a[4:8] == b"\x00\x00\x00\x00"      # mtime=0
    assert a[3] & 0x08 == 0                                             # sin nombre de archivo (FNAME)
    assert b"\r\n" not in gzip.decompress(a)
    path = write_feature_store(store, tmp_path / "otra" / "feature_store.csv.gz")
    assert sha256_file(path) == sha256_file(fs["paths"]["feature_store"]) == fs["metadata"]["feature_store"]["sha256"]
    leido = read_feature_store(path, fs["config"])
    pd.testing.assert_frame_equal(leido, store)


def test_compute_lags_igual_al_pipeline(fs):
    """Los rezagos del feature store son los de MiBiciFeatureEngineer.add_lag_features sobre la serie completa."""
    store = FeatureStore.load(fs["paths"]["feature_store"], fs["config"])
    df = fs["df"]
    rows = df[df["fecha_hora"].between(store.window_start, store.end)]
    got = store.lookup(rows["Origen_Id"], rows["fecha_hora"])
    np.testing.assert_array_equal(got, rows[list(LAG_FIELDS)].to_numpy(dtype=float))
    before = store.lookup([1], [store.window_start - pd.Timedelta(hours=1)])
    after = store.lookup([1], [store.window_end + pd.Timedelta(hours=1)])
    assert np.isnan(before).all() and np.isnan(after).all()
    futuras = compute_lags(df[["Origen_Id", "fecha_hora", "demand_count"]], fs["config"], store.window_end)
    futuras = futuras[futuras["fecha_hora"] > store.end]
    assert len(futuras) == 24 * df["Origen_Id"].nunique()
    np.testing.assert_array_equal(store.lookup(futuras["Origen_Id"], futuras["fecha_hora"]),
                                  futuras[list(LAG_FIELDS)].to_numpy(dtype=float))


def test_feature_store_incompleto_se_rechaza(fs):
    store = read_feature_store(fs["paths"]["feature_store"], fs["config"])
    with pytest.raises(ValueError, match="cuadrícula completa"):
        FeatureStore(store.drop(index=5), fs["config"])


def test_normalize_text_files(tmp_path):
    (tmp_path / "MLmodel").write_bytes(b"a: 1\r\nb: 2\r\n")
    (tmp_path / "model.pkl").write_bytes(b"\x80\x05\r\n")
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\r\n")
    assert normalize_text_files(tmp_path, tmp_path / "model.pkl") == ["MLmodel"]
    assert (tmp_path / "MLmodel").read_bytes() == b"a: 1\nb: 2\n"
    assert (tmp_path / "model.pkl").read_bytes() == b"\x80\x05\r\n"
    assert (tmp_path / "bin.dat").read_bytes() == b"\xff\xfe\r\n"


def test_referencia_del_feature_store(fs):
    ref, md = fs["reference"], fs["metadata"]["feature_store_reference_predictions"]
    assert list(ref.columns) == ["Origen_Id", "fecha_hora", *LAG_FIELDS, "demand_count", "in_data_period",
                                 "demand_pred"]
    assert len(ref) == md["rows"] == 60 and md["sha256"] == sha256_file(fs["paths"]["feature_store_reference"])
    assert md["rows_in_data_period"] + md["rows_after_data_period"] == 60 and md["rows_after_data_period"] > 0
    horas = pd.to_datetime(ref["fecha_hora"])
    assert horas.min() >= pd.Timestamp(fs["window"]["start"]) and horas.max() <= pd.Timestamp(fs["window"]["end"])
    assert ref.loc[ref["in_data_period"] == 0, "demand_count"].isna().all()
    assert b"\r\n" not in fs["paths"]["feature_store_reference"].read_bytes()


# --------------------------------------------------------------------------- API con feature store
@pytest.mark.integration
def test_predict_sin_rezagos_coincide_con_el_pipeline(fs, client):
    """Sin training-serving skew: rezagos y predicción de /predict sin rezagos = pipeline fuera de línea."""
    df, cols = fs["df"], fs["cols"]
    rows = df[df["fecha_hora"] >= pd.Timestamp(fs["window"]["start"])].reset_index(drop=True)
    assert len(rows) > 0
    r = client.post("/predict", json={"instances": [_sin_rezagos(s, t) for s, t in zip(rows["Origen_Id"],
                                                                                      rows["fecha_hora"])]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert {p["lags_source"] for p in preds} == {"feature_store"}
    for lag in LAG_FIELDS:
        np.testing.assert_array_equal([p[lag] for p in preds], rows[lag].to_numpy(dtype=float))
    offline = np.clip(fs["pipeline"].predict(rows[cols]), 0, None)
    np.testing.assert_array_equal([p["demand_pred"] for p in preds], offline)


@pytest.mark.integration
def test_predict_reproduce_la_referencia_del_feature_store(fs, client):
    ref = fs["reference"]
    r = client.post("/predict", json={"instances": [_sin_rezagos(s, t) for s, t in zip(ref["Origen_Id"],
                                                                                      ref["fecha_hora"])]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    np.testing.assert_array_equal([p["demand_pred"] for p in preds], ref["demand_pred"].to_numpy(dtype=float))
    for lag in LAG_FIELDS:
        np.testing.assert_array_equal([p[lag] for p in preds], ref[lag].to_numpy(dtype=float))


@pytest.mark.integration
def test_predict_con_rezagos_igual_que_version_1_0_0(fs, client):
    """Con los tres rezagos se usan los del cliente (lags_source=request) aunque haya feature store."""
    df, cols = fs["df"], fs["cols"]
    rows = df[temporal_split(df, fs["config"]) == "test"].head(30).reset_index(drop=True)
    rows[list(LAG_FIELDS)] = rows[list(LAG_FIELDS)] + 1.0   # valores distintos a los del feature store
    r = client.post("/predict", json={"instances": [_con_rezagos(row) for _, row in rows.iterrows()]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert {p["lags_source"] for p in preds} == {"request"}
    np.testing.assert_array_equal([p["demand_pred"] for p in preds],
                                  np.clip(fs["pipeline"].predict(rows[cols]), 0, None))
    np.testing.assert_array_equal([p["demand_lag_24h"] for p in preds], rows["demand_lag_24h"].to_numpy())


@pytest.mark.integration
def test_lote_mixto_conserva_orden_y_origen(fs, client):
    df = fs["df"]
    row = df[df["fecha_hora"] == pd.Timestamp(fs["window"]["start"])].iloc[0]
    futura = pd.Timestamp(fs["window"]["end"])
    r = client.post("/predict", json={"instances": [_sin_rezagos(row["Origen_Id"], futura), _con_rezagos(row),
                                                    _sin_rezagos(row["Origen_Id"], row["fecha_hora"])]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert [p["lags_source"] for p in preds] == ["feature_store", "request", "feature_store"]
    assert preds[0]["datetime"] == futura.isoformat()
    assert preds[1]["demand_pred"] == preds[2]["demand_pred"]     # mismos rezagos -> misma predicción


@pytest.mark.integration
@pytest.mark.parametrize("enviados", [("demand_lag_24h",), ("demand_lag_24h", "demand_same_hour_mean_7d"),
                                      ("demand_lag_168h",)])
def test_rezagos_incompletos_422(fs, client, enviados):
    inst = {**_sin_rezagos(1, fs["window"]["start"]), **{lag: 1.0 for lag in enviados}}
    r = client.post("/predict", json={"instances": [inst]})
    assert r.status_code == 422
    detail = r.json()["details"][0]
    assert detail["type"] == "lags_incomplete"
    assert all(lag in detail["msg"] for lag in LAG_FIELDS if lag not in enviados)


@pytest.mark.integration
@pytest.mark.parametrize("desfase_horas", [-1, 1, -24 * 30])
def test_fuera_de_la_ventana_422(fs, client, desfase_horas):
    start, end = pd.Timestamp(fs["window"]["start"]), pd.Timestamp(fs["window"]["end"])
    hora = (start if desfase_horas < 0 else end) + pd.Timedelta(hours=desfase_horas)
    r = client.post("/predict", json={"instances": [_sin_rezagos(2, start), _sin_rezagos(2, hora)]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "outside_feature_store_window"
    assert body["predictable_window"] == fs["window"]
    assert start.isoformat() in body["message"] and end.isoformat() in body["message"]
    assert [d["loc"] for d in body["details"]] == [["body", "instances", 1, "datetime"]]
    # La misma hora con los tres rezagos sí se predice.
    con = {**_sin_rezagos(2, hora), **{lag: 1.0 for lag in LAG_FIELDS}}
    assert client.post("/predict", json={"instances": [con]}).status_code == 200


@pytest.mark.integration
def test_zona_horaria_tambien_sin_rezagos(fs, client):
    local = pd.Timestamp(fs["window"]["start"]) + pd.Timedelta(hours=10)
    utc = (local + pd.Timedelta(hours=6, minutes=20)).isoformat() + "Z"
    a = client.post("/predict", json={"instances": [_sin_rezagos(3, local)]}).json()["predictions"][0]
    b = client.post("/predict", json={"instances": [{"station_id": 3, "datetime": utc}]}).json()["predictions"][0]
    assert a == b


@pytest.mark.integration
def test_model_info_informa_el_feature_store(fs, client):
    info = client.get("/model-info").json()["feature_store"]
    md = fs["metadata"]["feature_store"]
    assert info["enabled"] is True and info["sha256_verified"] is True
    assert info["sha256"] == md["sha256"] and info["predictable_window"] == md["predictable_window"]
    assert info["start"] == md["start"] and info["end"] == md["end"] and info["hours"] == STORE_HOURS
    assert info["num_stations"] == 4


@pytest.mark.integration
def test_feature_store_alterado_503(fs, tmp_path):
    metadata = copy.deepcopy(fs["metadata"])
    metadata["feature_store"]["sha256"] = "0" * 64
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with TestClient(create_app(_config_con(fs, metadata_file=str(path)))) as c:
        h = c.get("/health")
        assert h.status_code == 503 and "feature store" in h.json()["detail"]
    with TestClient(create_app(_config_con(fs, metadata_file=str(path), require_sha256_match=False))) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/model-info").json()["feature_store"]["sha256_verified"] is False
    metadata["feature_store"]["file"] = "no_existe.csv.gz"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with TestClient(create_app(_config_con(fs, metadata_file=str(path)))) as c:
        assert c.get("/health").status_code == 503 and "no_existe" in c.get("/health").json()["detail"]


# --------------------------------------------------------------------------- registro de predicciones
@pytest.mark.integration
def test_prediction_log_jsonl(fs, client, tmp_path, monkeypatch):
    log = tmp_path / "predicciones.jsonl"
    monkeypatch.setenv("PREDICTION_LOG", str(log))
    start = pd.Timestamp(fs["window"]["start"])
    ok = client.post("/predict", json={"instances": [_sin_rezagos(1, start), _sin_rezagos(2, start)]})
    bad = client.post("/predict", json={"instances": [_sin_rezagos(1, start - pd.Timedelta(hours=1))]})
    assert ok.status_code == 200 and bad.status_code == 422
    assert client.get("/model-info").json()["prediction_log_enabled"] is True
    lineas = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lineas) == 2
    primera, segunda = lineas
    assert {"timestamp", "service_version", "model_uri", "model_version", "status_code", "n_instances",
            "latency_ms", "inputs", "outputs"} <= set(primera)
    assert primera["status_code"] == 200 and primera["n_instances"] == 2
    assert primera["model_uri"] == "models:/mibici-demand-forecaster/7" and primera["model_version"] == 7
    assert primera["inputs"][0] == {"station_id": 1, "datetime": start.isoformat(),
                                    **{lag: None for lag in LAG_FIELDS}}
    assert primera["outputs"] == ok.json()["predictions"]
    assert primera["latency_ms"] >= 0
    assert segunda["status_code"] == 422 and segunda["error"]["error"] == "outside_feature_store_window"


@pytest.mark.integration
def test_prediction_log_con_fallo_no_tumba_la_peticion(fs, client, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path))     # una carpeta: open(..., "a") falla
    with caplog.at_level(logging.WARNING, logger="mibici.serving"):
        r = client.post("/predict", json={"instances": [_sin_rezagos(1, fs["window"]["start"])]})
    assert r.status_code == 200
    assert "No se pudo escribir el registro de predicciones" in caplog.text


@pytest.mark.integration
def test_prediction_log_desde_config(fs, tmp_path):
    log = tmp_path / "desde_config.jsonl"
    with TestClient(create_app(_config_con(fs, prediction_log_file=str(log)))) as c:
        assert c.post("/predict", json={"instances": [_sin_rezagos(4, fs["window"]["end"])]}).status_code == 200
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_prediction_log_apagado_por_omision(fs, client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert client.post("/predict", json={"instances": [_sin_rezagos(1, fs["window"]["start"])]}).status_code == 200
    assert list(tmp_path.iterdir()) == []
