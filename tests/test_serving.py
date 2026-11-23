"""Pruebas unitarias del servicio de predicción y de los esquemas de entrada."""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from src.serving.bundle import CARD_FILE, MODEL_FILE, STORE_FILE, load_card
from src.serving.schemas import MAX_INSTANCES, PredictRequest
from src.serving.service import ModelService, PredictionError


@pytest.fixture(scope="module")
def service(bundle_dir):
    return ModelService(bundle_dir)


def first_valid_instance(service):
    return {"station_id": int(service.stations.index[0]), "fecha_hora": service.predictable_window()["hasta"]}


def test_bundle_contents(bundle_dir):
    for name in (MODEL_FILE, CARD_FILE, STORE_FILE, "stations.csv", "reference_predictions.csv"):
        assert (bundle_dir / name).exists(), name
    card = load_card(bundle_dir)
    assert card["model_uri"] == "models:/mibici-test/1"
    assert len(card["model_sha256"]) == 64
    assert "Origen_Id" in card["input_dtypes"]


def test_predict_uses_feature_store(service):
    out = service.predict([first_valid_instance(service)])
    assert len(out) == 1
    assert out[0]["demand_pred"] >= 0
    assert out[0]["lags_source"] == "feature_store"


def test_request_lags_take_priority(service):
    inst = {"station_id": int(service.stations.index[0]), "fecha_hora": "2030-01-01T08:00:00",
            "demand_lag_24h": 3.0, "demand_lag_168h": 2.0, "demand_same_hour_mean_7d": 2.5}
    out = service.predict([inst])[0]
    assert out["lags_source"] == "request"
    assert out["demand_lag_24h"] == 3.0


def test_unknown_station_raises(service):
    with pytest.raises(PredictionError, match="desconocidas"):
        service.predict([{"station_id": 999999, "fecha_hora": service.predictable_window()["hasta"]}])


def test_out_of_window_without_lags_raises(service):
    with pytest.raises(PredictionError, match="historial"):
        service.predict([{"station_id": int(service.stations.index[0]), "fecha_hora": "2030-01-01T08:00:00"}])


def test_timezone_is_converted_to_local_hour(service):
    local = pd.Timestamp(service.predictable_window()["hasta"])
    utc = (local + pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    station = int(service.stations.index[0])
    a = service.predict([{"station_id": station, "fecha_hora": local}])[0]
    b = service.predict([{"station_id": station, "fecha_hora": utc}])[0]
    assert a["fecha_hora"] == b["fecha_hora"]
    assert a["demand_pred"] == pytest.approx(b["demand_pred"])


def test_station_id_type_does_not_change_prediction(service, features):
    """Un Origen_Id entero o float debe producir la misma predicción (tipos del entrenamiento)."""
    rows = features.tail(20).copy()
    as_float = service.predict_frame(rows.assign(Origen_Id=rows["Origen_Id"].astype(float)))
    as_int = service.predict_frame(rows.assign(Origen_Id=rows["Origen_Id"].astype(int)))
    assert (as_float == as_int).all()


def test_schema_validation():
    with pytest.raises(ValidationError):
        PredictRequest(instances=[])
    with pytest.raises(ValidationError):
        PredictRequest(instances=[{"station_id": -1, "fecha_hora": "2025-12-31T08:00:00"}])
    with pytest.raises(ValidationError):
        PredictRequest(instances=[{"station_id": 2, "fecha_hora": "no es fecha"}])
    with pytest.raises(ValidationError):
        PredictRequest(instances=[{"station_id": 2, "fecha_hora": "2025-12-31T08:00:00", "extra": 1}])
    with pytest.raises(ValidationError):
        PredictRequest(instances=[{"station_id": 2, "fecha_hora": "2025-12-31T08:00:00"}] * (MAX_INSTANCES + 1))
    ok = PredictRequest(instances=[{"station_id": 2, "fecha_hora": "2025-12-31T08:00:00"}])
    assert json.loads(ok.model_dump_json())["instances"][0]["station_id"] == 2


def test_global_seed_is_set_from_config():
    import random

    import numpy as np

    from src.config import load_config

    load_config()
    a = (random.random(), np.random.rand())
    load_config()
    assert (random.random(), np.random.rand()) == a
