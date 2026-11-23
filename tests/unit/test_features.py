"""Ingeniería de características: calendario, cuadrícula de demanda, rezagos sin fuga y partición."""

from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import features as features_module
from src.features import (
    MiBiciFeatureEngineer,
    _nth_monday,
    feature_columns,
    lag_columns,
    load_features,
    mexican_holidays,
    model_input_columns,
    temporal_split,
)

NUMERIC_BASE = ["sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "latitude", "longitude", "capacity"]
LAGS = ["demand_lag_24h", "demand_lag_168h", "demand_same_hour_mean_7d"]


# --------------------------------------------------------------------------- calendario
def test_mexican_holidays_2025_exact():
    """Art. 74 LFT en 2025: los lunes móviles caen el 3 feb, 17 mar y 17 nov; 2025 no es año de transmisión."""
    assert mexican_holidays(2025) == {
        date(2025, 1, 1), date(2025, 2, 3), date(2025, 3, 17), date(2025, 5, 1),
        date(2025, 9, 16), date(2025, 11, 17), date(2025, 12, 25),
    }


def test_mexican_holidays_2024():
    holidays_2024 = mexican_holidays(2024)
    assert date(2024, 5, 1) in holidays_2024
    assert date(2024, 2, 5) in holidays_2024    # primer lunes de febrero
    assert date(2024, 3, 18) in holidays_2024   # tercer lunes de marzo
    assert date(2024, 11, 18) in holidays_2024  # tercer lunes de noviembre
    assert date(2024, 10, 1) in holidays_2024   # cambio de gobierno
    assert len(holidays_2024) == 8


@pytest.mark.parametrize("year, has_transition", [(2024, True), (2025, False), (2029, False), (2030, True)])
def test_transition_of_power_every_six_years(year, has_transition):
    assert (date(year, 10, 1) in mexican_holidays(year)) is has_transition


@pytest.mark.parametrize("year, month, n, expected", [
    (2025, 2, 1, date(2025, 2, 3)),
    (2027, 2, 1, date(2027, 2, 1)),     # el día 1 ya es lunes
    (2025, 11, 3, date(2025, 11, 17)),
])
def test_nth_monday(year, month, n, expected):
    assert _nth_monday(year, month, n) == expected
    assert expected.weekday() == 0


@pytest.fixture
def calendar_grid():
    ts = pd.to_datetime([
        "2025-11-17 00:00",  # lunes festivo (Revolución)
        "2025-11-18 06:00",  # martes
        "2025-11-22 12:00",  # sábado
        "2025-11-23 18:00",  # domingo
        "2025-12-25 23:00",  # jueves festivo
    ])
    grid = pd.DataFrame({"Origen_Id": 1, "fecha_hora": ts, "demand_count": range(5)})
    return MiBiciFeatureEngineer.add_calendar_features(grid)


def test_cyclical_encoding_known_hours(calendar_grid):
    expected_sin = [0.0, 1.0, 0.0, -1.0, np.sin(2 * np.pi * 23 / 24)]
    expected_cos = [1.0, 0.0, -1.0, 0.0, np.cos(2 * np.pi * 23 / 24)]
    assert calendar_grid["sin_hour"].tolist() == pytest.approx(expected_sin, abs=1e-12)
    assert calendar_grid["cos_hour"].tolist() == pytest.approx(expected_cos, abs=1e-12)
    # La hora 23 queda junto a la 0 en el círculo.
    assert calendar_grid.loc[4, "cos_hour"] > 0.96
    for col in ("sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"):
        assert calendar_grid[col].between(-1.0, 1.0).all()


def test_calendar_flags(calendar_grid):
    assert calendar_grid["hour"].tolist() == [0, 6, 12, 18, 23]
    assert calendar_grid["day_of_week"].tolist() == [0, 1, 5, 6, 3]
    assert calendar_grid["month"].tolist() == [11, 11, 11, 11, 12]
    assert calendar_grid["is_weekend"].tolist() == [0, 0, 1, 1, 0]
    assert calendar_grid["is_holiday"].tolist() == [1, 0, 0, 0, 1]
    # Lunes (0) -> sin 0, cos 1.
    assert calendar_grid.loc[0, "sin_dayofweek"] == pytest.approx(0.0)
    assert calendar_grid.loc[0, "cos_dayofweek"] == pytest.approx(1.0)


def test_is_holiday_flag_marks_only_holidays():
    """is_holiday se calcula sobre la fecha de cada hora, independientemente del historial descartado."""
    grid = pd.DataFrame({
        "Origen_Id": [1, 1, 1],
        "fecha_hora": pd.to_datetime(["2024-05-01 08:00", "2024-05-01 23:00", "2024-05-02 08:00"]),
        "demand_count": [3, 1, 2],
    })
    flagged = MiBiciFeatureEngineer.add_calendar_features(grid)
    assert flagged["is_holiday"].tolist() == [1, 1, 0]


# --------------------------------------------------------------------------- cuadrícula y estaciones
def test_demand_grid_is_complete_with_zeros(config):
    trips = pd.DataFrame({
        "Inicio_del_viaje": ["2025-03-10 08:10", "2025-03-10 08:50", "2025-03-10 10:05", "2025-03-10 09:30"],
        "Origen_Id": [1, 1, 1, 2],
    })
    grid = MiBiciFeatureEngineer(config).build_demand_grid(trips)

    assert len(grid) == 6                               # 2 estaciones activas x 3 horas
    assert grid["Origen_Id"].tolist() == [1, 1, 1, 2, 2, 2]
    assert grid["fecha_hora"].dt.hour.tolist() == [8, 9, 10, 8, 9, 10]
    assert grid["demand_count"].tolist() == [2, 0, 1, 0, 1, 0]


def test_demand_grid_preserves_trips(demand_grid, clean_trips):
    assert demand_grid["demand_count"].sum() == len(clean_trips)
    assert (demand_grid["demand_count"] >= 0).all()
    n_hours = demand_grid["fecha_hora"].nunique()
    assert len(demand_grid) == demand_grid["Origen_Id"].nunique() * n_hours


def test_add_station_features_merges_catalog():
    grid = pd.DataFrame({"Origen_Id": [2, 1], "fecha_hora": pd.to_datetime(["2025-01-01", "2025-01-01"])})
    stations = pd.DataFrame({"id": [1, 2], "latitude": [20.1, 20.2], "longitude": [-103.1, -103.2],
                             "capacity": [10, 20], "name": ["a", "b"]})
    out = MiBiciFeatureEngineer.add_station_features(grid, stations)
    assert list(out.columns) == ["Origen_Id", "fecha_hora", "latitude", "longitude", "capacity"]
    assert out["Origen_Id"].tolist() == [1, 2]
    assert out["capacity"].tolist() == [10, 20]


# --------------------------------------------------------------------------- rezagos
def series_grid(hours: int = 200) -> pd.DataFrame:
    """Dos estaciones con demanda conocida: A(t) = t y B(t) = 1000 + t."""
    ts = pd.date_range("2025-01-01", periods=hours, freq="h")
    a = pd.DataFrame({"Origen_Id": 1, "fecha_hora": ts, "demand_count": np.arange(hours)})
    b = pd.DataFrame({"Origen_Id": 2, "fecha_hora": ts, "demand_count": 1000 + np.arange(hours)})
    return pd.concat([a, b], ignore_index=True)


def test_lag_values_are_exact_past_demand(config):
    out = MiBiciFeatureEngineer(config).add_lag_features(series_grid())
    t = out["demand_count"] - np.where(out["Origen_Id"] == 2, 1000, 0)   # índice horario de la fila
    base = np.where(out["Origen_Id"] == 2, 1000, 0)
    assert (out["demand_lag_24h"] == base + t - 24).all()
    assert (out["demand_lag_168h"] == base + t - 168).all()
    # Media de t-24, t-48, ..., t-168 = t - 96.
    assert np.allclose(out["demand_same_hour_mean_7d"], base + t - 96)


def test_lags_do_not_mix_stations(config):
    out = MiBiciFeatureEngineer(config).add_lag_features(series_grid())
    station_b = out[out["Origen_Id"] == 2]
    assert station_b["demand_lag_168h"].min() == 1000      # nunca toma valores de la estación A
    assert (station_b[LAGS] >= 1000).all().all()


def test_lag_24h_matches_grid_lookup(session_config, demand_grid, feature_matrix):
    """En los datos sintéticos, el rezago de 24 h de la hora t es la demanda de t-24 en la misma estación."""
    station = feature_matrix["Origen_Id"].iloc[0]
    grid = demand_grid[demand_grid["Origen_Id"] == station].set_index("fecha_hora")["demand_count"]
    rows = feature_matrix[feature_matrix["Origen_Id"] == station].iloc[[0, 10, 100]]
    for _, row in rows.iterrows():
        assert row["demand_lag_24h"] == grid.loc[row["fecha_hora"] - pd.Timedelta(hours=24)]
        assert row["demand_lag_168h"] == grid.loc[row["fecha_hora"] - pd.Timedelta(hours=168)]


def test_lags_never_use_future_information(config):
    """Cambiar la demanda futura no altera ninguna característica de las horas anteriores."""
    engineer = MiBiciFeatureEngineer(config)
    grid = series_grid()
    cutoff = grid["fecha_hora"].iloc[180]
    altered = grid.copy()
    altered.loc[altered["fecha_hora"] > cutoff, "demand_count"] = 99999

    before = engineer.add_lag_features(grid)
    after = engineer.add_lag_features(altered)
    past = before["fecha_hora"] <= cutoff
    pd.testing.assert_frame_equal(before.loc[past, LAGS], after.loc[past, LAGS])
    # Y la hora siguiente al corte sí ve el pasado, no el valor alterado.
    assert (after.loc[~past, LAGS] < 99999).all().all()


def test_incomplete_history_is_dropped(config):
    out = MiBiciFeatureEngineer(config).add_lag_features(series_grid(200))
    assert len(out) == 2 * (200 - 168)
    assert out["fecha_hora"].min() == pd.Timestamp("2025-01-01") + pd.Timedelta(hours=168)
    assert not out[LAGS].isna().any().any()


def test_incomplete_history_filled_with_zero_when_configured(config):
    config["features"]["drop_incomplete_history"] = False
    out = MiBiciFeatureEngineer(config).add_lag_features(series_grid(200))
    assert len(out) == 400
    first = out[(out["Origen_Id"] == 1)].iloc[0]
    assert first[LAGS].tolist() == [0, 0, 0]


def test_without_same_hour_mean(config):
    config["features"]["same_hour_mean_days"] = None
    config["features"]["lag_hours"] = [24]
    out = MiBiciFeatureEngineer(config).add_lag_features(series_grid(48))
    assert lag_columns(config) == ["demand_lag_24h"]
    assert "demand_same_hour_mean_7d" not in out.columns
    assert len(out) == 2 * 24


def test_incomplete_history_dropped_on_synthetic_data(session_config, demand_grid, feature_matrix):
    assert not feature_matrix[feature_columns(session_config)["numeric"]].isna().any().any()
    first_hour = demand_grid["fecha_hora"].min()
    max_lag = max(session_config["features"]["lag_hours"])
    assert feature_matrix["fecha_hora"].min() >= first_hour + pd.Timedelta(hours=max_lag)


def test_no_short_lags_leak_future_horizon(session_config):
    """Ningún rezago menor a 24 h (el horizonte máximo del Canvas) entra al modelo."""
    assert "demand_lag_1h" not in model_input_columns(session_config)
    assert all(h >= 24 for h in session_config["features"]["lag_hours"])


def test_cyclical_bounds_on_synthetic_data(feature_matrix):
    for col in ["sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"]:
        assert feature_matrix[col].between(-1.0 - 1e-9, 1.0 + 1e-9).all()


# --------------------------------------------------------------------------- columnas
def test_feature_columns_and_model_inputs(session_config):
    groups = feature_columns(session_config)
    assert groups == {
        "numeric": NUMERIC_BASE + LAGS,
        "categorical": ["Origen_Id"],
        "binary": ["is_weekend", "is_holiday"],
    }
    assert model_input_columns(session_config) == NUMERIC_BASE + LAGS + ["Origen_Id", "is_weekend", "is_holiday"]


def test_model_inputs_exist_in_features(session_config, feature_matrix):
    assert not set(model_input_columns(session_config)) - set(feature_matrix.columns)


def test_load_features_missing_file_raises(config):
    with pytest.raises(FileNotFoundError, match="python -m src.features"):
        load_features(config)


def test_load_features_parses_dates(session_config, feature_matrix):
    df = load_features(session_config)
    assert pd.api.types.is_datetime64_any_dtype(df["fecha_hora"])
    assert df.shape == feature_matrix.shape


# --------------------------------------------------------------------------- partición
def hourly_frame(hours: int = 100, stations: int = 2) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=hours, freq="h")
    return pd.DataFrame({"Origen_Id": np.repeat(np.arange(stations), hours), "fecha_hora": np.tile(ts, stations)})


def test_temporal_split_proportions_and_order(session_config):
    df = hourly_frame(100)
    split = temporal_split(df, session_config)
    hours_by_part = {p: df.loc[split == p, "fecha_hora"] for p in ("train", "val", "test")}
    # 70 / 15 / 15 horas de 100, para cada una de las 2 estaciones.
    assert {p: s.nunique() for p, s in hours_by_part.items()} == {"train": 70, "val": 15, "test": 15}
    assert (split.value_counts() == pd.Series({"train": 140, "val": 30, "test": 30})).all()
    assert hours_by_part["train"].max() < hours_by_part["val"].min()
    assert hours_by_part["val"].max() < hours_by_part["test"].min()
    assert split.name == "split"


def test_temporal_split_ignores_row_order(session_config):
    df = hourly_frame(100)
    shuffled = df.sample(frac=1.0, random_state=0)
    expected = temporal_split(df, session_config)
    assert temporal_split(shuffled, session_config).sort_index().equals(expected.sort_index())


def test_temporal_split_every_hour_in_single_partition(session_config):
    df = hourly_frame(100, stations=3)
    labels = df.assign(split=temporal_split(df, session_config)).groupby("fecha_hora")["split"].nunique()
    assert (labels == 1).all()


def test_temporal_split_is_chronological(feature_matrix, split_labels):
    times = feature_matrix["fecha_hora"]
    assert times[split_labels == "train"].max() < times[split_labels == "val"].min()
    assert times[split_labels == "val"].max() < times[split_labels == "test"].min()


# --------------------------------------------------------------------------- punto de entrada
def test_features_main_without_mlflow(session_config, cleaning_result, write_config, monkeypatch, tmp_path):
    cfg = copy.deepcopy(session_config)
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    monkeypatch.setattr("sys.argv", ["features", "--config", str(write_config(cfg)), "--no-mlflow"])
    features_module.main()
    out = pd.read_csv(tmp_path / "processed" / cfg["data"]["features_file"])
    assert not set(model_input_columns(cfg)) - set(out.columns)
    assert len(out) > 0
