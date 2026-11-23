"""Reglas de limpieza de `MiBiciDataCleaner`: una prueba por regla con conteos calculados a mano.

Reglas de config.yaml usadas: duración mínima 60 s, máxima 120 min, edad entre 14 y 90 años.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import dataset
from src.dataset import MiBiciDataCleaner, run

STATIONS = pd.DataFrame({"id": [1, 2, 3], "latitude": [20.6] * 3, "longitude": [-103.3] * 3, "capacity": [10] * 3})
START = pd.Timestamp("2025-03-10 08:00:00")


def make_trips(n: int = 10) -> pd.DataFrame:
    """`n` viajes válidos de 10 minutos entre estaciones del catálogo, usuarios nacidos en 1990."""
    start = START + pd.to_timedelta(np.arange(n), unit="h")
    return pd.DataFrame({
        "Viaje_Id": np.arange(n),
        "Usuario_Id": np.arange(100, 100 + n),
        "Genero": ["M"] * n,
        "Año_de_nacimiento": [1990.0] * n,
        "Inicio_del_viaje": start,
        "Fin_del_viaje": start + pd.Timedelta(minutes=10),
        "Origen_Id": [1] * n,
        "Destino_Id": [2] * n,
    })


def set_duration(df: pd.DataFrame, rows, seconds: float) -> None:
    df.loc[rows, "Fin_del_viaje"] = df.loc[rows, "Inicio_del_viaje"] + pd.Timedelta(seconds=seconds)


def clean(config, df: pd.DataFrame):
    cleaner = MiBiciDataCleaner(config)
    return cleaner.clean_frame(df, STATIONS), cleaner.metrics


RULE_KEYS = ["dropped_duplicates", "dropped_negative_durations", "dropped_short_trips",
             "dropped_excessive_durations", "dropped_invalid_stations", "dropped_invalid_ages"]


def assert_only(metrics, key: str, expected: int) -> None:
    """La regla `key` descarta `expected` filas y las demás reglas no descartan ninguna."""
    for k in RULE_KEYS:
        assert metrics[k] == (expected if k == key else 0), k


# --------------------------------------------------------------------------- una regla por prueba
def test_valid_trips_are_untouched(config):
    df, metrics = clean(config, make_trips(10))
    assert len(df) == 10
    assert metrics["total_dropped_rows"] == 0
    assert metrics["retention_rate_pct"] == 100.0
    assert df["duracion_minutos"].tolist() == [10.0] * 10


def test_rule_duplicates_keep_first(config):
    trips = make_trips(10)
    dup = trips.iloc[[3, 3, 7]].copy()
    dup.loc[:, "Genero"] = "F"                 # mismo Viaje_Id: se conserva la primera aparición
    df, metrics = clean(config, pd.concat([trips, dup], ignore_index=True))
    assert_only(metrics, "dropped_duplicates", 3)
    assert len(df) == 10
    assert (df["Genero"] == "M").all()


@pytest.mark.parametrize("seconds, expected", [(-300, 1), (0, 1), (1, 0)])
def test_rule_non_positive_duration(config, seconds, expected):
    trips = make_trips(10)
    set_duration(trips, [4], seconds)
    _, metrics = clean(config, trips)
    # 1 s es positivo: no lo quita esta regla, sino la de viajes cortos.
    assert metrics["dropped_negative_durations"] == expected
    assert metrics["dropped_negative_durations"] + metrics["dropped_short_trips"] == 1


@pytest.mark.parametrize("seconds, expected", [(20, 1), (59, 1), (60, 0), (61, 0)])
def test_rule_short_trips_below_60_seconds(config, seconds, expected):
    trips = make_trips(10)
    set_duration(trips, [0, 1], seconds)
    _, metrics = clean(config, trips)
    assert_only(metrics, "dropped_short_trips", 2 * expected)


@pytest.mark.parametrize("seconds, expected", [(120 * 60, 0), (120 * 60 + 1, 1), (5 * 3600, 1)])
def test_rule_excessive_duration_over_120_minutes(config, seconds, expected):
    trips = make_trips(10)
    set_duration(trips, [2, 5, 8], seconds)
    _, metrics = clean(config, trips)
    assert_only(metrics, "dropped_excessive_durations", 3 * expected)


@pytest.mark.parametrize("column", ["Origen_Id", "Destino_Id"])
def test_rule_station_not_in_catalog(config, column):
    trips = make_trips(10)
    trips.loc[[1, 6], column] = 999
    df, metrics = clean(config, trips)
    assert_only(metrics, "dropped_invalid_stations", 2)
    assert set(df[column]) <= {1, 2, 3}


@pytest.mark.parametrize("birth_year, dropped", [(2012, 1), (2011, 0), (1935, 0), (1934, 1), (200, 1)])
def test_rule_age_out_of_range(config, birth_year, dropped):
    """Edad = año máximo de los viajes (2025) − año de nacimiento; se aceptan de 14 a 90 años."""
    trips = make_trips(10)
    trips.loc[9, "Año_de_nacimiento"] = birth_year
    df, metrics = clean(config, trips)
    assert_only(metrics, "dropped_invalid_ages", dropped)
    assert df["Edad"].between(14, 90).all()


def test_imputation_of_gender_and_birth_year(config):
    trips = make_trips(5)
    trips["Año_de_nacimiento"] = [1980.0, 1990.0, 2000.0, np.nan, 1970.0]
    trips.loc[[0, 2], "Genero"] = None
    df, _ = clean(config, trips)
    assert df["Genero"].tolist() == ["Desconocido", "M", "Desconocido", "M", "M"]
    # Mediana de [1980, 1990, 2000, 1970] = 1985 -> edad 2025 - 1985 = 40.
    assert df.loc[3, "Año_de_nacimiento"] == 1985.0
    assert df.loc[3, "Edad"] == 40.0


def test_unparseable_dates_are_dropped(config):
    trips = make_trips(6).astype({"Inicio_del_viaje": str})
    trips.loc[2, "Inicio_del_viaje"] = "fecha inválida"
    df, metrics = clean(config, trips)
    assert len(df) == 5
    # Las fechas inválidas cuentan en el total, aunque no tienen una métrica de regla propia.
    assert metrics["total_dropped_rows"] == 1
    assert sum(metrics[k] for k in RULE_KEYS) == 0


def test_all_rules_combined_and_retention_metrics(config):
    """20 viajes: 2 duplicados, 1 negativo, 2 cortos, 1 largo, 1 estación inválida, 1 edad inválida."""
    trips = make_trips(18)
    set_duration(trips, [0], -60)
    set_duration(trips, [1, 2], 30)
    set_duration(trips, [3], 3 * 3600)
    trips.loc[4, "Origen_Id"] = 77
    trips.loc[5, "Año_de_nacimiento"] = 1900
    set_duration(trips, [6], 20 * 60)         # válido de 20 min: mueve media y mediana
    raw = pd.concat([trips, trips.iloc[[10, 11]]], ignore_index=True)

    df, m = clean(config, raw)

    assert [m[k] for k in RULE_KEYS] == [2, 1, 2, 1, 1, 1]
    assert m["initial_rows"] == 20
    assert m["final_rows"] == 12 == len(df)
    assert m["total_dropped_rows"] == 8
    assert m["retention_rate_pct"] == pytest.approx(60.0)
    # 11 viajes de 10 min y 1 de 20 min: media 130/12 = 10.83, mediana 10.
    assert m["mean_duration_minutes"] == pytest.approx(10.83)
    assert m["median_duration_minutes"] == pytest.approx(10.0)
    assert not df["Viaje_Id"].duplicated().any()


# --------------------------------------------------------------------------- limpieza sobre archivos
def test_cleaner_removes_injected_anomalies(cleaning_result):
    """Conteos de las anomalías inyectadas por `make_raw_fixture` (tests/conftest_data.py)."""
    metrics = cleaning_result["metrics"]
    assert metrics["dropped_duplicates"] == 2
    assert metrics["dropped_negative_durations"] == 10
    assert metrics["dropped_short_trips"] == 10
    assert metrics["dropped_excessive_durations"] == 10
    assert metrics["dropped_invalid_stations"] == 5
    assert metrics["dropped_invalid_ages"] == 5
    assert metrics["initial_rows"] == 3002
    assert metrics["final_rows"] == 3002 - 42


def test_cleaner_rules_come_from_config(session_config, clean_trips):
    rules = session_config["cleaning"]
    assert (clean_trips["duracion_segundos"] >= rules["min_duration_seconds"]).all()
    assert (clean_trips["duracion_minutos"] <= rules["max_duration_minutes"]).all()


def test_cleaner_valid_stations(clean_trips, raw_frames):
    valid_ids = set(raw_frames["stations"]["id"])
    assert clean_trips["Origen_Id"].isin(valid_ids).all()
    assert clean_trips["Destino_Id"].isin(valid_ids).all()


def test_clean_writes_interim_file(cleaning_result, session_config):
    assert cleaning_result["path"] == session_config.file("interim_dir", "interim_file")
    assert len(pd.read_csv(cleaning_result["path"])) == len(cleaning_result["df"])


def test_load_data_without_raw_files_raises(config):
    with pytest.raises(FileNotFoundError, match="Ejecuta primero la descarga"):
        MiBiciDataCleaner(config).load_data()


@pytest.fixture
def local_month_files(config, raw_frames, monkeypatch):
    """Coloca en raw_dir un mes de viajes y el catálogo con el formato del portal (sin red)."""
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: pytest.fail("no debe usar la red"))
    config["data"]["months"] = ["2024-05"]
    raw_dir = config.path("raw_dir")
    raw_frames["trips"].to_csv(raw_dir / "datos_abiertos_2024_05.csv", index=False)
    stations = raw_frames["stations"].rename(columns={"capacity": "dpcapacity"})
    stations.to_csv(raw_dir / config["data"]["stations_url"].rsplit("/", 1)[-1], index=False)
    return config


def test_run_without_mlflow_from_local_files(local_month_files):
    """`run` encadena catálogo + viajes + limpieza sin red cuando los archivos ya están en raw_dir."""
    run(local_month_files, use_mlflow=False)
    cleaned = pd.read_csv(local_month_files.file("interim_dir", "interim_file"))
    assert len(cleaned) == 3002 - 42


def test_dataset_main_cli(local_month_files, write_config, monkeypatch):
    monkeypatch.setattr("sys.argv", ["dataset", "--config", str(write_config(local_month_files)), "--no-mlflow"])
    dataset.main()
    assert len(pd.read_csv(local_month_files.file("raw_dir", "raw_trips_file"))) == 3002
    assert len(pd.read_csv(local_month_files.file("interim_dir", "interim_file"))) == 3002 - 42
