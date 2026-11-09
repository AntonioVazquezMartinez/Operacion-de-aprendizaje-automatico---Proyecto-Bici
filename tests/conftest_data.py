"""Datos sintéticos y configuración temporal compartidos por las pruebas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config


def make_raw_fixture(raw_dir: Path, n_trips: int = 3000, days: int = 21, seed: int = 42) -> None:
    """Viajes sintéticos con anomalías inyectadas en cantidades conocidas."""
    rng = np.random.default_rng(seed)
    stations = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": [f"(GDL-00{i}) Estación {i}" for i in range(1, 5)],
        "latitude": [20.67, 20.68, 20.66, 20.69],
        "longitude": [-103.35, -103.36, -103.37, -103.34],
        "capacity": [15, 20, 18, 12],
    })
    start = pd.Timestamp("2024-05-01") + pd.to_timedelta(rng.integers(0, days * 24 * 3600, n_trips), unit="s")
    duration = pd.to_timedelta(rng.lognormal(6.3, 0.5, n_trips), unit="s")
    trips = pd.DataFrame({
        "Viaje_Id": np.arange(n_trips),
        "Usuario_Id": rng.integers(1, 500, n_trips),
        "Genero": rng.choice(np.array(["M", "F", None], dtype=object), n_trips),
        "Año_de_nacimiento": rng.integers(1960, 2005, n_trips).astype(float),
        "Inicio_del_viaje": start,
        "Fin_del_viaje": start + duration,
        "Origen_Id": rng.choice(stations["id"], n_trips),
        "Destino_Id": rng.choice(stations["id"], n_trips),
    })
    trips.loc[0:9, "Fin_del_viaje"] = trips.loc[0:9, "Inicio_del_viaje"] - pd.Timedelta(minutes=5)
    trips.loc[10:19, "Fin_del_viaje"] = trips.loc[10:19, "Inicio_del_viaje"] + pd.Timedelta(seconds=20)
    trips.loc[20:29, "Fin_del_viaje"] = trips.loc[20:29, "Inicio_del_viaje"] + pd.Timedelta(hours=5)
    trips.loc[30:34, "Origen_Id"] = 999
    trips.loc[35:39, "Año_de_nacimiento"] = 200
    trips = pd.concat([trips, trips.iloc[[50, 51]]])

    raw_dir.mkdir(parents=True, exist_ok=True)
    trips.to_csv(raw_dir / "viajes_raw.csv", index=False)
    stations.to_csv(raw_dir / "nomenclatura_estaciones.csv", index=False)


def make_test_config(tmp: Path) -> Config:
    """Configuración del proyecto con todas las rutas y MLflow dentro de un directorio temporal."""
    return load_config(overrides={
        "paths": {
            "raw_dir": str(tmp / "raw"),
            "interim_dir": str(tmp / "interim"),
            "processed_dir": str(tmp / "processed"),
            "external_dir": str(tmp / "external"),
            "models_dir": str(tmp / "models"),
            "figures_dir": str(tmp / "figures"),
            "reports_dir": str(tmp / "reports"),
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{(tmp / 'mlruns.db').as_posix()}",
            "artifact_dir": str(tmp / "mlartifacts"),
        },
        "tuning": {"n_trials": 2, "timeout_seconds": 60},
    })
