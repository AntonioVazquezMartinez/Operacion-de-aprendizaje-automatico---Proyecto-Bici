"""Feature store del servicio: demanda horaria reciente por estación para calcular los rezagos en línea.

La idea (guardar con el modelo las últimas horas de demanda por estación y calcular los rezagos en el servicio
cuando el cliente no los envía, con una ventana de horas predecibles) está adaptada de la rama entrega-final
(Jesús Rincón, `src/serving/bundle.py` y `src/serving/service.py`). Aquí los rezagos no se recalculan con
una fórmula propia: se obtienen con `MiBiciFeatureEngineer.add_lag_features`, la misma función que construyó
la matriz de entrenamiento, aplicada a la cuadrícula estación x hora del feature store.

El archivo es un CSV comprimido (`feature_store.csv.gz`) con las columnas `Origen_Id`, `fecha_hora` y el
objetivo (`demand_count`). Se escribe de forma determinista (fin de línea LF, formato de fecha fijo y cabecera
gzip sin nombre de archivo y con mtime=0), así que dos exportaciones con los mismos datos dan el mismo SHA-256.

Este módulo solo depende de pandas, numpy y src.features: se usa en la exportación (src.modeling.export) y
dentro del contenedor del servicio.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.features import MiBiciFeatureEngineer, lag_columns

STATION_KEY = "Origen_Id"
TIME_KEY = "fecha_hora"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def lag_offsets_hours(config: Config) -> List[int]:
    """Desfases (horas hacia atrás) que usan los rezagos de `features`: 24, 168 y 24*d de la media por hora."""
    feats = config["features"]
    offsets = [int(h) for h in feats["lag_hours"]]
    days = feats.get("same_hour_mean_days")
    if days:
        offsets += [24 * d for d in range(1, int(days) + 1)]
    return offsets


def predictable_window(start: pd.Timestamp, end: pd.Timestamp, config: Config) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Primera y última hora con todos los rezagos disponibles a partir de un historial [start, end]."""
    offsets = lag_offsets_hours(config)
    return start + pd.Timedelta(hours=max(offsets)), end + pd.Timedelta(hours=min(offsets))


def build_feature_store(df: pd.DataFrame, config: Config, hours: int) -> pd.DataFrame:
    """Últimas `hours` horas de demanda por estación de la matriz procesada (cuadrícula horaria y continua)."""
    target = config["features"]["target"]
    needed = max(lag_offsets_hours(config))
    if hours <= needed:
        raise ValueError(f"serving.feature_store_hours ({hours}) debe ser mayor que el rezago más largo ({needed} h).")
    times = pd.to_datetime(df[TIME_KEY])
    end = times.max()
    start = end - pd.Timedelta(hours=hours - 1)
    store = df.loc[times >= start, [STATION_KEY, TIME_KEY, target]].copy()
    store[TIME_KEY] = pd.to_datetime(store[TIME_KEY])
    store[STATION_KEY] = store[STATION_KEY].astype(np.int64)
    store[target] = store[target].astype(np.int64)
    return store.sort_values([STATION_KEY, TIME_KEY]).reset_index(drop=True)


def feature_store_bytes(store: pd.DataFrame) -> bytes:
    """CSV comprimido con gzip de forma determinista (mismo contenido -> mismos bytes y mismo SHA-256)."""
    text = store.to_csv(index=False, lineterminator="\n", date_format=DATE_FORMAT)
    buffer = io.BytesIO()
    # filename="" y mtime=0: la cabecera gzip no guarda ni el nombre ni la fecha del archivo.
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as gz:
        gz.write(text.encode("utf-8"))
    return buffer.getvalue()


def write_feature_store(store: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(feature_store_bytes(store))
    return path


def read_feature_store(path: Path, config: Config) -> pd.DataFrame:
    target = config["features"]["target"]
    store = pd.read_csv(path, compression="gzip", parse_dates=[TIME_KEY])
    missing = {STATION_KEY, TIME_KEY, target} - set(store.columns)
    if missing:
        raise ValueError(f"Al feature store le faltan columnas: {sorted(missing)}")
    return store


def _lag_engineer(config: Config) -> MiBiciFeatureEngineer:
    """Instancia de MiBiciFeatureEngineer solo para `add_lag_features`.

    El constructor resuelve las rutas de datos con `Config.file`, que crea carpetas (data/raw, data/interim,
    ...): dentro del contenedor el usuario sin privilegios no puede ni debe crearlas, así que se omite. Se fuerza
    `drop_incomplete_history: true`: una hora sin historial completo nunca se rellena con ceros, queda fuera de
    la ventana predecible.
    """
    engineer = object.__new__(MiBiciFeatureEngineer)
    engineer.config = config
    engineer.feats = {**config["features"], "drop_incomplete_history": True}
    engineer.target = config["features"]["target"]
    return engineer


def compute_lags(history: pd.DataFrame, config: Config, until: pd.Timestamp) -> pd.DataFrame:
    """Rezagos de cada estación para las horas hasta `until` con `MiBiciFeatureEngineer.add_lag_features`.

    `history` tiene la demanda observada por estación y hora; las horas posteriores a la última observada se
    agregan con demanda desconocida (NaN), de modo que sus rezagos solo usan demanda pasada. Regresa una fila
    por (estación, hora) con todos los rezagos disponibles.
    """
    target = config["features"]["target"]
    start = pd.to_datetime(history[TIME_KEY]).min()
    stations = np.sort(history[STATION_KEY].unique())
    hours = pd.date_range(start, until, freq="h")
    series = history.set_index([STATION_KEY, TIME_KEY])[target].astype(float)
    if series.index.has_duplicates:
        raise ValueError("El historial de demanda tiene horas repetidas para una misma estación.")
    grid_index = pd.MultiIndex.from_product([stations, hours], names=[STATION_KEY, TIME_KEY])
    grid = series.reindex(grid_index).rename(target).reset_index()   # cuadrícula continua por estación
    lags = _lag_engineer(config).add_lag_features(grid)
    return lags[[STATION_KEY, TIME_KEY, *lag_columns(config)]]


class FeatureStore:
    """Historial reciente de demanda y rezagos precalculados para cada hora de la ventana predecible."""

    def __init__(self, store: pd.DataFrame, config: Config):
        self.lag_names = lag_columns(config)
        times = pd.to_datetime(store[TIME_KEY])
        self.start, self.end = times.min(), times.max()
        self.window_start, self.window_end = predictable_window(self.start, self.end, config)
        self.num_stations = int(store[STATION_KEY].nunique())
        self.rows = int(len(store))
        self.hours = int((self.end - self.start) / pd.Timedelta(hours=1)) + 1
        if self.rows != self.num_stations * self.hours:
            raise ValueError(f"El feature store no es una cuadrícula completa: {self.rows} filas para "
                             f"{self.num_stations} estaciones x {self.hours} horas.")
        lags = compute_lags(store, config, self.window_end)
        self.table = lags.set_index([STATION_KEY, TIME_KEY])[self.lag_names].sort_index()

    @classmethod
    def load(cls, path: Path, config: Config) -> "FeatureStore":
        return cls(read_feature_store(path, config), config)

    def lookup(self, station_ids: Sequence[int], hours: Sequence[Any]) -> np.ndarray:
        """Matriz (n, rezagos) en el orden de `lag_names`; NaN si la hora no tiene historial suficiente."""
        keys = pd.MultiIndex.from_arrays([np.asarray(station_ids, dtype=np.int64), pd.to_datetime(list(hours))])
        return self.table.reindex(keys).to_numpy(dtype=np.float64)

    def window(self) -> Dict[str, str]:
        return {"start": self.window_start.isoformat(), "end": self.window_end.isoformat()}

    def info(self) -> Dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "hours": self.hours,
                "num_stations": self.num_stations, "rows": self.rows, "predictable_window": self.window()}
