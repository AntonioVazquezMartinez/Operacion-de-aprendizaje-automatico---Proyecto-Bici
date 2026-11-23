"""Generadores de conjuntos de monitoreo con distribución alterada respecto de la línea base.

Dos familias:

1. Corrupción de variables de entrada (`familia = "corrupcion_entradas"`): la demanda real observada NO
   cambia; se simula una falla del pipeline que calcula las variables (rezagos faltantes, unidades o
   escala equivocadas, reloj/zona horaria corridos). El modelo es el mismo y el mundo es el mismo:
   cualquier degradación es culpa de los datos de entrada.
2. Cambio real de distribución (`familia = "cambio_real"`): cambia el mundo. Se alteran los VIAJES
   (más o menos viajes, estaciones nuevas) y se recalculan la cuadrícula estación-hora, las variables y
   el objetivo con la misma lógica de `MiBiciFeatureEngineer`, o se toma un periodo real distinto
   (vacaciones de diciembre). Aquí la degradación indica que el modelo quedó desactualizado.

Todas las funciones son puras: reciben DataFrames y regresan copias; el azar usa la semilla recibida.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features import MiBiciFeatureEngineer

CALENDAR_COLUMNS = ["hour", "day_of_week", "month", "is_weekend", "is_holiday",
                    "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"]


# --------------------------------------------------------------------------- familia 1: corrupción
def missing_lags(base: pd.DataFrame, lag_cols: Sequence[str], fraction: float, seed: int) -> pd.DataFrame:
    """Pone NaN en TODOS los rezagos de una fracción aleatoria de filas (falla de la consulta histórica)."""
    out = base.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(out)) < float(fraction)
    out.loc[mask, list(lag_cols)] = np.nan
    return out


def scale_lags(base: pd.DataFrame, lag_cols: Sequence[str], factor: float) -> pd.DataFrame:
    """Multiplica los rezagos por `factor` (unidades o agregación equivocadas en el pipeline)."""
    out = base.copy()
    out[list(lag_cols)] = out[list(lag_cols)].astype(float) * float(factor)
    return out


def clock_shift(base: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Recalcula TODAS las variables de calendario con el reloj corrido `hours` horas (p. ej. UTC vs. local).

    Los rezagos y el objetivo no cambian; `fecha_hora` conserva la hora real para poder evaluar.
    """
    out = base.copy()
    shifted = MiBiciFeatureEngineer.add_calendar_features(
        pd.DataFrame({"fecha_hora": pd.to_datetime(base["fecha_hora"]) + pd.Timedelta(hours=int(hours))}))
    for col in CALENDAR_COLUMNS:
        if col in out.columns:
            out[col] = shifted[col].to_numpy()
    return out


# --------------------------------------------------------------------------- familia 2: cambio real
def change_demand(trips: pd.DataFrame, factor: float, seed: int) -> pd.DataFrame:
    """Cambia el volumen de viajes por remuestreo: factor > 1 duplica viajes al azar (con reemplazo),
    factor < 1 conserva cada viaje con probabilidad `factor` (adelgazamiento binomial)."""
    factor = float(factor)
    rng = np.random.default_rng(seed)
    if factor >= 1:
        n_extra = int(round(len(trips) * (factor - 1)))
        extra = trips.iloc[rng.integers(0, len(trips), n_extra)]
        return pd.concat([trips, extra], ignore_index=True)
    keep = rng.random(len(trips)) < factor
    return trips.loc[keep].reset_index(drop=True)


def new_stations(trips: pd.DataFrame, stations: pd.DataFrame, station_ids: Sequence[int], fraction: float,
                 opening: pd.Timestamp, id_offset: int, seed: int
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], Dict[int, int]]:
    """Simula estaciones nuevas: una fracción de estaciones recibe un Id nunca visto y "abre" en `opening`.

    Sus viajes previos a la apertura se eliminan (no tienen historial: rezagos en 0 la primera semana)
    y el TargetEncoder no conoce su Id. Ubicación y capacidad se copian de la estación original.
    Regresa (viajes, catálogo extendido, Ids de la cuadrícula, mapeo original -> nuevo).
    """
    rng = np.random.default_rng(seed)
    ids = np.sort(np.asarray(station_ids))
    n_new = max(1, int(round(len(ids) * float(fraction))))
    chosen = np.sort(rng.choice(ids, size=n_new, replace=False))
    mapping = {int(old): int(old) + int(id_offset) for old in chosen}

    out = trips.copy()
    moved = out["Origen_Id"].isin(chosen)
    before_opening = pd.to_datetime(out["Inicio_del_viaje"]) < opening
    out = out.loc[~(moved & before_opening)].copy()
    out["Origen_Id"] = np.where(out["Origen_Id"].isin(chosen), out["Origen_Id"] + int(id_offset),
                                out["Origen_Id"]).astype(int)

    clones = stations[stations["id"].isin(chosen)].copy()
    clones["id"] = clones["id"].map(mapping)
    catalog = pd.concat([stations, clones], ignore_index=True)
    grid_ids = [mapping.get(int(s), int(s)) for s in ids]
    return out, catalog, grid_ids, mapping


def time_window(base: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Subconjunto real de la línea base entre `start` y `end` (inclusive)."""
    ts = pd.to_datetime(base["fecha_hora"])
    return base.loc[(ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))].reset_index(drop=True)


# --------------------------------------------------------------------------- reconstrucción de variables
def history_hours(config: dict) -> int:
    """Horas de historial que necesitan los rezagos más largos."""
    feats = config["features"]
    return int(max(max(feats["lag_hours"]), 24 * int(feats.get("same_hour_mean_days") or 0)))


def rebuild_features(engineer: MiBiciFeatureEngineer, trips: pd.DataFrame, stations: pd.DataFrame,
                     station_ids: Iterable[int], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Recalcula la matriz estación-hora [start, end] desde viajes con la lógica de `MiBiciFeatureEngineer`.

    La cuadrícula se fija a `station_ids` x horas [start - historial, end] (las estaciones u horas sin
    viajes quedan en 0, igual que en la matriz anual) y luego se aplican calendario, estación y rezagos.
    Sin alteraciones reproduce exactamente las filas de la matriz procesada en esa ventana.
    """
    hist_start = pd.Timestamp(start) - pd.Timedelta(hours=history_hours(engineer.config))
    ts = pd.to_datetime(trips["Inicio_del_viaje"])
    window = trips.loc[(ts >= hist_start) & (ts < pd.Timestamp(end) + pd.Timedelta(hours=1))]
    demand = engineer.build_demand_grid(window)
    index = pd.MultiIndex.from_product([list(station_ids), pd.date_range(hist_start, end, freq="h")],
                                       names=["Origen_Id", "fecha_hora"])
    grid = (demand.set_index(["Origen_Id", "fecha_hora"])[engineer.target]
            .reindex(index, fill_value=0).reset_index())
    grid = grid.sort_values(["Origen_Id", "fecha_hora"]).reset_index(drop=True)
    grid = engineer.add_calendar_features(grid)
    grid = engineer.add_station_features(grid, stations)
    grid = engineer.add_lag_features(grid)
    grid = grid.loc[grid["fecha_hora"] >= pd.Timestamp(start)]
    return grid.sort_values(["fecha_hora", "Origen_Id"]).reset_index(drop=True)
