"""Servicio de predicción: carga el paquete de despliegue y construye las variables en línea.

El cliente solo envía la estación y la fecha-hora. Las variables de calendario se calculan con
la misma función que usa el entrenamiento (`MiBiciFeatureEngineer.add_calendar_features`) y los
rezagos se buscan en el feature store del paquete, con lo que se evita el training-serving skew.
Si el cliente ya tiene los rezagos, puede enviarlos y tienen prioridad sobre el feature store.
"""

from __future__ import annotations

import time
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.features import MiBiciFeatureEngineer
from src.serving.bundle import MODEL_FILE, STATIONS_FILE, STORE_FILE, load_card

# Guadalajara no tiene horario de verano desde 2022: UTC-06:00 todo el año.
LOCAL_TZ = timezone(timedelta(hours=-6))


class PredictionError(ValueError):
    """Error de entrada que la API devuelve como 422 con un mensaje claro."""


class ModelService:
    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)
        if not (self.bundle_dir / MODEL_FILE).exists():
            raise FileNotFoundError(
                f"No existe {self.bundle_dir / MODEL_FILE}. Ejecuta primero: python -m src.serving.export_model")
        self.card: Dict[str, Any] = load_card(self.bundle_dir)
        self.model = joblib.load(self.bundle_dir / MODEL_FILE)
        self.input_columns: List[str] = list(self.card["input_columns"])
        self.lag_hours: List[int] = [int(h) for h in self.card["lag_hours"]]
        self.same_hour_days: Optional[int] = self.card.get("same_hour_mean_days")

        store = pd.read_csv(self.bundle_dir / STORE_FILE, parse_dates=["fecha_hora"])
        self.history = store.set_index(["Origen_Id", "fecha_hora"])["demand_count"].astype(float).sort_index()
        self.store_start = store["fecha_hora"].min()
        self.store_end = store["fecha_hora"].max()
        self.stations = pd.read_csv(self.bundle_dir / STATIONS_FILE).set_index("id")
        self.loaded_at = time.time()

    # ------------------------------------------------------------------ información
    @property
    def lag_columns(self) -> List[str]:
        cols = [f"demand_lag_{h}h" for h in self.lag_hours]
        if self.same_hour_days:
            cols.append(f"demand_same_hour_mean_{self.same_hour_days}d")
        return cols

    def predictable_window(self) -> Dict[str, str]:
        """Rango de fechas que se puede predecir solo con el feature store."""
        max_offset = max(self.lag_hours + [24 * (self.same_hour_days or 0)])
        min_offset = min(self.lag_hours + ([24] if self.same_hour_days else []))
        return {"desde": str(self.store_start + pd.Timedelta(hours=max_offset)),
                "hasta": str(self.store_end + pd.Timedelta(hours=min_offset))}

    def info(self) -> Dict[str, Any]:
        keys = ("registered_model_name", "model_version", "model_alias", "model_uri", "model_uri_alias",
                "training_run_id", "algorithm", "model_sha256", "registry_metrics", "feature_store_start",
                "feature_store_end", "num_stations", "export_environment", "exported_at_utc")
        return {**{k: self.card.get(k) for k in keys}, "input_columns": self.input_columns,
                "predictable_window": self.predictable_window()}

    # ------------------------------------------------------------------ variables
    @staticmethod
    def _to_local_hour(values: pd.Series) -> pd.Series:
        def convert(ts: pd.Timestamp) -> pd.Timestamp:
            ts = pd.Timestamp(ts)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(LOCAL_TZ).tz_localize(None)
            return ts.floor("h")
        return values.map(convert).astype("datetime64[ns]")

    def _lookup(self, stations: pd.Series, times: pd.Series, hours: int) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([stations.to_numpy(), (times - pd.Timedelta(hours=hours)).to_numpy()])
        return self.history.reindex(keys).to_numpy(dtype=float)

    def build_features(self, instances: List[Dict[str, Any]]) -> pd.DataFrame:
        if not instances:
            raise PredictionError("La solicitud no contiene instancias.")
        df = pd.DataFrame(instances)
        df["station_id"] = df["station_id"].astype(int)
        df["fecha_hora"] = self._to_local_hour(df["fecha_hora"])

        unknown = sorted(set(df["station_id"]) - set(self.stations.index))
        if unknown:
            raise PredictionError(f"Estaciones desconocidas o inactivas: {unknown[:20]}")

        for col in self.lag_columns:
            if col not in df.columns:
                df[col] = np.nan
        df["lags_source"] = np.where(df[self.lag_columns].notna().all(axis=1), "request", "feature_store")

        # Rezagos faltantes: se buscan en el feature store.
        for h in self.lag_hours:
            col = f"demand_lag_{h}h"
            df[col] = df[col].astype(float).fillna(pd.Series(self._lookup(df["station_id"], df["fecha_hora"], h),
                                                             index=df.index))
        if self.same_hour_days:
            col = f"demand_same_hour_mean_{self.same_hour_days}d"
            same_hour = np.column_stack([self._lookup(df["station_id"], df["fecha_hora"], 24 * d)
                                         for d in range(1, self.same_hour_days + 1)])
            mean = pd.Series(same_hour.mean(axis=1), index=df.index)  # NaN si falta algún día
            df[col] = df[col].astype(float).fillna(mean)

        missing = df[self.lag_columns].isna().any(axis=1)
        if missing.any():
            window = self.predictable_window()
            bad = df.loc[missing, ["station_id", "fecha_hora"]].head(5).astype(str).to_dict("records")
            raise PredictionError(
                "No hay historial para calcular los rezagos de: " + str(bad) +
                f". Con el feature store se puede predecir del {window['desde']} al {window['hasta']}; "
                "fuera de ese rango envía demand_lag_* y demand_same_hour_mean_* en la solicitud.")

        df = MiBiciFeatureEngineer.add_calendar_features(df)
        df = df.join(self.stations[["latitude", "longitude", "capacity"]], on="station_id")
        df["Origen_Id"] = df["station_id"].astype(float)
        return df

    # ------------------------------------------------------------------ predicción
    def predict_frame(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.input_columns].copy()
        # Se fuerzan los tipos del entrenamiento. Ejemplo real: el TargetEncoder aprendió las estaciones
        # como float (2.0, 3.0...); un id entero (2) se trataría como categoría desconocida y recibiría
        # la media global, cambiando la predicción sin ningún error visible.
        dtypes = self.card.get("input_dtypes") or {"Origen_Id": "float64"}
        return np.clip(self.model.predict(X.astype(dtypes)), 0, None)

    def predict(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        df = self.build_features(instances)
        df["demand_pred"] = self.predict_frame(df)
        out = df[["station_id", "fecha_hora", "demand_pred", "lags_source"] + self.lag_columns].copy()
        out["fecha_hora"] = out["fecha_hora"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return out.to_dict("records")
