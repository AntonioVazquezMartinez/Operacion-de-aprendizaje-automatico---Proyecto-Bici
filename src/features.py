"""Etapa de características: demanda horaria por estación, variables de calendario y rezagos,
partición cronológica y transformador de preprocesamiento de scikit-learn.

Uso:
    python -m src.features                # genera data/processed/mibici_hourly_demand.csv
    python -m src.features --no-mlflow
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from src.config import Config, base_arg_parser, load_config


# --------------------------------------------------------------------------- calendario
def _nth_monday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    first_monday = first + timedelta(days=(7 - first.weekday()) % 7)
    return first_monday + timedelta(weeks=n - 1)


def mexican_holidays(year: int) -> Set[date]:
    """Días de descanso obligatorio en México (Art. 74 de la Ley Federal del Trabajo)."""
    holidays = {
        date(year, 1, 1),
        _nth_monday(year, 2, 1),   # Día de la Constitución
        _nth_monday(year, 3, 3),   # Natalicio de Benito Juárez
        date(year, 5, 1),
        date(year, 9, 16),
        _nth_monday(year, 11, 3),  # Revolución Mexicana
        date(year, 12, 25),
    }
    if (year - 2024) % 6 == 0:     # Transmisión del Poder Ejecutivo Federal
        holidays.add(date(year, 10, 1))
    return holidays


# --------------------------------------------------------------------------- columnas
def lag_columns(config: Config) -> List[str]:
    feats = config["features"]
    cols = [f"demand_lag_{h}h" for h in feats["lag_hours"]]
    if feats.get("same_hour_mean_days"):
        cols.append(f"demand_same_hour_mean_{feats['same_hour_mean_days']}d")
    return cols


def feature_columns(config: Config) -> Dict[str, List[str]]:
    """Columnas de entrada del modelo agrupadas por tipo de transformación."""
    feats = config["features"]
    return {
        "numeric": list(feats["numeric_base"]) + lag_columns(config),
        "categorical": list(feats["categorical"]),
        "binary": list(feats["binary"]),
    }


def model_input_columns(config: Config) -> List[str]:
    groups = feature_columns(config)
    return groups["numeric"] + groups["categorical"] + groups["binary"]


# --------------------------------------------------------------------------- ingeniería
class MiBiciFeatureEngineer:
    """Construye la matriz estación-hora con la demanda (extracciones) y sus variables."""

    def __init__(self, config: Config):
        self.config = config
        self.feats = config["features"]
        self.target = self.feats["target"]
        self.cleaned_path = config.file("interim_dir", "interim_file")
        self.stations_path = config.file("raw_dir", "stations_file")
        self.output_path = config.file("processed_dir", "features_file")

    def build_demand_grid(self, df_trips: pd.DataFrame) -> pd.DataFrame:
        """Cuadrícula completa estaciones x horas; las horas sin viajes tienen demanda 0."""
        df_trips = df_trips.copy()
        df_trips["fecha_hora"] = pd.to_datetime(df_trips["Inicio_del_viaje"]).dt.floor("h")
        demand = df_trips.groupby(["Origen_Id", "fecha_hora"]).size().rename(self.target)

        # Solo estaciones activas en el periodo (el catálogo incluye estaciones abiertas después).
        stations = demand.index.get_level_values("Origen_Id").unique()
        hours = pd.date_range(demand.index.get_level_values("fecha_hora").min(),
                              demand.index.get_level_values("fecha_hora").max(), freq="h")
        full_index = pd.MultiIndex.from_product([stations, hours], names=["Origen_Id", "fecha_hora"])
        grid = demand.reindex(full_index, fill_value=0).reset_index()
        return grid.sort_values(["Origen_Id", "fecha_hora"]).reset_index(drop=True)

    @staticmethod
    def add_calendar_features(grid: pd.DataFrame) -> pd.DataFrame:
        grid = grid.copy()
        ts = grid["fecha_hora"]
        grid["hour"] = ts.dt.hour
        grid["day_of_week"] = ts.dt.dayofweek
        grid["month"] = ts.dt.month
        grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(int)
        holidays = set().union(*(mexican_holidays(y) for y in ts.dt.year.unique()))
        grid["is_holiday"] = ts.dt.date.isin(holidays).astype(int)
        # Codificación cíclica: la hora 23 queda junto a la 0 y el domingo junto al lunes.
        grid["sin_hour"] = np.sin(2 * np.pi * grid["hour"] / 24.0)
        grid["cos_hour"] = np.cos(2 * np.pi * grid["hour"] / 24.0)
        grid["sin_dayofweek"] = np.sin(2 * np.pi * grid["day_of_week"] / 7.0)
        grid["cos_dayofweek"] = np.cos(2 * np.pi * grid["day_of_week"] / 7.0)
        return grid

    @staticmethod
    def add_station_features(grid: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
        stations = df_stations[["id", "latitude", "longitude", "capacity"]]
        grid = grid.merge(stations, left_on="Origen_Id", right_on="id", how="left").drop(columns=["id"])
        return grid.sort_values(["Origen_Id", "fecha_hora"]).reset_index(drop=True)

    def add_lag_features(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Rezagos por estación usando solo información pasada (la cuadrícula es horaria y continua)."""
        grid = grid.copy()
        by_station = grid.groupby("Origen_Id")[self.target]
        for h in self.feats["lag_hours"]:
            grid[f"demand_lag_{h}h"] = by_station.shift(h)
        days = self.feats.get("same_hour_mean_days")
        if days:
            same_hour = [by_station.shift(24 * d) for d in range(1, days + 1)]
            grid[f"demand_same_hour_mean_{days}d"] = pd.concat(same_hour, axis=1).mean(axis=1, skipna=False)

        lags = lag_columns(self.config)
        if self.feats.get("drop_incomplete_history", True):
            grid = grid.dropna(subset=lags).reset_index(drop=True)
        else:
            grid[lags] = grid[lags].fillna(0)
        return grid

    def build_features(self, df_trips: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
        grid = self.build_demand_grid(df_trips)
        grid = self.add_calendar_features(grid)
        grid = self.add_station_features(grid, df_stations)
        return self.add_lag_features(grid)

    def run(self) -> Tuple[pd.DataFrame, Path]:
        print("=== Ingeniería de características ===")
        df_trips = pd.read_csv(self.cleaned_path, parse_dates=["Inicio_del_viaje"])
        df_stations = pd.read_csv(self.stations_path)
        df_features = self.build_features(df_trips, df_stations)
        df_features.to_csv(self.output_path, index=False, encoding="utf-8")
        print(f"Matriz de características: {self.output_path} {df_features.shape}")
        return df_features, self.output_path


# --------------------------------------------------------------------------- partición
def temporal_split(df: pd.DataFrame, config: Config) -> pd.Series:
    """Asigna train/val/test por bloques de horas consecutivas (sin mezclar futuro y pasado)."""
    split_cfg = config["split"]
    hours = np.sort(pd.to_datetime(df["fecha_hora"]).unique())
    train_end = hours[int(len(hours) * split_cfg["train_size"]) - 1]
    val_end = hours[int(len(hours) * (split_cfg["train_size"] + split_cfg["val_size"])) - 1]
    ts = pd.to_datetime(df["fecha_hora"])
    labels = np.select([ts <= train_end, ts <= val_end], ["train", "val"], "test")
    return pd.Series(labels, index=df.index, name="split")


def load_features(config: Config) -> pd.DataFrame:
    path = config.file("processed_dir", "features_file")
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: python -m src.features")
    return pd.read_csv(path, parse_dates=["fecha_hora"])


# --------------------------------------------------------------------------- preprocesamiento
def build_preprocessor(config: Config) -> ColumnTransformer:
    """Transformador de entrada del modelo (se ajusta solo con los datos de entrenamiento).

    - Numéricas: imputación por mediana (defensiva) + StandardScaler.
    - Estación (cientos de categorías): TargetEncoder con validación cruzada interna,
      en lugar de one-hot, para no generar cientos de columnas dispersas.
    - Binarias: sin transformación.
    """
    groups = feature_columns(config)
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), groups["numeric"]),
            ("station", Pipeline([
                ("encoder", TargetEncoder(target_type="continuous", random_state=config["seed"])),
                ("scaler", StandardScaler()),
            ]), groups["categorical"]),
            ("bin", "passthrough", groups["binary"]),
        ],
        verbose_feature_names_out=False,
    )


def run(config: Config, use_mlflow: bool = True) -> None:
    engineer = MiBiciFeatureEngineer(config)
    df_features, output_path = engineer.run()
    if use_mlflow:
        from src.tracking import MLflowDataVersioner

        target = config["features"]["target"]
        MLflowDataVersioner(config).log_dataset_version(
            run_name="03_Processed_Feature_Engineering",
            df=df_features,
            file_path=output_path,
            stage="processed",
            extra_params={
                "target_variable": target,
                "lag_hours": config["features"]["lag_hours"],
                "same_hour_mean_days": config["features"].get("same_hour_mean_days"),
                "model_inputs": ",".join(model_input_columns(config)),
            },
            extra_metrics={
                "mean_hourly_demand": round(float(df_features[target].mean()), 3),
                "zero_demand_pct": round(float((df_features[target] == 0).mean() * 100), 2),
                "num_stations": float(df_features["Origen_Id"].nunique()),
                "num_hours": float(df_features["fecha_hora"].nunique()),
            },
            tags={"role": "Data Scientist", "phase": "Fase 2"},
        )
        print("Versión processed registrada en MLflow.")


def main() -> None:
    parser = base_arg_parser("Ingeniería de características de MiBici")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
