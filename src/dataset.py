"""Etapa de datos: descarga de los datos abiertos de MiBici y limpieza con reglas de negocio.

Uso:
    python -m src.dataset                 # descarga (si hace falta) + limpieza + registro en MLflow
    python -m src.dataset --no-mlflow     # sin registrar versiones en MLflow
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests

from src.config import Config, base_arg_parser, load_config

TRIP_COLUMNS = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]


class MiBiciDataDownloader:
    """Obtiene los viajes mensuales y el catálogo de estaciones definidos en config.yaml."""

    def __init__(self, config: Config):
        self.config = config
        self.raw_dir = config.path("raw_dir")
        self.months = list(config["data"]["months"])
        self.trip_urls: Dict[str, str] = config["data"].get("trip_urls", {}) or {}
        self.stations_url: str = config["data"]["stations_url"]
        self.timeout = int(config["data"].get("download_timeout_seconds", 120))

    def _download(self, url: str, path: Path) -> None:
        if path.exists():
            return
        print(f"Descargando {url}")
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        path.write_bytes(response.content)

    def month_path(self, month: str) -> Path:
        year, mm = month.split("-")
        return self.raw_dir / f"datos_abiertos_{year}_{mm}.csv"

    @staticmethod
    def read_month(path: Path) -> pd.DataFrame:
        """Lee un CSV mensual de viajes de forma robusta.

        - Codificación: UTF-8 o latin-1 (MiBici usa ambas según el archivo).
        - Si algún registro trae una comilla suelta, se reintenta leyendo las comillas como
          texto normal (QUOTE_NONE), se descartan las filas mal formadas y se reporta cuántas.
        - Los identificadores se convierten a número; valores inválidos quedan vacíos y la
          limpieza los descarta.
        """
        df = None
        for encoding in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                with open(path, "rb") as fh:
                    data_lines = sum(1 for _ in fh) - 1
                df = pd.read_csv(path, encoding=encoding, quoting=csv.QUOTE_NONE,
                                 on_bad_lines="skip", low_memory=False)
                df.columns = [str(c).strip('"') for c in df.columns]
                for col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].astype("string").str.strip('"')
                print(f"Aviso: {path.name} tenía comillas mal cerradas; se leyó sin comillas "
                      f"y se descartaron {data_lines - len(df)} filas mal formadas.")
                break
        if df is None:
            raise ValueError(f"No se pudo leer {path.name} ni en UTF-8 ni en latin-1.")

        if df.shape[1] != len(TRIP_COLUMNS):
            raise ValueError(
                f"{path.name} tiene {df.shape[1]} columnas y se esperaban {len(TRIP_COLUMNS)}; "
                "revisa si cambió el formato del archivo de ese mes."
            )
        first, last = str(df.columns[0]).lower(), str(df.columns[-1]).lower()
        if "viaje" not in first or "destino" not in last:
            raise ValueError(f"{path.name}: encabezado inesperado {list(df.columns)}")
        df.columns = TRIP_COLUMNS
        for col in ("Viaje_Id", "Usuario_Id", "Año_de_nacimiento", "Origen_Id", "Destino_Id"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def download_stations_catalog(self) -> pd.DataFrame:
        source_path = self.raw_dir / Path(self.stations_url).name
        self._download(self.stations_url, source_path)
        try:
            df_stations = pd.read_csv(source_path, encoding="utf-8")
        except UnicodeDecodeError:
            df_stations = pd.read_csv(source_path, encoding="latin-1")
        df_stations = df_stations.rename(columns={"dpcapacity": "capacity"})
        output_path = self.config.file("raw_dir", "stations_file")
        df_stations.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Catálogo de estaciones: {output_path} ({len(df_stations)} estaciones)")
        return df_stations

    def download_trips(self) -> Tuple[pd.DataFrame, Path]:
        frames = []
        for month in self.months:
            path = self.month_path(month)
            if not path.exists():
                if month not in self.trip_urls:
                    raise FileNotFoundError(
                        f"No existe {path} ni una URL para {month} en data.trip_urls de config.yaml. "
                        "Descarga el CSV del portal de MiBici y colócalo en data/raw."
                    )
                self._download(self.trip_urls[month], path)
            frames.append(self.read_month(path))

        df_trips = pd.concat(frames, ignore_index=True)
        output_path = self.config.file("raw_dir", "raw_trips_file")
        df_trips.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Viajes raw: {output_path} ({len(df_trips)} filas, meses {', '.join(self.months)})")
        return df_trips, output_path


class MiBiciDataCleaner:
    """Aplica reglas de negocio y filtros de calidad sobre los viajes brutos."""

    def __init__(self, config: Config):
        self.config = config
        self.rules = config["cleaning"]
        self.raw_trips_path = config.file("raw_dir", "raw_trips_file")
        self.stations_path = config.file("raw_dir", "stations_file")
        self.output_path = config.file("interim_dir", "interim_file")
        self.metrics: Dict[str, float] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for path in (self.raw_trips_path, self.stations_path):
            if not path.exists():
                raise FileNotFoundError(f"No se encontró {path}. Ejecuta primero la descarga.")
        return pd.read_csv(self.raw_trips_path), pd.read_csv(self.stations_path)

    def clean_frame(self, df_trips: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
        """Limpia un DataFrame de viajes y guarda en self.metrics cuántas filas descartó cada regla."""
        rules = self.rules
        initial_rows = len(df_trips)
        valid_station_ids = set(df_stations["id"].unique())

        # 1. Duplicados de Viaje_Id
        df = df_trips.drop_duplicates(subset=["Viaje_Id"]).copy()
        duplicates = initial_rows - len(df)

        # 2. Fechas válidas y duración
        df["Inicio_del_viaje"] = pd.to_datetime(df["Inicio_del_viaje"], errors="coerce")
        df["Fin_del_viaje"] = pd.to_datetime(df["Fin_del_viaje"], errors="coerce")
        df = df.dropna(subset=["Inicio_del_viaje", "Fin_del_viaje"]).copy()
        df["duracion_segundos"] = (df["Fin_del_viaje"] - df["Inicio_del_viaje"]).dt.total_seconds()
        df["duracion_minutos"] = df["duracion_segundos"] / 60.0

        # 3. Reglas de duración (en orden, para reportar cuántas filas elimina cada una)
        def drop(mask: pd.Series) -> int:
            nonlocal df
            removed = int(mask.sum())
            df = df[~mask].copy()
            return removed

        dropped_neg = drop(df["duracion_segundos"] <= 0)
        dropped_short = drop(df["duracion_segundos"] < rules["min_duration_seconds"])
        dropped_long = drop(df["duracion_minutos"] > rules["max_duration_minutes"])

        # 4. Estaciones que no existen en el catálogo oficial
        dropped_stations = drop(~df["Origen_Id"].isin(valid_station_ids) | ~df["Destino_Id"].isin(valid_station_ids))

        # 5. Demografía: imputación y edades fuera de rango
        df["Genero"] = df["Genero"].fillna("Desconocido")
        df["Año_de_nacimiento"] = df["Año_de_nacimiento"].fillna(df["Año_de_nacimiento"].median())
        df["Edad"] = df["Inicio_del_viaje"].dt.year.max() - df["Año_de_nacimiento"]
        dropped_age = drop((df["Edad"] < rules["min_age"]) | (df["Edad"] > rules["max_age"]))

        final_rows = len(df)
        self.metrics = {
            "initial_rows": float(initial_rows),
            "final_rows": float(final_rows),
            "total_dropped_rows": float(initial_rows - final_rows),
            "dropped_duplicates": float(duplicates),
            "dropped_negative_durations": float(dropped_neg),
            "dropped_short_trips": float(dropped_short),
            "dropped_excessive_durations": float(dropped_long),
            "dropped_invalid_stations": float(dropped_stations),
            "dropped_invalid_ages": float(dropped_age),
            "retention_rate_pct": round(final_rows / initial_rows * 100.0, 2),
            "mean_duration_minutes": round(float(df["duracion_minutos"].mean()), 2),
            "median_duration_minutes": round(float(df["duracion_minutos"].median()), 2),
        }
        return df

    def clean(self) -> Tuple[pd.DataFrame, Path]:
        print("=== Limpieza y validación de datos ===")
        df_trips, df_stations = self.load_data()
        df_clean = self.clean_frame(df_trips, df_stations)
        df_clean.to_csv(self.output_path, index=False, encoding="utf-8")
        print(
            f"{int(self.metrics['initial_rows'])} -> {int(self.metrics['final_rows'])} filas "
            f"(retención {self.metrics['retention_rate_pct']}%). Guardado en {self.output_path}"
        )
        return df_clean, self.output_path


def run(config: Config, use_mlflow: bool = True) -> None:
    """Ejecuta descarga + limpieza y, opcionalmente, registra las versiones raw e interim."""
    downloader = MiBiciDataDownloader(config)
    downloader.download_stations_catalog()
    df_raw, raw_path = downloader.download_trips()

    cleaner = MiBiciDataCleaner(config)
    df_clean, clean_path = cleaner.clean()

    if use_mlflow:
        from src.tracking import MLflowDataVersioner

        versioner = MLflowDataVersioner(config)
        versioner.log_dataset_version(
            run_name="01_Raw_Data_Ingestion",
            df=df_raw,
            file_path=raw_path,
            stage="raw",
            extra_params={"source": "MiBici_Datos_Abiertos", "months": ",".join(downloader.months)},
            tags={"role": "Data Engineer", "phase": "Fase 2"},
        )
        versioner.log_dataset_version(
            run_name="02_Interim_Data_Cleaning",
            df=df_clean,
            file_path=clean_path,
            stage="interim",
            extra_params=config["cleaning"],
            extra_metrics=cleaner.metrics,
            tags={"role": "Data Engineer", "phase": "Fase 2"},
        )
        print("Versiones raw e interim registradas en MLflow.")


def main() -> None:
    parser = base_arg_parser("Descarga y limpieza de los datos de MiBici")
    parser.add_argument("--no-mlflow", action="store_true", help="No registrar las versiones en MLflow")
    args = parser.parse_args()
    run(load_config(args.config), use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
