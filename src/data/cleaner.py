"""Módulo de limpieza, validación y control de calidad de datos para MiBici."""

import os
from typing import Dict, Tuple
import pandas as pd

from src.utils.mlflow_logger import MLflowDataVersioner


class MiBiciDataCleaner:
    """Aplica reglas de negocio y filtros de calidad sobre los viajes brutos de MiBici."""

    def __init__(
        self,
        raw_trips_path: str = "data/raw/viajes_raw.csv",
        raw_stations_path: str = "data/raw/nomenclatura_estaciones.csv",
        output_dir: str = "data/interim",
    ):
        self.raw_trips_path = raw_trips_path
        self.raw_stations_path = raw_stations_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics: Dict[str, float] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carga los archivos raw de viajes y estaciones."""
        if not os.path.exists(self.raw_trips_path):
            raise FileNotFoundError(f"No se encontró el archivo de viajes en: {self.raw_trips_path}")
        if not os.path.exists(self.raw_stations_path):
            raise FileNotFoundError(f"No se encontró el catálogo de estaciones en: {self.raw_stations_path}")

        df_trips = pd.read_csv(self.raw_trips_path)
        df_stations = pd.read_csv(self.raw_stations_path)
        return df_trips, df_stations

    def clean(self) -> Tuple[pd.DataFrame, str]:
        """Ejecuta el pipeline completo de limpieza y depuración."""
        print("=== Iniciando Limpieza y Validación de Datos (Data Engineer / ML Engineer) ===")
        df_trips, df_stations = self.load_data()
        initial_rows = len(df_trips)
        valid_station_ids = set(df_stations["id"].unique())

        # 1. Eliminación de duplicados en Viaje_Id
        df_trips = df_trips.drop_duplicates(subset=["Viaje_Id"]).copy()
        dedup_rows = len(df_trips)
        duplicates_removed = initial_rows - dedup_rows

        # 2. Conversión a tipo Datetime
        df_trips["Inicio_del_viaje"] = pd.to_datetime(df_trips["Inicio_del_viaje"], errors="coerce")
        df_trips["Fin_del_viaje"] = pd.to_datetime(df_trips["Fin_del_viaje"], errors="coerce")
        df_trips = df_trips.dropna(subset=["Inicio_del_viaje", "Fin_del_viaje"]).copy()

        # 3. Cálculo de duración en segundos y minutos
        df_trips["duracion_segundos"] = (df_trips["Fin_del_viaje"] - df_trips["Inicio_del_viaje"]).dt.total_seconds()
        df_trips["duracion_minutos"] = df_trips["duracion_segundos"] / 60.0

        # 4. Filtrado de anomalías de duración
        # - Descartar duraciones negativas (errores de reloj)
        neg_mask = df_trips["duracion_segundos"] <= 0
        dropped_neg = int(neg_mask.sum())
        df_trips = df_trips[~neg_mask].copy()

        # - Descartar viajes ultracortos (< 1 minuto: intentos fallidos de desanclaje)
        short_mask = df_trips["duracion_segundos"] < 60
        dropped_short = int(short_mask.sum())
        df_trips = df_trips[~short_mask].copy()

        # - Descartar viajes excesivos (> 120 minutos: posibles pérdidas, robos o penalizaciones)
        long_mask = df_trips["duracion_minutos"] > 120
        dropped_long = int(long_mask.sum())
        df_trips = df_trips[~long_mask].copy()

        # 5. Filtrado de estaciones inexistentes en catálogo
        invalid_origin = ~df_trips["Origen_Id"].isin(valid_station_ids)
        invalid_dest = ~df_trips["Destino_Id"].isin(valid_station_ids)
        dropped_invalid_stations = int((invalid_origin | invalid_dest).sum())
        df_trips = df_trips[~(invalid_origin | invalid_dest)].copy()

        # 6. Manejo de demografía y edad
        df_trips["Genero"] = df_trips["Genero"].fillna("Desconocido")
        # Imputación de año de nacimiento con la mediana
        median_birth_year = df_trips["Año_de_nacimiento"].median()
        df_trips["Año_de_nacimiento"] = df_trips["Año_de_nacimiento"].fillna(median_birth_year)
        reference_year = df_trips["Inicio_del_viaje"].dt.year.max()
        df_trips["Edad"] = reference_year - df_trips["Año_de_nacimiento"]

        # Filtrar edades no razonables para usuarios del servicio (14 a 90 años)
        invalid_age = (df_trips["Edad"] < 14) | (df_trips["Edad"] > 90)
        dropped_age = int(invalid_age.sum())
        df_trips = df_trips[~invalid_age].copy()

        final_rows = len(df_trips)
        total_dropped = initial_rows - final_rows
        retention_rate = (final_rows / initial_rows) * 100.0

        # Guardar métricas para registro
        self.metrics = {
            "initial_rows": float(initial_rows),
            "final_rows": float(final_rows),
            "total_dropped_rows": float(total_dropped),
            "dropped_duplicates": float(duplicates_removed),
            "dropped_negative_durations": float(dropped_neg),
            "dropped_short_trips": float(dropped_short),
            "dropped_excessive_durations": float(dropped_long),
            "dropped_invalid_stations": float(dropped_invalid_stations),
            "dropped_invalid_ages": float(dropped_age),
            "retention_rate_pct": round(retention_rate, 2),
            "mean_duration_minutes": round(float(df_trips["duracion_minutos"].mean()), 2),
            "median_duration_minutes": round(float(df_trips["duracion_minutos"].median()), 2),
        }

        output_path = os.path.join(self.output_dir, "viajes_cleaned.csv")
        df_trips.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Dataset limpio guardado en: {output_path}")
        print(f"Filas iniciales: {initial_rows} -> Filas limpias: {final_rows} (Retención: {retention_rate:.2f}%)")
        return df_trips, output_path

    def run(self) -> None:
        """Ejecuta la limpieza y versiona los datos intermedios en MLflow."""
        df_clean, output_path = self.clean()

        versioner = MLflowDataVersioner()
        run_id = versioner.log_dataset_version(
            run_name="02_Interim_Data_Cleaning",
            df=df_clean,
            file_path=output_path,
            stage="interim",
            extra_params={
                "min_duration_seconds": 60,
                "max_duration_minutes": 120,
                "age_range": "14-90",
            },
            extra_metrics=self.metrics,
            tags={"role": "Data Engineer", "version": "v1.1.0-interim"},
        )
        print(f"Versión INTERIM registrada en MLflow con Run ID: {run_id}")


if __name__ == "__main__":
    cleaner = MiBiciDataCleaner()
    cleaner.run()
