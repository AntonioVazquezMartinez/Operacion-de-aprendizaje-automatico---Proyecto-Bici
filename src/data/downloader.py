"""Descarga de los datos abiertos de MiBici Guadalajara."""

import os
from typing import Dict, Tuple

import pandas as pd
import requests

from src.utils.mlflow_logger import MLflowDataVersioner

BASE_URL = "https://www.mibici.net/site/assets/files"

TRIPS_URLS: Dict[str, str] = {
    "2024-05": f"{BASE_URL}/81663/datos_abiertos_2024_05.csv",
}
STATIONS_URL = f"{BASE_URL}/1118/nomenclatura_2026_09.csv"

# El portal de MiBici responde HTTP 406 al User-Agent por defecto de requests;
# se envía uno de navegador para que acepte la descarga.
REQUEST_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT_SECONDS = 120

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
    """Descarga los viajes mensuales y el catálogo de estaciones de MiBici."""

    def __init__(self, raw_dir: str = "data/raw", months: Tuple[str, ...] = ("2024-05",)):
        self.raw_dir = raw_dir
        self.months = months
        os.makedirs(self.raw_dir, exist_ok=True)

    def _download(self, url: str, path: str) -> None:
        if os.path.exists(path):
            return
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)

    def download_stations_catalog(self) -> pd.DataFrame:
        source_path = os.path.join(self.raw_dir, os.path.basename(STATIONS_URL))
        self._download(STATIONS_URL, source_path)

        df_stations = pd.read_csv(source_path, encoding="utf-8").rename(columns={"dpcapacity": "capacity"})
        output_path = os.path.join(self.raw_dir, "nomenclatura_estaciones.csv")
        df_stations.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Catálogo de estaciones guardado en: {output_path} ({len(df_stations)} estaciones)")
        return df_stations

    def download_trips(self) -> Tuple[pd.DataFrame, str]:
        frames = []
        for month in self.months:
            source_path = os.path.join(self.raw_dir, os.path.basename(TRIPS_URLS[month]))
            self._download(TRIPS_URLS[month], source_path)
            # El encabezado original trae el nombre de la columna de año de nacimiento corrupto.
            df_month = pd.read_csv(source_path, encoding="latin-1")
            df_month.columns = TRIP_COLUMNS
            frames.append(df_month)

        df_trips = pd.concat(frames, ignore_index=True)
        output_path = os.path.join(self.raw_dir, "viajes_raw.csv")
        df_trips.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Dataset de viajes raw guardado en: {output_path} ({len(df_trips)} filas)")
        return df_trips, output_path

    def run(self) -> None:
        print("=== Iniciando Ingesta de Datos (Data Engineer) ===")
        self.download_stations_catalog()
        df_trips, raw_path = self.download_trips()

        versioner = MLflowDataVersioner()
        run_id = versioner.log_dataset_version(
            run_name="01_Raw_Data_Ingestion",
            df=df_trips,
            file_path=raw_path,
            stage="raw",
            extra_params={
                "source": "MiBici_Datos_Abiertos",
                "months": ",".join(self.months),
                "trips_urls": ",".join(TRIPS_URLS[m] for m in self.months),
                "stations_url": STATIONS_URL,
            },
            tags={"role": "Data Engineer", "version": "v1.0.0-raw"},
        )
        print(f"Versión RAW registrada en MLflow con Run ID: {run_id}")


if __name__ == "__main__":
    MiBiciDataDownloader().run()
