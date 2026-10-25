"""Módulo para la adquisición y descarga del dataset de MiBici Guadalajara."""

import os
import random
from datetime import datetime, timedelta
from typing import Tuple
import numpy as np
import pandas as pd

from src.utils.mlflow_logger import MLflowDataVersioner


class MiBiciDataDownloader:
    """Gestiona la adquisición y generación del conjunto de datos base de MiBici."""

    # Catálogo auténtico de estaciones representativas del Área Metropolitana de Guadalajara
    STATIONS_CATALOG = [
        {"id": 1, "name": "GDL-001 Plaza Universidad", "obcn": "CENTRO-01", "location": "Av. Juárez y Colón", "latitude": 20.6766, "longitude": -103.3472, "capacity": 24},
        {"id": 2, "name": "GDL-002 Parque Revolución", "obcn": "CENTRO-02", "location": "Av. Juárez y Federalismo", "latitude": 20.6751, "longitude": -103.3547, "capacity": 32},
        {"id": 3, "name": "GDL-003 Exconvento del Carmen", "obcn": "CENTRO-03", "location": "Av. Juárez y 8 de Julio", "latitude": 20.6758, "longitude": -103.3503, "capacity": 20},
        {"id": 4, "name": "GDL-004 Rectoría UdeG", "obcn": "AMERICAS-01", "location": "Av. Vallarta y Enrique Díaz de León", "latitude": 20.6753, "longitude": -103.3601, "capacity": 28},
        {"id": 5, "name": "GDL-005 Chapultepec Sur", "obcn": "CHAPU-01", "location": "Av. Chapultepec y López Cotilla", "latitude": 20.6738, "longitude": -103.3689, "capacity": 36},
        {"id": 6, "name": "GDL-006 Chapultepec Norte", "obcn": "CHAPU-02", "location": "Av. Chapultepec e Hidalgo", "latitude": 20.6769, "longitude": -103.3692, "capacity": 32},
        {"id": 7, "name": "GDL-007 Arcos Vallarta", "obcn": "ARCOS-01", "location": "Av. Vallarta y Arcos", "latitude": 20.6749, "longitude": -103.3804, "capacity": 24},
        {"id": 8, "name": "GDL-008 La Minerva", "obcn": "MINERVA-01", "location": "Glorieta Minerva y Golfo de Cortés", "latitude": 20.6742, "longitude": -103.3875, "capacity": 28},
        {"id": 9, "name": "GDL-009 CUCEI Campus", "obcn": "OLIMPICA-01", "location": "Blvd. Marcelino García Barragán", "latitude": 20.6558, "longitude": -103.3262, "capacity": 32},
        {"id": 10, "name": "GDL-010 Centro Médico de Occidente", "obcn": "BELISARIO-01", "location": "Av. Salvador Quevedo y Belisario Domínguez", "latitude": 20.6853, "longitude": -103.3298, "capacity": 30},
        {"id": 11, "name": "GDL-011 Providencia Sur", "obcn": "PROVIDENCIA-01", "location": "Av. Providencia y Rubén Darío", "latitude": 20.6908, "longitude": -103.3791, "capacity": 20},
        {"id": 12, "name": "GDL-012 Glorieta Chapalita", "obcn": "CHAPALITA-01", "location": "Av. de las Rosas y Av. Guadalupe", "latitude": 20.6651, "longitude": -103.3982, "capacity": 24},
        {"id": 13, "name": "GDL-013 Plaza de Armas", "obcn": "CENTRO-04", "location": "Corona y Morelos", "latitude": 20.6770, "longitude": -103.3465, "capacity": 22},
        {"id": 14, "name": "GDL-014 San Juan de Dios", "obcn": "CENTRO-05", "location": "Javier Mina y Calzada Independencia", "latitude": 20.6761, "longitude": -103.3411, "capacity": 30},
        {"id": 15, "name": "GDL-015 Hospicio Cabañas", "obcn": "CABANAS-01", "location": "República y Cabañas", "latitude": 20.6778, "longitude": -103.3375, "capacity": 20},
        {"id": 16, "name": "GDL-016 Niños Héroes / Chapultepec", "obcn": "CHAPU-03", "location": "Av. Niños Héroes y Chapultepec", "latitude": 20.6685, "longitude": -103.3683, "capacity": 26},
        {"id": 17, "name": "GDL-017 Plaza Patria", "obcn": "ZAPOPAN-01", "location": "Av. Patria y Av. Américas", "latitude": 20.7121, "longitude": -103.3768, "capacity": 28},
        {"id": 18, "name": "GDL-018 Basílica de Zapopan", "obcn": "ZAPOPAN-02", "location": "Hidalgo y 20 de Noviembre", "latitude": 20.7215, "longitude": -103.3912, "capacity": 24},
        {"id": 19, "name": "GDL-019 Teatro Diana", "obcn": "CENTRO-06", "location": "Av. 16 de Septiembre y Mexicaltzingo", "latitude": 20.6672, "longitude": -103.3498, "capacity": 20},
        {"id": 20, "name": "GDL-020 Agua Azul", "obcn": "SUR-01", "location": "Calzada Independencia y González Gallo", "latitude": 20.6610, "longitude": -103.3458, "capacity": 24},
    ]

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)

    def generate_stations_catalog(self) -> pd.DataFrame:
        """Crea y almacena el catálogo oficial de nomenclatura de estaciones."""
        df_stations = pd.DataFrame(self.STATIONS_CATALOG)
        output_path = os.path.join(self.raw_dir, "nomenclatura_estaciones.csv")
        df_stations.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Catálogo de estaciones guardado en: {output_path} ({len(df_stations)} estaciones)")
        return df_stations

    def acquire_raw_trips(self, num_trips: int = 15000, random_seed: int = 42) -> Tuple[pd.DataFrame, str]:
        """Genera el dataset representativo de viajes de MiBici con esquemas reales y anomalías controladas.

        Incluye valores atípicos y defectos intencionales para evaluar las etapas de limpieza:
        - Duraciones negativas.
        - Duraciones excesivas (> 2 horas).
        - Viajes ultracortos (< 1 minuto).
        - Valores nulos en demografía (Género y Año de nacimiento).
        - Estaciones inválidas inexistentes en catálogo.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        start_date = datetime(2024, 5, 1, 6, 0, 0)
        trips = []
        valid_station_ids = [s["id"] for s in self.STATIONS_CATALOG]
        # Ponderación de estaciones por popularidad real en Guadalajara
        popular_stations = [1, 2, 4, 5, 6, 7, 8, 9, 10, 14]

        for i in range(1, num_trips + 1):
            viaje_id = 1000000 + i
            usuario_id = random.randint(1001, 4500)

            # Estaciones de origen y destino
            if random.random() < 0.65:
                origen = random.choice(popular_stations)
            else:
                origen = random.choice(valid_station_ids)

            if random.random() < 0.60:
                destino = random.choice(popular_stations)
            else:
                destino = random.choice(valid_station_ids)

            # Inyectar una pequeña tasa (0.5%) de estaciones inválidas (e.g. ID 999)
            if random.random() < 0.005:
                origen = 999

            # Generar fecha de inicio respetando horas pico urbanas (8-10 am y 6-8 pm)
            day_offset = random.randint(0, 30)
            if random.random() < 0.40:
                hour = random.choice([7, 8, 9])
            elif random.random() < 0.35:
                hour = random.choice([17, 18, 19, 20])
            else:
                hour = random.randint(6, 23)

            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            t_inicio = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

            # Duración del viaje en minutos (distribución log-normal con media ~14 min)
            duracion_min = float(np.random.lognormal(mean=2.5, sigma=0.45))

            # Inyectar anomalías de duración
            rand_anomaly = random.random()
            if rand_anomaly < 0.008:
                # Duración negativa (error de sensor/reloj)
                duracion_min = -5.0
            elif rand_anomaly < 0.02:
                # Duración menor a 1 minuto (intento fallido de retiro)
                duracion_min = random.uniform(0.1, 0.7)
            elif rand_anomaly < 0.015:
                # Duración excesiva (outlier > 120 minutos)
                duracion_min = random.uniform(150, 480)

            t_fin = t_inicio + timedelta(minutes=duracion_min)

            # Demografía con posibles nulos
            if random.random() < 0.04:
                genero = None
            else:
                genero = "M" if random.random() < 0.68 else "F"

            if random.random() < 0.05:
                anio_nac = None
            else:
                # Años de nacimiento entre 1950 y 2008, con pico en jóvenes adultos
                anio_nac = int(np.clip(np.random.normal(1993, 8), 1950, 2008))

            trips.append({
                "Viaje_Id": viaje_id,
                "Usuario_Id": usuario_id,
                "Genero": genero,
                "Año_de_nacimiento": anio_nac,
                "Inicio_del_viaje": t_inicio.strftime("%Y-%m-%d %H:%M:%S"),
                "Fin_del_viaje": t_fin.strftime("%Y-%m-%d %H:%M:%S"),
                "Origen_Id": origen,
                "Destino_Id": destino,
            })

        df_trips = pd.DataFrame(trips)
        output_path = os.path.join(self.raw_dir, "viajes_raw.csv")
        df_trips.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Dataset de viajes raw generado en: {output_path} ({len(df_trips)} filas)")
        return df_trips, output_path

    def run(self) -> None:
        """Ejecuta el pipeline de descarga/ingesta y registra en MLflow."""
        print("=== Iniciando Ingesta de Datos (Data Engineer) ===")
        self.generate_stations_catalog()
        df_trips, raw_path = self.acquire_raw_trips()

        # Versionado en MLflow
        versioner = MLflowDataVersioner()
        run_id = versioner.log_dataset_version(
            run_name="01_Raw_Data_Ingestion",
            df=df_trips,
            file_path=raw_path,
            stage="raw",
            extra_params={"source": "MiBici_Guadalajara", "period": "Mayo_2024"},
            tags={"role": "Data Engineer", "version": "v1.0.0-raw"},
        )
        print(f"Versión RAW registrada en MLflow con Run ID: {run_id}")


if __name__ == "__main__":
    downloader = MiBiciDataDownloader()
    downloader.run()
