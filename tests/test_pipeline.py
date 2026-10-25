"""Pruebas unitarias automatizadas para el pipeline de MLOps de MiBici (unittest / pytest compatible)."""

import os
import unittest
import pandas as pd

from src.data.cleaner import MiBiciDataCleaner
from src.data.downloader import MiBiciDataDownloader
from src.features.engineer import MiBiciFeatureEngineer


class TestMiBiciPipeline(unittest.TestCase):
    """Suite de validación de calidad de datos, transformaciones y contratos de features."""

    @classmethod
    def setUpClass(cls):
        """Prepara los datos ejecutando el pipeline en modo reducido para testing."""
        cls.raw_dir = "data/raw"
        cls.interim_dir = "data/interim"
        cls.processed_dir = "data/processed"

        # Verificar y asegurar que los archivos existan
        if not os.path.exists(os.path.join(cls.raw_dir, "viajes_raw.csv")):
            downloader = MiBiciDataDownloader(raw_dir=cls.raw_dir)
            downloader.generate_stations_catalog()
            downloader.acquire_raw_trips(num_trips=1500, random_seed=42)

        cleaner = MiBiciDataCleaner(
            raw_trips_path=os.path.join(cls.raw_dir, "viajes_raw.csv"),
            raw_stations_path=os.path.join(cls.raw_dir, "nomenclatura_estaciones.csv"),
            output_dir=cls.interim_dir,
        )
        cls.df_clean, cls.clean_path = cleaner.clean()

        engineer = MiBiciFeatureEngineer(
            cleaned_trips_path=cls.clean_path,
            stations_path=os.path.join(cls.raw_dir, "nomenclatura_estaciones.csv"),
            output_dir=cls.processed_dir,
            figures_dir="reports/figures",
        )
        cls.df_features, cls.feat_path = engineer.build_features()

        cls.df_stations = pd.read_csv(os.path.join(cls.raw_dir, "nomenclatura_estaciones.csv"))

    def test_cleaner_no_negative_durations(self):
        """Verifica que el pipeline de limpieza elimine todas las duraciones negativas."""
        self.assertTrue(
            (self.df_clean["duracion_segundos"] > 0).all(),
            "Fallo: Existen registros con duración negativa o cero en los datos limpios.",
        )

    def test_cleaner_duration_bounds(self):
        """Verifica que no haya viajes menores a 1 minuto ni mayores a 120 minutos."""
        self.assertTrue(
            (self.df_clean["duracion_segundos"] >= 60).all(),
            "Fallo: Se encontraron viajes con duración inferior a 60 segundos.",
        )
        self.assertTrue(
            (self.df_clean["duracion_minutos"] <= 120.0).all(),
            "Fallo: Se encontraron viajes con duración superior a 120 minutos.",
        )

    def test_cleaner_valid_stations(self):
        """Verifica que todas las estaciones de origen y destino pertenezcan al catálogo oficial."""
        valid_ids = set(self.df_stations["id"].unique())
        self.assertTrue(
            self.df_clean["Origen_Id"].isin(valid_ids).all(),
            "Fallo: Existen viajes con estación de origen no registrada en el catálogo.",
        )
        self.assertTrue(
            self.df_clean["Destino_Id"].isin(valid_ids).all(),
            "Fallo: Existen viajes con estación de destino no registrada en el catálogo.",
        )

    def test_feature_engineer_columns(self):
        """Verifica la existencia de todas las variables requeridas para el modelo de regresión."""
        required_cols = [
            "Origen_Id",
            "fecha_hora",
            "demand_count",
            "hour",
            "day_of_week",
            "is_weekend",
            "sin_hour",
            "cos_hour",
            "sin_dayofweek",
            "cos_dayofweek",
            "latitude",
            "longitude",
            "capacity",
            "demand_lag_1h",
            "demand_lag_24h",
        ]
        for col in required_cols:
            self.assertIn(col, self.df_features.columns, f"Falta la variable '{col}' en el dataset procesado.")

    def test_cyclical_bounds(self):
        """Verifica que las transformaciones trigonométricas estén en el rango [-1, 1]."""
        for col in ["sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"]:
            self.assertTrue(
                (self.df_features[col] >= -1.0 - 1e-6).all(),
                f"Fallo en límite inferior para {col}.",
            )
            self.assertTrue(
                (self.df_features[col] <= 1.0 + 1e-6).all(),
                f"Fallo en límite superior para {col}.",
            )

    def test_demand_count_non_negative(self):
        """Verifica que la variable objetivo de demanda sea siempre mayor o igual a cero."""
        self.assertTrue(
            (self.df_features["demand_count"] >= 0).all(),
            "Fallo: La demanda predicha/agregada no puede ser negativa.",
        )


if __name__ == "__main__":
    unittest.main()
