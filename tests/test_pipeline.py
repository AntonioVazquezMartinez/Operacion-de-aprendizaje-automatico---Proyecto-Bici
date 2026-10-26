"""Pruebas del pipeline de MiBici sobre un dataset pequeño y controlado en un directorio temporal."""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.cleaner import MiBiciDataCleaner
from src.features.engineer import MiBiciFeatureEngineer
from src.features.preprocessor import MiBiciPreprocessor


def make_raw_fixture(raw_dir: str, n_trips: int = 2000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    stations = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": [f"(GDL-00{i}) Estación {i}" for i in range(1, 5)],
        "latitude": [20.67, 20.68, 20.66, 20.69],
        "longitude": [-103.35, -103.36, -103.37, -103.34],
        "capacity": [15, 20, 18, 12],
    })
    start = pd.Timestamp("2024-05-01") + pd.to_timedelta(rng.integers(0, 14 * 24 * 3600, n_trips), unit="s")
    duration = pd.to_timedelta(rng.lognormal(6.3, 0.5, n_trips), unit="s")
    trips = pd.DataFrame({
        "Viaje_Id": np.arange(n_trips),
        "Usuario_Id": rng.integers(1, 500, n_trips),
        "Genero": rng.choice(["M", "F", None], n_trips),
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

    trips.to_csv(os.path.join(raw_dir, "viajes_raw.csv"), index=False)
    stations.to_csv(os.path.join(raw_dir, "nomenclatura_estaciones.csv"), index=False)


class TestMiBiciPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        raw_dir = os.path.join(cls.tmp, "raw")
        os.makedirs(raw_dir)
        make_raw_fixture(raw_dir)
        stations_path = os.path.join(raw_dir, "nomenclatura_estaciones.csv")

        cleaner = MiBiciDataCleaner(
            raw_trips_path=os.path.join(raw_dir, "viajes_raw.csv"),
            raw_stations_path=stations_path,
            output_dir=os.path.join(cls.tmp, "interim"),
        )
        cls.df_clean, clean_path = cleaner.clean()
        cls.metrics = cleaner.metrics

        engineer = MiBiciFeatureEngineer(
            cleaned_trips_path=clean_path,
            stations_path=stations_path,
            output_dir=os.path.join(cls.tmp, "processed"),
            figures_dir=os.path.join(cls.tmp, "figures"),
        )
        cls.df_features, _ = engineer.build_features()
        cls.df_stations = pd.read_csv(stations_path)

        cls.preprocessor = MiBiciPreprocessor(output_dir=os.path.join(cls.tmp, "processed"))
        cls.split = cls.preprocessor.temporal_split(cls.df_features)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cleaner_removes_injected_anomalies(self):
        self.assertEqual(self.metrics["dropped_duplicates"], 2)
        self.assertEqual(self.metrics["dropped_negative_durations"], 10)
        self.assertEqual(self.metrics["dropped_short_trips"], 10)
        self.assertEqual(self.metrics["dropped_excessive_durations"], 10)
        self.assertEqual(self.metrics["dropped_invalid_stations"], 5)
        self.assertEqual(self.metrics["dropped_invalid_ages"], 5)

    def test_cleaner_duration_bounds(self):
        self.assertTrue((self.df_clean["duracion_segundos"] >= 60).all())
        self.assertTrue((self.df_clean["duracion_minutos"] <= 120.0).all())

    def test_cleaner_valid_stations(self):
        valid_ids = set(self.df_stations["id"])
        self.assertTrue(self.df_clean["Origen_Id"].isin(valid_ids).all())
        self.assertTrue(self.df_clean["Destino_Id"].isin(valid_ids).all())

    def test_feature_engineer_columns(self):
        required_cols = [
            "Origen_Id", "fecha_hora", "demand_count", "hour", "day_of_week", "is_weekend", "month",
            "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek",
            "latitude", "longitude", "capacity", "demand_lag_1h", "demand_lag_24h",
        ]
        self.assertEqual(sorted(self.df_features.columns), sorted(required_cols))

    def test_demand_grid_preserves_trips(self):
        self.assertEqual(self.df_features["demand_count"].sum(), len(self.df_clean))
        self.assertTrue((self.df_features["demand_count"] >= 0).all())

    def test_cyclical_bounds(self):
        for col in ["sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"]:
            self.assertTrue(self.df_features[col].between(-1.0 - 1e-6, 1.0 + 1e-6).all())

    def test_temporal_split_is_chronological(self):
        times = self.df_features["fecha_hora"]
        self.assertLess(times[self.split == "train"].max(), times[self.split == "val"].min())
        self.assertLess(times[self.split == "val"].max(), times[self.split == "test"].min())

    def test_scaler_fit_only_on_train(self):
        df_out, _ = self.preprocessor.transform_version(self.df_features, self.split, use_pca=False)
        train = df_out[df_out["split"] == "train"]
        self.assertAlmostEqual(train["demand_lag_1h"].mean(), 0.0, places=6)

    def test_pca_reaches_target_variance(self):
        df_out, metrics = self.preprocessor.transform_version(self.df_features, self.split, use_pca=True)
        self.assertGreaterEqual(metrics["pca_explained_variance"], self.preprocessor.pca_variance)
        self.assertEqual(len(df_out), len(self.df_features))


if __name__ == "__main__":
    unittest.main()
