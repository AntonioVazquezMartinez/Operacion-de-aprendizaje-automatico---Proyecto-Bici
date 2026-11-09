"""Pruebas de datos y características sobre un dataset sintético en un directorio temporal.

Ejecutar desde la raíz del repositorio:  python -m unittest discover -s tests -v
"""

import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import MiBiciDataCleaner
from src.features import (
    MiBiciFeatureEngineer,
    build_preprocessor,
    feature_columns,
    mexican_holidays,
    model_input_columns,
    temporal_split,
)
from tests.conftest_data import make_raw_fixture, make_test_config


class TestDataAndFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.config = make_test_config(cls.tmp)
        make_raw_fixture(cls.config.path("raw_dir"))

        cleaner = MiBiciDataCleaner(cls.config)
        cls.df_clean, _ = cleaner.clean()
        cls.metrics = cleaner.metrics
        cls.df_stations = pd.read_csv(cls.config.file("raw_dir", "stations_file"))

        cls.engineer = MiBiciFeatureEngineer(cls.config)
        cls.grid = cls.engineer.build_demand_grid(cls.df_clean)
        cls.df_features, _ = cls.engineer.run()
        cls.split = temporal_split(cls.df_features, cls.config)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- limpieza
    def test_cleaner_removes_injected_anomalies(self):
        self.assertEqual(self.metrics["dropped_duplicates"], 2)
        self.assertEqual(self.metrics["dropped_negative_durations"], 10)
        self.assertEqual(self.metrics["dropped_short_trips"], 10)
        self.assertEqual(self.metrics["dropped_excessive_durations"], 10)
        self.assertEqual(self.metrics["dropped_invalid_stations"], 5)
        self.assertEqual(self.metrics["dropped_invalid_ages"], 5)

    def test_cleaner_rules_come_from_config(self):
        rules = self.config["cleaning"]
        self.assertTrue((self.df_clean["duracion_segundos"] >= rules["min_duration_seconds"]).all())
        self.assertTrue((self.df_clean["duracion_minutos"] <= rules["max_duration_minutes"]).all())

    def test_cleaner_valid_stations(self):
        valid_ids = set(self.df_stations["id"])
        self.assertTrue(self.df_clean["Origen_Id"].isin(valid_ids).all())
        self.assertTrue(self.df_clean["Destino_Id"].isin(valid_ids).all())

    # ---------------------------------------------------------------- características
    def test_demand_grid_preserves_trips(self):
        self.assertEqual(self.grid["demand_count"].sum(), len(self.df_clean))
        self.assertTrue((self.grid["demand_count"] >= 0).all())

    def test_model_inputs_exist_in_features(self):
        missing = set(model_input_columns(self.config)) - set(self.df_features.columns)
        self.assertFalse(missing)

    def test_no_short_lags_leak_future_horizon(self):
        """Ningún rezago menor a 24 h (el horizonte máximo del Canvas) entra al modelo."""
        self.assertNotIn("demand_lag_1h", model_input_columns(self.config))
        self.assertTrue(all(h >= 24 for h in self.config["features"]["lag_hours"]))

    def test_lag_values_match_past_demand(self):
        station = self.df_features["Origen_Id"].iloc[0]
        grid = self.grid[self.grid["Origen_Id"] == station].set_index("fecha_hora")["demand_count"]
        row = self.df_features[self.df_features["Origen_Id"] == station].iloc[10]
        expected = grid.loc[row["fecha_hora"] - pd.Timedelta(hours=24)]
        self.assertEqual(row["demand_lag_24h"], expected)

    def test_incomplete_history_dropped(self):
        self.assertFalse(self.df_features[feature_columns(self.config)["numeric"]].isna().any().any())
        first_hour = self.grid["fecha_hora"].min()
        max_lag = max(self.config["features"]["lag_hours"])
        self.assertGreaterEqual(self.df_features["fecha_hora"].min(), first_hour + pd.Timedelta(hours=max_lag))

    def test_cyclical_bounds(self):
        for col in ["sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek"]:
            self.assertTrue(self.df_features[col].between(-1.0 - 1e-9, 1.0 + 1e-9).all())

    def test_mexican_holidays(self):
        holidays_2024 = mexican_holidays(2024)
        self.assertIn(date(2024, 5, 1), holidays_2024)
        self.assertIn(date(2024, 2, 5), holidays_2024)    # primer lunes de febrero
        self.assertIn(date(2024, 11, 18), holidays_2024)  # tercer lunes de noviembre
        self.assertIn(date(2024, 10, 1), holidays_2024)   # cambio de gobierno
        may_first = self.df_features[self.df_features["fecha_hora"].dt.date == date(2024, 5, 1)]
        if len(may_first):
            self.assertTrue((may_first["is_holiday"] == 1).all())

    # ---------------------------------------------------------------- partición y preprocesamiento
    def test_temporal_split_is_chronological(self):
        times = self.df_features["fecha_hora"]
        self.assertLess(times[self.split == "train"].max(), times[self.split == "val"].min())
        self.assertLess(times[self.split == "val"].max(), times[self.split == "test"].min())

    def test_preprocessor_fit_only_on_train(self):
        train = self.df_features[self.split == "train"]
        test = self.df_features[self.split == "test"]
        cols = model_input_columns(self.config)
        pre = build_preprocessor(self.config).fit(train[cols], train["demand_count"])
        names = list(pre.get_feature_names_out())
        lag_idx = names.index("demand_lag_24h")
        train_t = pre.transform(train[cols])
        self.assertAlmostEqual(float(np.mean(train_t[:, lag_idx])), 0.0, places=6)
        self.assertEqual(pre.transform(test[cols]).shape[1], len(names))


if __name__ == "__main__":
    unittest.main()
