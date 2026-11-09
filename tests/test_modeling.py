"""Pruebas del pipeline de modelado y prueba de integración con MLflow + Optuna.

La prueba de integración entrena, registra en un Model Registry temporal y vuelve a cargar
el modelo `champion` para predecir. Se omite si mlflow u optuna no están instalados.
"""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.dataset import MiBiciDataCleaner
from src.features import MiBiciFeatureEngineer, model_input_columns, temporal_split
from src.modeling.pipeline import build_estimator, build_pipeline, evaluate, suggest_params
from tests.conftest_data import make_raw_fixture, make_test_config

HAS_TRACKING = all(importlib.util.find_spec(m) for m in ("mlflow", "optuna"))


class FakeTrial:
    """Imita un trial de Optuna devolviendo el límite inferior de cada rango."""

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_int(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


class TestModelingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.config = make_test_config(cls.tmp)
        make_raw_fixture(cls.config.path("raw_dir"))
        MiBiciDataCleaner(cls.config).clean()
        cls.df, _ = MiBiciFeatureEngineer(cls.config).run()
        split = temporal_split(cls.df, cls.config)
        cols = model_input_columns(cls.config)
        cls.X_train, cls.y_train = cls.df.loc[split == "train", cols], cls.df.loc[split == "train", "demand_count"]
        cls.X_test, cls.y_test = cls.df.loc[split == "test", cols], cls.df.loc[split == "test", "demand_count"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_configured_model_trains_and_predicts(self):
        for name, model_cfg in self.config["models"].items():
            params = suggest_params(FakeTrial(), model_cfg.get("search_space", {}))
            pipeline = build_pipeline(self.config, name, params)
            self.assertIsInstance(pipeline, Pipeline)
            self.assertEqual([s for s, _ in pipeline.steps], ["preprocessor", "model"])
            preds = pipeline.fit(self.X_train, self.y_train).predict(self.X_test)
            self.assertEqual(len(preds), len(self.X_test), name)
            self.assertTrue(np.isfinite(preds).all(), name)

    def test_seed_is_injected(self):
        rf = build_estimator(self.config["models"]["random_forest"], seed=123)
        self.assertEqual(rf.get_params()["random_state"], 123)

    def test_training_is_reproducible(self):
        preds = [
            build_pipeline(self.config, "hist_gradient_boosting").fit(self.X_train, self.y_train).predict(self.X_test)
            for _ in range(2)
        ]
        np.testing.assert_allclose(preds[0], preds[1])

    def test_evaluate_metrics(self):
        metrics = evaluate(pd.Series([0, 1, 2]), np.array([0.0, 1.0, 2.0]), "val")
        self.assertAlmostEqual(metrics["val_mae"], 0.0)
        self.assertAlmostEqual(metrics["val_r2"], 1.0)
        self.assertEqual(set(metrics), {"val_mae", "val_rmse", "val_r2", "val_poisson_deviance"})

    @unittest.skipUnless(HAS_TRACKING, "mlflow y optuna no están instalados")
    def test_train_register_and_predict_champion(self):
        from src.modeling.predict import predict
        from src.modeling.train import ModelTrainer

        trainer = ModelTrainer(self.config, model_names=["dummy_mean", "poisson", "hist_gradient_boosting"],
                               n_trials=2)
        table = trainer.run()
        self.assertEqual(len(table), 3)

        import mlflow

        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self.config["mlflow"]["training_experiment"])
        comparison = client.search_runs([experiment.experiment_id], "tags.stage = 'comparison'")
        self.assertEqual(len(comparison), 1)
        artifacts = {a.path for a in client.list_artifacts(comparison[0].info.run_id)}
        self.assertIn("config.yaml", artifacts)

        name = self.config["mlflow"]["registered_model_name"]
        champion = client.get_model_version_by_alias(name, "champion")
        self.assertIn("val_mae", champion.tags)
        self.assertIn("hyperparameters", champion.tags)

        out = predict(self.config)
        self.assertIn("demand_pred", out.columns)
        self.assertTrue((out["demand_pred"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
