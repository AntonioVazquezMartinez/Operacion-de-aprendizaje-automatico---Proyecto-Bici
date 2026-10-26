"""Modelos baseline sobre cada versión preprocesada del dataset, registrados en MLflow."""

import os
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.mlflow_logger import MLflowDataVersioner

TARGET = "demand_count"
VERSIONS = ["scaled", "pca"]


class BaselineTrainer:
    """Entrena baselines por versión de datos y compara su desempeño en validación y prueba."""

    def __init__(self, processed_dir: str = "data/processed", experiment_name: str = "MiBici_Baselines_Fase1"):
        self.processed_dir = processed_dir
        self.versioner = MLflowDataVersioner(experiment_name=experiment_name)

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        }

    def run(self) -> None:
        print("=== Entrenamiento de Baselines (ML Engineer) ===")
        models = {
            "dummy_mean": lambda: DummyRegressor(strategy="mean"),
            "poisson": lambda: PoissonRegressor(alpha=1e-4, max_iter=300),
        }

        for version in VERSIONS:
            path = os.path.join(self.processed_dir, f"mibici_hourly_demand_{version}.csv")
            df = pd.read_csv(path)
            feature_cols = [c for c in df.columns if c not in (TARGET, "fecha_hora", "split")]
            parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
            data_hash = self.versioner.calculate_sha256(path)

            for model_name, factory in models.items():
                model = factory().fit(parts["train"][feature_cols], parts["train"][TARGET])
                metrics = {}
                for split in ["val", "test"]:
                    preds = model.predict(parts[split][feature_cols])
                    metrics.update(self.evaluate(parts[split][TARGET], preds, split))

                with mlflow.start_run(run_name=f"{model_name}_{version}"):
                    mlflow.log_params({
                        "model": model_name,
                        "dataset_version": version,
                        "dataset_sha256": data_hash,
                        "num_features": len(feature_cols),
                        "rows_train": len(parts["train"]),
                    })
                    mlflow.log_metrics(metrics)
                    mlflow.set_tag("role", "ML Engineer")
                print(f"{model_name:<11} | {version:<6} | val MAE {metrics['val_mae']:.4f} | test MAE {metrics['test_mae']:.4f}")


if __name__ == "__main__":
    BaselineTrainer().run()
