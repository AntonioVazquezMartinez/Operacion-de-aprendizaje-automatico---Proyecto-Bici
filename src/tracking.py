"""Utilidades de trazabilidad con MLflow: conexión, hashes y versionado de datasets."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional

import mlflow
import pandas as pd

from src.config import Config


def setup_mlflow(config: Config, experiment_name: str) -> str:
    """Conecta MLflow a la base SQLite del proyecto y activa (o crea) el experimento.

    Los artefactos de experimentos nuevos se guardan en `mlflow.artifact_dir` dentro del
    repositorio, de modo que la ubicación no depende de la carpeta desde donde se ejecute.
    """
    mlflow.set_tracking_uri(config.tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name, artifact_location=config.artifact_location)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def calculate_sha256(file_path: str | os.PathLike) -> str:
    """Hash SHA-256 de un archivo para garantizar que una versión de datos no cambió."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()


class MLflowDataVersioner:
    """Registra cada versión de los datos (raw, interim, processed) como una corrida de MLflow."""

    def __init__(self, config: Config):
        self.config = config
        self.log_artifacts = bool(config["mlflow"].get("log_dataset_artifacts", False))
        setup_mlflow(config, config["mlflow"]["data_experiment"])

    def log_dataset_version(
        self,
        run_name: str,
        df: pd.DataFrame,
        file_path: str | os.PathLike,
        stage: str,
        extra_params: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Registra parámetros, métricas, hash y el dataset vinculado; regresa el run_id."""
        file_hash = calculate_sha256(file_path)
        mem_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                "dataset_stage": stage,
                "dataset_file": os.path.basename(file_path),
                "sha256_digest": file_hash,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                **{k: str(v) for k, v in (extra_params or {}).items()},
            })
            mlflow.log_metrics({
                "memory_usage_mb": round(mem_mb, 4),
                "row_count": float(len(df)),
                "column_count": float(len(df.columns)),
                **{k: float(v) for k, v in (extra_metrics or {}).items()},
            })
            mlflow.set_tags({"framework": "Cookiecutter_MLOps", "dataset_stage": stage, **(tags or {})})
            mlflow.log_input(
                mlflow.data.from_pandas(df, source=str(file_path), name=os.path.basename(file_path)),
                context=stage,
            )
            if self.log_artifacts:
                mlflow.log_artifact(str(file_path), artifact_path="datasets")
            return run.info.run_id
