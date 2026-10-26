"""Módulo de utilidades para versionado y trazabilidad de datos con MLflow."""

import hashlib
import os
from typing import Any, Dict, Optional
import mlflow
import pandas as pd


class MLflowDataVersioner:
    """Gestiona el registro de experimentos, versionado y artefactos en MLflow."""

    def __init__(self, db_path: str = "sqlite:///mlruns.db", experiment_name: str = "MiBici_MLOps_Fase1"):
        """Inicializa la conexión con el servidor/backend de MLflow.

        Args:
            db_path: URI de backend store (por defecto SQLite local).
            experiment_name: Nombre del experimento en MLflow.
        """
        self.tracking_uri = db_path
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def calculate_sha256(file_path: str) -> str:
        """Calcula el hash SHA-256 de un archivo para garantizar trazabilidad e inmutabilidad."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def log_dataset_version(
        self,
        run_name: str,
        df: pd.DataFrame,
        file_path: str,
        stage: str,
        extra_params: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Registra una versión de dataset en MLflow con metadatos, métricas y artefactos.

        Args:
            run_name: Nombre representativo de la corrida.
            df: DataFrame correspondiente a la versión.
            file_path: Ruta del archivo en disco.
            stage: Etapa del pipeline ('raw', 'interim', 'processed').
            extra_params: Parámetros adicionales a registrar.
            extra_metrics: Métricas cuantitativas (ej. conteos, memoria, porcentaje filtrado).
            tags: Etiquetas de contexto y gobernanza (ej. rol responsable, versión).

        Returns:
            run_id de la corrida registrada en MLflow.
        """
        file_hash = self.calculate_sha256(file_path)
        mem_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)

        with mlflow.start_run(run_name=run_name) as run:
            # Registrar parámetros base
            mlflow.log_param("dataset_stage", stage)
            mlflow.log_param("dataset_file", os.path.basename(file_path))
            mlflow.log_param("sha256_digest", file_hash)
            mlflow.log_param("num_rows", len(df))
            mlflow.log_param("num_columns", len(df.columns))
            mlflow.log_param("columns_list", ",".join(df.columns.tolist()[:15]))

            if extra_params:
                for k, v in extra_params.items():
                    mlflow.log_param(k, str(v))

            # Registrar métricas cuantitativas
            mlflow.log_metric("memory_usage_mb", round(mem_mb, 4))
            mlflow.log_metric("row_count", len(df))
            mlflow.log_metric("column_count", len(df.columns))

            if extra_metrics:
                for k, v in extra_metrics.items():
                    mlflow.log_metric(k, float(v))

            # Asignar tags de gobernanza
            mlflow.set_tag("framework", "Cookiecutter_MLOps")
            mlflow.set_tag("dataset_stage", stage)
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)

            mlflow.log_input(
                mlflow.data.from_pandas(df, source=file_path, name=os.path.basename(file_path)),
                context=stage,
            )

            # Subir el archivo como artefacto del run
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, artifact_path="datasets")

            return run.info.run_id
