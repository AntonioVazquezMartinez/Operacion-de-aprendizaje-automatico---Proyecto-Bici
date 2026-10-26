"""Preprocesamiento del dataset horario: partición temporal, escalado, codificación y PCA."""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from src.utils.mlflow_logger import MLflowDataVersioner

TARGET = "demand_count"
NUMERIC_FEATURES = [
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
BINARY_FEATURES = ["is_weekend"]
CATEGORICAL_FEATURES = ["Origen_Id"]


class MiBiciPreprocessor:
    """Genera las versiones escalada y reducida (PCA) del dataset con partición cronológica."""

    def __init__(
        self,
        features_path: str = "data/processed/mibici_hourly_demand.csv",
        output_dir: str = "data/processed",
        train_size: float = 0.70,
        val_size: float = 0.15,
        pca_variance: float = 0.95,
        random_state: int = 42,
    ):
        self.features_path = features_path
        self.output_dir = output_dir
        self.train_size = train_size
        self.val_size = val_size
        self.pca_variance = pca_variance
        self.random_state = random_state
        os.makedirs(self.output_dir, exist_ok=True)

    def temporal_split(self, df: pd.DataFrame) -> pd.Series:
        """Asigna train/val/test por bloques de tiempo para evitar fuga de información futura."""
        hours = np.sort(df["fecha_hora"].unique())
        train_end = hours[int(len(hours) * self.train_size) - 1]
        val_end = hours[int(len(hours) * (self.train_size + self.val_size)) - 1]
        return pd.Series(
            np.select([df["fecha_hora"] <= train_end, df["fecha_hora"] <= val_end], ["train", "val"], "test"),
            index=df.index,
        )

    def build_transformer(self, use_pca: bool) -> Pipeline:
        columns = ColumnTransformer(
            [
                ("num", StandardScaler(), NUMERIC_FEATURES),
                # Alta cardinalidad (cientos de estaciones): target encoding en lugar de one-hot.
                ("station", Pipeline([
                    ("encoder", TargetEncoder(target_type="continuous", random_state=self.random_state)),
                    ("scaler", StandardScaler()),
                ]), CATEGORICAL_FEATURES),
                ("bin", "passthrough", BINARY_FEATURES),
            ],
            verbose_feature_names_out=False,
        )
        steps = [("columns", columns)]
        if use_pca:
            steps.append(("pca", PCA(n_components=self.pca_variance, random_state=self.random_state)))
        return Pipeline(steps)

    def transform_version(self, df: pd.DataFrame, split: pd.Series, use_pca: bool) -> Tuple[pd.DataFrame, Dict[str, float]]:
        transformer = self.build_transformer(use_pca)
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
        train_mask = split == "train"

        # fit_transform en train aplica cross-fitting del TargetEncoder y evita fuga del objetivo.
        train_matrix = transformer.fit_transform(df.loc[train_mask, features], df.loc[train_mask, TARGET])
        rest_matrix = transformer.transform(df.loc[~train_mask, features])

        if use_pca:
            columns = [f"pc_{i + 1}" for i in range(train_matrix.shape[1])]
        else:
            columns = list(transformer.get_feature_names_out())
        df_out = pd.concat([
            pd.DataFrame(train_matrix, columns=columns, index=df.index[train_mask]),
            pd.DataFrame(rest_matrix, columns=columns, index=df.index[~train_mask]),
        ]).loc[df.index]
        df_out[TARGET] = df[TARGET]
        df_out["fecha_hora"] = df["fecha_hora"]
        df_out["split"] = split

        metrics = {"num_features": float(len(columns))}
        if use_pca:
            pca = transformer.named_steps["pca"]
            metrics["pca_components"] = float(pca.n_components_)
            metrics["pca_explained_variance"] = round(float(pca.explained_variance_ratio_.sum()), 4)
        return df_out, metrics

    def run(self) -> None:
        print("=== Iniciando Preprocesamiento (Data Scientist / ML Engineer) ===")
        df = pd.read_csv(self.features_path, parse_dates=["fecha_hora"])
        split = self.temporal_split(df)
        split_counts = split.value_counts()

        versioner = MLflowDataVersioner()
        for version, use_pca, run_name in [
            ("scaled", False, "04_Processed_Scaled_Encoded"),
            ("pca", True, "05_Processed_PCA"),
        ]:
            df_out, metrics = self.transform_version(df, split, use_pca)
            output_path = os.path.join(self.output_dir, f"mibici_hourly_demand_{version}.csv")
            df_out.to_csv(output_path, index=False, encoding="utf-8")
            print(f"Versión '{version}' guardada en: {output_path} {df_out.shape}")

            metrics.update({f"rows_{s}": float(split_counts.get(s, 0)) for s in ["train", "val", "test"]})
            run_id = versioner.log_dataset_version(
                run_name=run_name,
                df=df_out,
                file_path=output_path,
                stage=f"processed_{version}",
                extra_params={
                    "split": f"temporal {self.train_size:.0%}/{self.val_size:.0%}/{1 - self.train_size - self.val_size:.0%}",
                    "scaler": "StandardScaler (fit en train)",
                    "station_encoding": "TargetEncoder (fit en train)",
                    "pca_variance": self.pca_variance if use_pca else "none",
                },
                extra_metrics=metrics,
                tags={"role": "Data Scientist / ML Engineer", "version": f"v1.3.0-{version}"},
            )
            print(f"Versión {version.upper()} registrada en MLflow con Run ID: {run_id}")


if __name__ == "__main__":
    MiBiciPreprocessor().run()
