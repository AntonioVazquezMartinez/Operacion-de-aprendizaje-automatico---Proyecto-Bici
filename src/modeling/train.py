"""Entrenamiento, ajuste de hiperparámetros, seguimiento en MLflow y registro de modelos.

Flujo por cada modelo habilitado en config.yaml:
    1. Optuna busca hiperparámetros entrenando en `train` y minimizando el MAE en `val`.
       Cada trial queda como corrida anidada en MLflow.
    2. Con los mejores hiperparámetros se reporta el desempeño en `val`.
    3. El pipeline se reentrena con `train + val` (los datos más recientes disponibles)
       y se evalúa una sola vez en `test`.
    4. Se registran parámetros, métricas, dataset, gráfica y el pipeline completo.
Al final se comparan los modelos, se registran en el Model Registry y el de menor MAE de
validación recibe el alias `champion` (el segundo, `challenger`).

Uso:
    python -m src.modeling.train
    python -m src.modeling.train --models poisson hist_gradient_boosting --n-trials 10
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn  # noqa: F401  (carga explícita del flavor de scikit-learn)
import optuna
import pandas as pd
import yaml
from mlflow.models import infer_signature

from src.config import Config, base_arg_parser, load_config
from src.features import load_features, model_input_columns, temporal_split
from src.modeling.pipeline import build_pipeline, evaluate, suggest_params
from src.plots import plot_hourly_profile, plot_model_comparison
from src.tracking import calculate_sha256, setup_mlflow


class ModelTrainer:
    """Orquesta el entrenamiento, la comparación y el registro de los modelos de demanda."""

    def __init__(self, config: Config, model_names: Optional[List[str]] = None, n_trials: Optional[int] = None,
                 register: bool = True):
        self.config = config
        self.target = config["features"]["target"]
        self.features = model_input_columns(config)
        self.tuning = config["tuning"]
        self.metric = self.tuning.get("metric", "val_mae")
        self.n_trials_override = n_trials
        self.register = register
        enabled = [name for name, cfg in config["models"].items() if cfg.get("enabled", True)]
        self.model_names = model_names or enabled
        unknown = set(self.model_names) - set(config["models"])
        if unknown:
            raise ValueError(f"Modelos no definidos en config.yaml: {sorted(unknown)}")

    # ------------------------------------------------------------------ datos
    def load_data(self) -> Dict[str, pd.DataFrame]:
        df = load_features(self.config)
        df["split"] = temporal_split(df, self.config)
        parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
        parts["train_val"] = pd.concat([parts["train"], parts["val"]], ignore_index=True)
        self.data_path = self.config.file("processed_dir", "features_file")
        self.data_info = {
            "dataset_file": self.data_path.name,
            "dataset_sha256": calculate_sha256(self.data_path),
            "rows_train": len(parts["train"]),
            "rows_val": len(parts["val"]),
            "rows_test": len(parts["test"]),
            "period_start": str(df["fecha_hora"].min()),
            "period_end": str(df["fecha_hora"].max()),
            "test_start": str(parts["test"]["fecha_hora"].min()),
            "train_val_end": str(parts["train_val"]["fecha_hora"].max()),
        }
        self.dataset_input = mlflow.data.from_pandas(
            parts["train_val"].head(1000), source=str(self.data_path), name=self.data_path.name,
            targets=self.target,
        )
        return parts

    def _xy(self, df: pd.DataFrame):
        return df[self.features], df[self.target]

    # ------------------------------------------------------------------ ajuste
    def tune(self, name: str, parts: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Búsqueda de hiperparámetros con Optuna (TPE). Regresa los mejores parámetros."""
        model_cfg = self.config["models"][name]
        search_space = model_cfg.get("search_space")
        if not search_space:
            return {}

        n_trials = self.n_trials_override or model_cfg.get("n_trials") or self.tuning["n_trials"]
        train = parts["train"]
        sample_rows = self.tuning.get("train_sample_rows")
        if sample_rows and len(train) > sample_rows:
            train = train.sample(n=int(sample_rows), random_state=self.config["seed"])
        X_train, y_train = self._xy(train)
        X_val, y_val = self._xy(parts["val"])

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial, search_space)
            with mlflow.start_run(run_name=f"{name}_trial_{trial.number:03d}", nested=True):
                pipeline = build_pipeline(self.config, name, params).fit(X_train, y_train)
                metrics = evaluate(y_val, pipeline.predict(X_val), "val")
                mlflow.log_params({"model": name, **params})
                mlflow.log_metrics(metrics)
                mlflow.set_tags({"trial_number": str(trial.number), "stage": "tuning"})
            return metrics[self.metric]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{name}_tuning",
            sampler=optuna.samplers.TPESampler(seed=self.config["seed"]),
        )
        study.optimize(objective, n_trials=int(n_trials), timeout=self.tuning.get("timeout_seconds"))
        mlflow.log_metrics({"tuning_best_val": float(study.best_value), "tuning_trials": float(len(study.trials))})
        print(f"  {name}: {len(study.trials)} trials, mejor {self.metric} = {study.best_value:.4f}")
        return dict(study.best_params)

    # ------------------------------------------------------------------ entrenamiento
    def train_model(self, name: str, parts: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        model_cfg = self.config["models"][name]
        print(f"=== Modelo: {name} ===")
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"model": name, "role": "Data Scientist / ML Engineer", "phase": "Fase 2",
                             "estimator_class": model_cfg["class"]})
            mlflow.log_input(self.dataset_input, context="training")

            best_params = self.tune(name, parts)

            # Desempeño en validación con los mejores hiperparámetros (entrenado solo con train).
            X_train, y_train = self._xy(parts["train"])
            X_val, y_val = self._xy(parts["val"])
            val_metrics = evaluate(y_val, build_pipeline(self.config, name, best_params)
                                   .fit(X_train, y_train).predict(X_val), "val")

            # Modelo final: train + val, evaluado una sola vez en test.
            X_tv, y_tv = self._xy(parts["train_val"])
            X_test, y_test = self._xy(parts["test"])
            final_pipeline = build_pipeline(self.config, name, best_params).fit(X_tv, y_tv)
            y_pred = final_pipeline.predict(X_test)
            test_metrics = evaluate(y_test, y_pred, "test")

            all_params = {**model_cfg.get("params", {}), **best_params}
            mlflow.log_params({
                "model": name,
                "estimator_class": model_cfg["class"],
                "features": ",".join(self.features),
                "seed": self.config["seed"],
                **self.data_info,
                **{f"hp_{k}": v for k, v in all_params.items()},
            })
            mlflow.log_metrics({**val_metrics, **test_metrics})

            with tempfile.TemporaryDirectory() as tmp:
                fig = plot_hourly_profile(parts["test"]["fecha_hora"], y_test.values, y_pred,
                                          Path(tmp) / f"perfil_horario_{name}.png",
                                          title=f"{name}: demanda real vs. predicha (test)")
                mlflow.log_artifact(str(fig), artifact_path="plots")

            input_example = X_test.head(5)
            model_info = mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                name="model",
                signature=infer_signature(input_example, final_pipeline.predict(input_example)),
                input_example=input_example,
                serialization_format="cloudpickle",
            )

        print(f"  val MAE {val_metrics['val_mae']:.4f} | test MAE {test_metrics['test_mae']:.4f} | "
              f"test RMSE {test_metrics['test_rmse']:.4f}")
        return {
            "model": name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "register": bool(model_cfg.get("register", True)),
            "params": all_params,
            **val_metrics,
            **test_metrics,
        }

    # ------------------------------------------------------------------ comparación y registro
    def log_comparison(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        table = pd.DataFrame(results).drop(columns=["params", "register", "model_uri"]).sort_values(self.metric)
        reports_dir = self.config.path("reports_dir")
        csv_path = reports_dir / "model_comparison.csv"
        table.to_csv(csv_path, index=False)
        fig_path = plot_model_comparison(table, self.config.path("figures_dir") / "06_comparacion_modelos.png")
        with mlflow.start_run(run_name="00_Comparativa_Modelos"):
            mlflow.set_tags({"phase": "Fase 2", "stage": "comparison", "best_model": table.iloc[0]["model"]})
            mlflow.log_artifact(str(csv_path))
            mlflow.log_artifact(str(fig_path), artifact_path="plots")
            with tempfile.TemporaryDirectory() as tmp:
                config_snapshot = Path(tmp) / "config.yaml"
                config_snapshot.write_text(yaml.safe_dump(dict(self.config), allow_unicode=True, sort_keys=False),
                                           encoding="utf-8")
                mlflow.log_artifact(str(config_snapshot))
        print("\n" + table[["model", "val_mae", "test_mae", "test_rmse", "test_r2"]].to_string(index=False))
        return table

    def register_models(self, results: List[Dict[str, Any]]) -> None:
        """Registra los modelos en el Model Registry con versión, hiperparámetros y métricas."""
        reg = self.config["mlflow"]
        name = reg["registered_model_name"]
        client = mlflow.MlflowClient()
        candidates = sorted((r for r in results if r["register"]), key=lambda r: r[self.metric])
        if not candidates:
            return

        for result in candidates:
            version = mlflow.register_model(result["model_uri"], name).version
            result["version"] = version
            tags = {
                "algorithm": result["model"],
                "hyperparameters": json.dumps(result["params"], default=str),
                "dataset_sha256": self.data_info["dataset_sha256"],
                "training_run_id": result["run_id"],
                **{k: f"{result[k]:.4f}" for k in ("val_mae", "test_mae", "test_rmse", "test_r2")},
            }
            for key, value in tags.items():
                client.set_model_version_tag(name, version, key, value)
            client.update_model_version(
                name, version,
                description=(f"{result['model']} entrenado con {self.data_info['period_start']} a "
                             f"{self.data_info['train_val_end']} (train + val); evaluado en test del "
                             f"{self.data_info['test_start']} al {self.data_info['period_end']}. "
                             f"val MAE={result['val_mae']:.4f}, test MAE={result['test_mae']:.4f}, "
                             f"test RMSE={result['test_rmse']:.4f}."),
            )
            print(f"Registrado {name} v{version} ({result['model']})")

        client.update_registered_model(
            name, description="Predicción de extracciones por estación y hora (MiBici Guadalajara), Equipo 19.")
        client.set_registered_model_alias(name, reg["champion_alias"], candidates[0]["version"])
        print(f"Alias '{reg['champion_alias']}' -> v{candidates[0]['version']} ({candidates[0]['model']})")
        if len(candidates) > 1:
            client.set_registered_model_alias(name, reg["challenger_alias"], candidates[1]["version"])
            print(f"Alias '{reg['challenger_alias']}' -> v{candidates[1]['version']} ({candidates[1]['model']})")

    def run(self) -> pd.DataFrame:
        setup_mlflow(self.config, self.config["mlflow"]["training_experiment"])
        parts = self.load_data()
        print(f"Datos: train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])} filas")
        results = [self.train_model(name, parts) for name in self.model_names]
        table = self.log_comparison(results)
        if self.register:
            self.register_models(results)
        return table


def main() -> None:
    parser = base_arg_parser("Entrenamiento y registro de modelos de demanda MiBici")
    parser.add_argument("--models", nargs="+", help="Subconjunto de modelos de config.yaml a entrenar")
    parser.add_argument("--n-trials", type=int, help="Trials de Optuna por modelo (sobrescribe config.yaml)")
    parser.add_argument("--no-register", action="store_true", help="No registrar en el Model Registry")
    args = parser.parse_args()
    trainer = ModelTrainer(load_config(args.config), args.models, args.n_trials, register=not args.no_register)
    trainer.run()


if __name__ == "__main__":
    main()
