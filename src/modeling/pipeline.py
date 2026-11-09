"""Construcción del pipeline de scikit-learn (preprocesamiento + modelo) y métricas de evaluación.

Cada modelo del proyecto es un único objeto `Pipeline`:

    Pipeline([
        ("preprocessor", ColumnTransformer(...)),   # imputación, escalado y TargetEncoder
        ("model", <estimador definido en config.yaml>),
    ])

Así el preprocesamiento se ajusta solo con los datos de entrenamiento en cada `fit`, y el
mismo objeto que se evalúa es el que se registra en MLflow y se usa para predecir.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import Config
from src.features import build_preprocessor


def build_estimator(model_cfg: Dict[str, Any], seed: int, params: Dict[str, Any] | None = None) -> BaseEstimator:
    """Instancia el estimador indicado en `class` con sus parámetros fijos + los ajustados."""
    module_name, class_name = model_cfg["class"].rsplit(".", 1)
    estimator_cls = getattr(importlib.import_module(module_name), class_name)
    kwargs = {**model_cfg.get("params", {}), **(params or {})}
    if "random_state" in estimator_cls().get_params():
        kwargs.setdefault("random_state", seed)
    return estimator_cls(**kwargs)


def build_pipeline(config: Config, model_name: str, params: Dict[str, Any] | None = None) -> Pipeline:
    """Pipeline completo para el modelo `model_name` de config.yaml."""
    model_cfg = config["models"][model_name]
    return Pipeline([
        ("preprocessor", build_preprocessor(config)),
        ("model", build_estimator(model_cfg, config["seed"], params)),
    ])


def suggest_params(trial: Any, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Traduce el `search_space` de config.yaml a sugerencias de un trial de Optuna."""
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                               log=bool(spec.get("log", False)))
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                             log=bool(spec.get("log", False)))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Tipo de hiperparámetro no soportado: {kind}")
    return params


def evaluate(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    """Métricas de regresión de conteos.

    - MAE: error medio en bicicletas por hora; es la métrica operativa (y la que decide).
    - RMSE: penaliza más los errores grandes (picos mal estimados).
    - R2: proporción de varianza explicada.
    - Devianza de Poisson: ajuste apropiado para conteos no negativos.
    """
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_poisson_deviance": float(mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))),
    }
