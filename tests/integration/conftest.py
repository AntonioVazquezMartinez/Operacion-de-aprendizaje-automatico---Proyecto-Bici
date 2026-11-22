"""Fixture de integración: corre el flujo completo una sola vez por sesión en un directorio temporal.

carga (CSV mensual con formato del portal) -> limpieza -> características -> entrenamiento con Optuna
-> comparación -> registro en un Model Registry temporal (SQLite en tmp). Sin red y sin `data/` real.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest

from src import dataset as dataset_module
from src import features as features_module
from src.modeling.train import ModelTrainer
from tests.conftest_data import make_isolated_config, make_raw_fixture, provision_tracking_db

E2E_MODELS = ["dummy_mean", "poisson", "hist_gradient_boosting"]
E2E_TRIALS = 2


def _no_network(*args, **kwargs):
    raise AssertionError("Las pruebas de integración no deben usar la red")


@pytest.fixture(scope="session")
def e2e(tmp_path_factory: pytest.TempPathFactory, mlflow_db_template: Path) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("integracion")
    config = provision_tracking_db(make_isolated_config(root), mlflow_db_template)
    config["data"]["months"] = ["2024-05"]
    config["data"]["trip_urls"] = {}

    # Archivos como los publica el portal: un CSV por mes y el catálogo con la columna `dpcapacity`.
    portal = root / "portal"
    make_raw_fixture(portal, demand_pattern=True)
    raw_dir = config.path("raw_dir")
    pd.read_csv(portal / "viajes_raw.csv").to_csv(raw_dir / "datos_abiertos_2024_05.csv", index=False)
    stations_name = config["data"]["stations_url"].rsplit("/", 1)[-1]
    (pd.read_csv(portal / "nomenclatura_estaciones.csv").rename(columns={"capacity": "dpcapacity"})
     .to_csv(raw_dir / stations_name, index=False))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dataset_module.requests, "get", _no_network)
        dataset_module.run(config, use_mlflow=True)
        features_module.run(config, use_mlflow=True)
        trainer = ModelTrainer(config, model_names=E2E_MODELS, n_trials=E2E_TRIALS)
        table = trainer.run()

    return SimpleNamespace(
        config=config,
        table=table,
        trainer=trainer,
        client=mlflow.MlflowClient(tracking_uri=config.tracking_uri),
        model_name=config["mlflow"]["registered_model_name"],
    )
