"""Fixtures de pytest compartidas por las pruebas de la Entrega Final.

Todo corre sobre el dataset sintético de tests/conftest_data.py en un directorio temporal:
no se requieren los datos reales de MiBici ni la base de MLflow del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.dataset import MiBiciDataCleaner
from src.features import MiBiciFeatureEngineer, model_input_columns, temporal_split
from src.modeling.pipeline import build_pipeline
from src.serving.bundle import export_bundle
from tests.conftest_data import make_raw_fixture, make_test_config


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("mibici")


@pytest.fixture(scope="session")
def config(workdir):
    cfg = make_test_config(workdir)
    cfg["serving"] = {"bundle_dir": str(workdir / "bundle"), "feature_store_hours": 240, "reference_rows": 200}
    cfg["monitoring"] = {"output_dir": str(workdir / "drift"), "reference_sample_rows": 5000}
    make_raw_fixture(cfg.path("raw_dir"))
    return cfg


@pytest.fixture(scope="session")
def features(config) -> pd.DataFrame:
    """Carga de datos -> limpieza -> matriz de características (se escribe en processed/)."""
    MiBiciDataCleaner(config).clean()
    df, _ = MiBiciFeatureEngineer(config).run()
    return df


@pytest.fixture(scope="session")
def splits(config, features):
    labels = temporal_split(features, config)
    return {s: features[labels == s].reset_index(drop=True) for s in ("train", "val", "test")}


@pytest.fixture(scope="session")
def trained_model(config, splits):
    """Pipeline HistGradientBoosting entrenado con train + val (como el champion)."""
    data = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    cols = model_input_columns(config)
    pipeline = build_pipeline(config, "hist_gradient_boosting", {"max_iter": 50})
    return pipeline.fit(data[cols], data[config["features"]["target"]])


@pytest.fixture(scope="session")
def bundle_dir(config, features, trained_model) -> Path:
    """Paquete de despliegue exportado con el modelo de prueba."""
    stations = pd.read_csv(config.file("raw_dir", "stations_file"))
    meta = {"registered_model_name": "mibici-test", "model_version": "1",
            "model_uri": "models:/mibici-test/1", "registry_metrics": {}}
    out = Path(config["serving"]["bundle_dir"])
    export_bundle(config, trained_model, features, stations, out, meta)
    return out
