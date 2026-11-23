"""Configuración (`src.config`), semillas globales y utilidades de trazabilidad (`src.tracking`)."""

from __future__ import annotations

import hashlib
import os
import random

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, PROJ_ROOT, Config, base_arg_parser, load_config, set_global_seeds
from src.tracking import MLflowDataVersioner, calculate_sha256, setup_mlflow


# --------------------------------------------------------------------------- load_config
def test_load_config_defaults_to_repo_file():
    config = load_config()
    assert isinstance(config, Config)
    assert DEFAULT_CONFIG_PATH == PROJ_ROOT / "config.yaml"
    assert config["seed"] == 42
    assert {"paths", "data", "features", "split", "mlflow", "models"} <= set(config)


def test_load_config_overrides_merge_sections():
    config = load_config(overrides={
        "split": {"train_size": 0.5},          # dict: se mezcla con la sección existente
        "seed": 7,                              # escalar: se reemplaza
        "nueva": {"clave": 1},                  # sección inexistente: se crea
    })
    assert config["split"] == {"train_size": 0.5, "val_size": 0.15}
    assert config["seed"] == 7
    assert config["nueva"] == {"clave": 1}
    assert load_config()["split"]["train_size"] == 0.70     # el archivo no cambia


def test_load_config_from_custom_file(tmp_path):
    path = tmp_path / "otra.yaml"
    path.write_text(yaml.safe_dump({"seed": 1, "paths": {"raw_dir": "x"}}), encoding="utf-8")
    config = load_config(path, overrides={"paths": {"interim_dir": "y"}})
    assert config == {"seed": 1, "paths": {"raw_dir": "x", "interim_dir": "y"}}


def test_config_path_absolute_is_created(tmp_path):
    target = tmp_path / "a" / "b"
    config = Config({"paths": {"raw_dir": str(target)}, "data": {"raw_trips_file": "viajes.csv"}})
    assert config.path("raw_dir") == target
    assert target.is_dir()
    assert config.file("raw_dir", "raw_trips_file") == target / "viajes.csv"


def test_config_path_relative_is_anchored_to_project_root():
    config = Config({"paths": {"models_dir": "models"}})     # carpeta que ya existe en el repositorio
    assert config.path("models_dir") == PROJ_ROOT / "models"


@pytest.mark.parametrize("uri, expected", [
    ("sqlite:///mlruns.db", "sqlite:///" + (PROJ_ROOT / "mlruns.db").as_posix()),
    ("http://localhost:5000", "http://localhost:5000"),
])
def test_tracking_uri(uri, expected):
    assert Config({"mlflow": {"tracking_uri": uri}}).tracking_uri == expected


def test_tracking_uri_absolute_sqlite_unchanged(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
    assert Config({"mlflow": {"tracking_uri": uri}}).tracking_uri == uri


def test_artifact_location(tmp_path):
    folder = tmp_path / "artefactos"
    location = Config({"mlflow": {"artifact_dir": str(folder)}}).artifact_location
    assert location == folder.as_uri()
    assert folder.is_dir()
    relative = Config({"mlflow": {"artifact_dir": "models"}}).artifact_location
    assert relative == (PROJ_ROOT / "models").as_uri()


def test_base_arg_parser():
    parser = base_arg_parser("prueba")
    assert parser.parse_args([]).config == str(DEFAULT_CONFIG_PATH)
    assert parser.parse_args(["--config", "otro.yaml"]).config == "otro.yaml"


# --------------------------------------------------------------------------- semillas
def test_set_global_seeds_is_reproducible():
    draws = []
    for _ in range(2):
        set_global_seeds(42)
        draws.append((random.random(), float(np.random.rand())))
    assert draws[0] == draws[1]
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_different_seeds_give_different_draws():
    set_global_seeds(1)
    a = np.random.rand(3)
    set_global_seeds("2")                      # acepta texto convertible a entero
    assert not np.array_equal(a, np.random.rand(3))
    assert os.environ["PYTHONHASHSEED"] == "2"


# --------------------------------------------------------------------------- hash
@pytest.mark.parametrize("content, expected", [
    (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
])
def test_calculate_sha256_known_values(tmp_path, content, expected):
    path = tmp_path / "archivo.bin"
    path.write_bytes(content)
    assert calculate_sha256(path) == expected
    assert calculate_sha256(str(path)) == expected


def test_calculate_sha256_large_file_in_blocks(tmp_path):
    data = os.urandom(65536 * 3 + 17)          # varios bloques de 64 KiB y un resto
    path = tmp_path / "grande.bin"
    path.write_bytes(data)
    assert calculate_sha256(path) == hashlib.sha256(data).hexdigest()


def test_calculate_sha256_detects_single_byte_change(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"a,b\n1,2\n")
    before = calculate_sha256(path)
    path.write_bytes(b"a,b\n1,3\n")
    assert calculate_sha256(path) != before


# --------------------------------------------------------------------------- MLflow (base temporal)
def test_setup_mlflow_creates_and_reuses_experiment(config):
    exp_id = setup_mlflow(config, "Experimento_Prueba")
    assert setup_mlflow(config, "Experimento_Prueba") == exp_id
    assert mlflow.get_tracking_uri() == config.tracking_uri
    experiment = mlflow.MlflowClient().get_experiment(exp_id)
    assert experiment.artifact_location == config.artifact_location


@pytest.mark.parametrize("log_artifacts", [False, True])
def test_data_versioner_logs_hash_params_and_metrics(config, tmp_path, log_artifacts):
    config["mlflow"]["log_dataset_artifacts"] = log_artifacts
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "datos.csv"
    df.to_csv(path, index=False)

    run_id = MLflowDataVersioner(config).log_dataset_version(
        run_name="prueba", df=df, file_path=path, stage="raw",
        extra_params={"meses": ["2025-01"]}, extra_metrics={"filas_descartadas": 2}, tags={"role": "QA"},
    )

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.params["sha256_digest"] == calculate_sha256(path)
    assert run.data.params["num_rows"] == "3"
    assert run.data.params["num_columns"] == "2"
    assert run.data.params["meses"] == "['2025-01']"
    assert run.data.metrics["row_count"] == 3.0
    assert run.data.metrics["filas_descartadas"] == 2.0
    assert run.data.tags["dataset_stage"] == "raw"
    assert run.data.tags["role"] == "QA"
    assert len(run.inputs.dataset_inputs) == 1
    artifacts = [a.path for a in client.list_artifacts(run_id, "datasets")]
    assert artifacts == (["datasets/datos.csv"] if log_artifacts else [])
