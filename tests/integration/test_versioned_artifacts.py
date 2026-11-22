"""Coherencia de los artefactos de servicio versionados en el repositorio (models/).

`champion_metadata.json` y `reference_predictions.csv` se versionan en Git; la carpeta `models/champion/`
solo existe donde se exportó el modelo (no en CI), así que su prueba se omite si no está.
Estas pruebas solo leen archivos: no escriben en models/ ni usan la base de MLflow.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.modeling.export import reference_columns, serving_paths, verify_reference
from src.tracking import calculate_sha256

pytestmark = pytest.mark.integration

REPO_CONFIG = load_config()
PATHS = serving_paths(REPO_CONFIG)


@pytest.fixture(scope="module")
def metadata():
    if not PATHS["metadata"].exists() or not PATHS["reference"].exists():
        pytest.skip(f"No existen {PATHS['metadata'].name} y {PATHS['reference'].name}; ejecuta src.modeling.export")
    return json.loads(PATHS["metadata"].read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference(metadata):
    return pd.read_csv(PATHS["reference"])


def test_reference_columns_match_metadata(metadata, reference):
    assert set(metadata["input_columns"]) <= set(reference.columns)
    assert list(reference.columns) == reference_columns(REPO_CONFIG)
    assert sorted(metadata["input_columns"]) == sorted(set(reference.columns) - {"fecha_hora", "demand_count",
                                                                                 "demand_pred"})
    assert metadata["target"] in reference.columns


def test_reference_rows_and_values(metadata, reference):
    expected_rows = int(REPO_CONFIG["serving"]["reference_sample_rows"])
    assert len(reference) == metadata["reference_predictions"]["rows"] == expected_rows
    assert metadata["reference_predictions"]["seed"] == REPO_CONFIG["seed"]
    assert (reference["demand_pred"] >= 0).all()
    assert np.isfinite(reference[metadata["input_columns"]].to_numpy(dtype=float)).all()
    assert not reference.duplicated(["Origen_Id", "fecha_hora"]).any()


def test_reference_sha256_matches_metadata(metadata):
    """El SHA-256 de los metadatos corresponde al CSV versionado.

    Se acepta el mismo contenido con fin de línea LF o CRLF: el archivo se exportó en Windows y Git
    (core.autocrlf) puede entregarlo con cualquiera de los dos. El contenido debe ser idéntico.
    """
    raw = PATHS["reference"].read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    candidates = {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}
    assert metadata["reference_predictions"]["sha256"] in candidates
    assert calculate_sha256(PATHS["reference"]) == hashlib.sha256(raw).hexdigest()


def test_metadata_is_consistent_with_config(metadata):
    assert metadata["registered_model_name"] == REPO_CONFIG["mlflow"]["registered_model_name"]
    assert metadata["model_uri"] == f"models:/{metadata['registered_model_name']}/{metadata['version']}"
    assert metadata["alias"] in metadata["aliases"]
    assert metadata["target"] == REPO_CONFIG["features"]["target"]
    assert all(np.isfinite(v) for v in metadata["metrics"].values())


def test_local_champion_reproduces_reference(metadata):
    model_file = PATHS["model_dir"] / metadata["model_file"]
    if not model_file.exists():
        pytest.skip(f"{PATHS['model_dir']} no existe en este entorno (en CI no se versiona); "
                    "ejecuta `python -m src.modeling.export` para generarla")
    assert calculate_sha256(model_file) == metadata["model_sha256"]
    # En el sistema donde se exportó la diferencia es 0; en otro sistema operativo el exp() final de la
    # pérdida de Poisson puede variar 1 ulp (~1e-15), por eso se usa la misma tolerancia que
    # scripts/verify_reproducibility.py.
    assert verify_reference(REPO_CONFIG) <= 1e-9
