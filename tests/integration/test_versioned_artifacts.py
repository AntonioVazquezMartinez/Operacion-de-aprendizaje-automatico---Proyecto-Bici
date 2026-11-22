"""Coherencia de los artefactos de servicio versionados en el repositorio (models/).

`champion_metadata.json`, las dos referencias y, desde la versión 1.1.0, la carpeta `models/champion/` (modelo +
feature store) se versionan en Git, así que también corren en CI (Linux). Las pruebas que necesitan la carpeta se
omiten si falta. Estas pruebas solo leen archivos: no escriben en models/ ni usan la base de MLflow.
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
        pytest.skip(f"{PATHS['model_dir']} no existe en este entorno; ejecuta `python -m src.modeling.export`")
    assert calculate_sha256(model_file) == metadata["model_sha256"]
    # En el sistema donde se exportó la diferencia es 0; en otro sistema operativo el exp() final de la
    # pérdida de Poisson puede variar 1 ulp (~1e-15), por eso se usa la misma tolerancia que
    # scripts/verify_reproducibility.py.
    assert verify_reference(REPO_CONFIG) <= 1e-9


def test_feature_store_sha256_and_window(metadata):
    fs = metadata.get("feature_store")
    path = PATHS["feature_store"]
    if not fs or not path.exists():
        pytest.skip("El modelo exportado no tiene feature store (python -m src.modeling.export)")
    assert calculate_sha256(path) == fs["sha256"]
    assert path.read_bytes()[4:8] == b"\x00\x00\x00\x00"      # gzip con mtime=0 (determinista)
    assert fs["hours"] == REPO_CONFIG["serving"]["feature_store_hours"]
    assert fs["rows"] == fs["hours"] * fs["num_stations"]
    assert fs["end"] == pd.Timestamp(metadata["data_period"]["end"]).isoformat()


def test_feature_store_reference_matches_metadata(metadata):
    md = metadata.get("feature_store_reference_predictions")
    path = PATHS["feature_store_reference"]
    if not md or not path.exists():
        pytest.skip("No existe la referencia del feature store")
    assert calculate_sha256(path) == md["sha256"]
    reference = pd.read_csv(path)
    assert len(reference) == md["rows"] == REPO_CONFIG["serving"]["feature_store_reference_rows"]
    window = metadata["feature_store"]["predictable_window"]
    hours = pd.to_datetime(reference["fecha_hora"])
    assert hours.min() >= pd.Timestamp(window["start"]) and hours.max() <= pd.Timestamp(window["end"])


def test_versioned_text_files_use_lf(metadata):
    """Archivos de texto de models/ con LF: los mismos bytes (y los mismos hashes de los metadatos y de dvc.lock)
    en Windows y en Linux; .gitattributes los fija con eol=lf."""
    files = [PATHS["metadata"], PATHS["reference"], PATHS["feature_store_reference"]]
    if PATHS["model_dir"].exists():
        files += [p for p in PATHS["model_dir"].iterdir() if p.suffix not in (".pkl", ".gz")]
    for path in files:
        if path.exists():
            assert b"\r\n" not in path.read_bytes(), path.name
