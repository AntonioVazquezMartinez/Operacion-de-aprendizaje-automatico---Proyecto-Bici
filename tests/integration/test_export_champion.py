"""Exportación del champion desde el Model Registry temporal y verificación de la referencia."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from src.features import load_features, model_input_columns, temporal_split
from src.modeling import export
from src.modeling.export import STATION_COLUMNS, export_champion, reference_columns, serving_paths, verify_reference
from src.tracking import calculate_sha256

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def exported(e2e):
    metadata = export_champion(e2e.config)
    return metadata, serving_paths(e2e.config)


def test_metadata_points_to_champion(e2e, exported):
    metadata, paths = exported
    champion = e2e.client.get_model_version_by_alias(e2e.model_name, "champion")
    assert metadata["version"] == int(champion.version)
    assert metadata["alias"] == "champion"
    assert "champion" in metadata["aliases"]
    assert metadata["model_uri"] == f"models:/{e2e.model_name}/{champion.version}"
    assert metadata["run_id"] == champion.run_id
    assert metadata["algorithm"] == champion.tags["algorithm"]
    assert metadata["input_columns"] == model_input_columns(e2e.config)
    assert f"{metadata['metrics']['test_mae']:.4f}" == champion.tags["test_mae"]
    assert json.loads(paths["metadata"].read_text(encoding="utf-8")) == metadata


def test_metadata_hashes_match_files(e2e, exported):
    metadata, paths = exported
    assert metadata["model_sha256"] == calculate_sha256(paths["model_dir"] / metadata["model_file"])
    assert metadata["dataset_sha256"] == calculate_sha256(e2e.config.file("processed_dir", "features_file"))
    assert metadata["dataset_sha256_matches_registry"] is True
    assert metadata["stations_sha256"] == calculate_sha256(paths["stations"])
    assert metadata["reference_predictions"]["sha256"] == calculate_sha256(paths["reference"])
    assert b"\r\n" not in paths["reference"].read_bytes()      # mismo hash en Windows y Linux
    assert metadata["feature_store"]["sha256"] == calculate_sha256(paths["feature_store"])
    assert metadata["feature_store_reference_predictions"]["sha256"] == calculate_sha256(
        paths["feature_store_reference"])
    # Metadatos y archivos de texto de la carpeta MLflow con LF (mismo contenido en Windows y Linux).
    assert b"\r\n" not in paths["metadata"].read_bytes()
    for name in ("MLmodel", "conda.yaml", "python_env.yaml", "requirements.txt"):
        assert b"\r\n" not in (paths["model_dir"] / name).read_bytes(), name


def test_feature_store_exported(e2e, exported):
    metadata, paths = exported
    fs = metadata["feature_store"]
    store = pd.read_csv(paths["feature_store"], parse_dates=["fecha_hora"])
    df = load_features(e2e.config)
    assert store["fecha_hora"].max() == df["fecha_hora"].max() == pd.Timestamp(fs["end"])
    assert fs["rows"] == len(store) == fs["num_stations"] * fs["hours"]
    assert fs["predictable_window"]["start"] == (pd.Timestamp(fs["start"]) + pd.Timedelta(hours=168)).isoformat()
    assert fs["predictable_window"]["end"] == (pd.Timestamp(fs["end"]) + pd.Timedelta(hours=24)).isoformat()
    reference = pd.read_csv(paths["feature_store_reference"])
    assert len(reference) == metadata["feature_store_reference_predictions"]["rows"] > 0
    assert (reference["demand_pred"] >= 0).all()


def test_reference_sample(e2e, exported):
    metadata, paths = exported
    df = load_features(e2e.config)
    n_test = int((temporal_split(df, e2e.config) == "test").sum())
    reference = pd.read_csv(paths["reference"])
    assert list(reference.columns) == reference_columns(e2e.config)
    assert len(reference) == metadata["reference_predictions"]["rows"] == min(500, n_test)
    assert (reference["demand_pred"] >= 0).all()
    stations = pd.read_csv(paths["stations"])
    assert list(stations.columns) == STATION_COLUMNS
    assert stations["id"].tolist() == [1, 2, 3, 4] and metadata["num_stations"] == 4


def test_exported_folder_reproduces_reference_exactly(e2e, exported):
    assert verify_reference(e2e.config) == 0.0


def test_export_main_with_alias(e2e, exported, write_config, monkeypatch, tmp_path, capsys):
    cfg = copy.deepcopy(e2e.config)
    cfg["serving"].update({
        "model_dir": str(tmp_path / "challenger"),
        "metadata_file": str(tmp_path / "challenger_metadata.json"),
        "reference_predictions_file": str(tmp_path / "challenger_reference.csv"),
        "feature_store_reference_file": str(tmp_path / "challenger_feature_store_reference.csv"),
    })
    monkeypatch.setattr("sys.argv", ["export", "--config", str(write_config(cfg)), "--alias", "challenger"])
    export.main()
    metadata = json.loads((tmp_path / "challenger_metadata.json").read_text(encoding="utf-8"))
    challenger = e2e.client.get_model_version_by_alias(e2e.model_name, "challenger")
    assert metadata["alias"] == "challenger"
    assert metadata["version"] == int(challenger.version)
    assert "diferencia máxima absoluta = 0.000e+00" in capsys.readouterr().out
