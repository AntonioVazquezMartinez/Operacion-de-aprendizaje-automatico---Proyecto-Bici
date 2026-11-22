"""Inferencia por lotes (`src.modeling.predict`) y utilidades de exportación (`src.modeling.export`).

El modelo del Registry se sustituye por el pipeline pequeño de la sesión (monkeypatch), y la carpeta
exportada se genera con `mlflow.sklearn.save_model` en un directorio temporal: no se usa la base real.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from src.config import PROJ_ROOT
from src.features import model_input_columns, temporal_split
from src.modeling import export, predict as predict_module
from src.modeling.export import (
    _pickled_model_file,
    _resolve,
    predict_clipped,
    reference_columns,
    serving_paths,
    verify_reference,
)
from src.modeling.pipeline import evaluate
from src.modeling.predict import predict


class NegativeModel:
    """Modelo que siempre predice -1 (para comprobar el recorte a cero)."""

    def predict(self, X):
        return np.full(len(X), -1.0)


@pytest.fixture
def patched_registry(monkeypatch, trained_model):
    """Sustituye la carga del Registry por el pipeline entrenado en la sesión."""
    calls = []

    def fake_load(config, alias=None):
        calls.append(alias)
        return trained_model, f"models:/fake@{alias or 'champion'}"

    monkeypatch.setattr(predict_module, "load_registered_model", fake_load)
    return calls


# --------------------------------------------------------------------------- predict
def test_predict_default_uses_test_split(session_config, feature_matrix, split_labels, trained_model,
                                         patched_registry, tmp_path, capsys):
    out = predict(session_config, output_path=str(tmp_path / "pred.csv"))

    test = feature_matrix[split_labels == "test"].reset_index(drop=True)
    assert list(out.columns) == ["Origen_Id", "fecha_hora", "demand_count", "demand_pred"]
    assert len(out) == len(test)
    expected = np.clip(trained_model.predict(test[model_input_columns(session_config)]), 0, None)
    np.testing.assert_array_equal(out["demand_pred"].to_numpy(), expected)
    assert (out["demand_pred"] >= 0).all()
    mae = evaluate(test["demand_count"], expected, "pred")["pred_mae"]
    assert f"MAE {mae:.4f}" in capsys.readouterr().out
    assert len(pd.read_csv(tmp_path / "pred.csv")) == len(test)


def test_predict_default_output_path_uses_alias(config, feature_matrix, patched_registry, session_config):
    config["paths"]["processed_dir"] = session_config["paths"]["processed_dir"]   # reutiliza la matriz
    predict(config, alias="challenger")
    assert patched_registry == ["challenger"]
    assert (config.path("processed_dir") / "predicciones_challenger.csv").exists()


def test_predict_from_input_file_without_target(session_config, model_data, patched_registry, tmp_path, capsys):
    inputs = model_data["X_test"].head(20)
    in_path = tmp_path / "entrada.csv"
    inputs.to_csv(in_path, index=False)
    out = predict(session_config, input_path=str(in_path), output_path=str(tmp_path / "salida.csv"))
    assert list(out.columns) == ["Origen_Id", "demand_pred"]     # sin objetivo no hay métricas
    assert len(out) == 20
    assert "MAE" not in capsys.readouterr().out


def test_predict_missing_columns_raises(session_config, model_data, patched_registry, tmp_path):
    in_path = tmp_path / "incompleto.csv"
    model_data["X_test"].drop(columns=["demand_lag_24h", "capacity"]).head(5).to_csv(in_path, index=False)
    with pytest.raises(ValueError, match=r"Faltan columnas de entrada: \['capacity', 'demand_lag_24h'\]"):
        predict(session_config, input_path=str(in_path), output_path=str(tmp_path / "x.csv"))


def test_predictions_are_never_negative(session_config, model_data, monkeypatch, tmp_path):
    monkeypatch.setattr(predict_module, "load_registered_model", lambda c, a=None: (NegativeModel(), "fake"))
    in_path = tmp_path / "entrada.csv"
    model_data["X_test"].head(10).to_csv(in_path, index=False)
    out = predict(session_config, input_path=str(in_path), output_path=str(tmp_path / "s.csv"))
    assert (out["demand_pred"] == 0).all()
    assert (predict_clipped(NegativeModel(), model_data["X_test"].head(3)) == 0).all()


def test_load_registered_model_builds_alias_uri(session_config, monkeypatch):
    seen = {}
    monkeypatch.setattr(predict_module.mlflow.sklearn, "load_model", lambda uri: seen.setdefault("uri", uri))
    model, uri = predict_module.load_registered_model(session_config)
    assert uri == "models:/mibici-demand-forecaster@champion" == seen["uri"]
    _, uri = predict_module.load_registered_model(session_config, alias="challenger")
    assert uri.endswith("@challenger")


# --------------------------------------------------------------------------- export: utilidades
def test_reference_columns(session_config):
    cols = reference_columns(session_config)
    assert cols[:2] == ["Origen_Id", "fecha_hora"]
    assert cols[-2:] == ["demand_count", "demand_pred"]
    assert cols.count("Origen_Id") == 1
    assert set(model_input_columns(session_config)) <= set(cols)


def test_serving_paths_and_resolve(session_config, tmp_path):
    paths = serving_paths(session_config)
    assert paths["stations"] == paths["model_dir"] / "stations.csv"
    assert all(p.is_absolute() for p in paths.values())
    assert _resolve("models/champion") == PROJ_ROOT / "models" / "champion"
    assert _resolve(str(tmp_path)) == tmp_path


def test_library_versions_keys():
    assert set(export._library_versions()) == {"python", "scikit-learn", "mlflow", "pandas", "numpy"}


# --------------------------------------------------------------------------- export: carpeta + referencia
@pytest.fixture
def exported_folder(config, feature_matrix, trained_model):
    """Carpeta de modelo MLflow y referencia escritas como lo hace `export_champion` (sin Registry)."""
    paths = serving_paths(config)
    mlflow.sklearn.save_model(trained_model, str(paths["model_dir"]), serialization_format="cloudpickle",
                              pip_requirements=["scikit-learn"])
    test = feature_matrix[temporal_split(feature_matrix, config) == "test"]
    sample = test.sample(n=50, random_state=config["seed"]).sort_values(["fecha_hora", "Origen_Id"])
    sample = sample.reset_index(drop=True)
    sample["demand_pred"] = predict_clipped(trained_model, sample[model_input_columns(config)])
    paths["reference"].parent.mkdir(parents=True, exist_ok=True)
    sample[reference_columns(config)].to_csv(paths["reference"], index=False)
    return paths


def test_pickled_model_file_reads_mlmodel(exported_folder):
    pkl = _pickled_model_file(exported_folder["model_dir"])
    assert pkl.parent == exported_folder["model_dir"]
    assert pkl.exists()


def test_exported_folder_reproduces_reference_exactly(config, exported_folder):
    """Recarga la carpeta exportada y compara con la referencia escrita en CSV: diferencia 0."""
    assert verify_reference(config) == 0.0


def test_verify_reference_detects_tampering(config, exported_folder):
    ref = pd.read_csv(exported_folder["reference"])
    ref.loc[0, "demand_pred"] += 0.5
    ref.to_csv(exported_folder["reference"], index=False)
    assert verify_reference(config) == pytest.approx(0.5)


def test_export_main_verify_only(config, exported_folder, write_config, monkeypatch, capsys):
    monkeypatch.setattr(export, "export_champion", lambda *a, **k: pytest.fail("--verify-only no exporta"))
    monkeypatch.setattr("sys.argv", ["export", "--config", str(write_config(config)), "--verify-only"])
    export.main()
    assert "diferencia máxima absoluta = 0.000e+00" in capsys.readouterr().out


def test_predict_main(session_config, model_data, patched_registry, write_config, monkeypatch, tmp_path):
    in_path = tmp_path / "entrada.csv"
    model_data["X_test"].head(5).to_csv(in_path, index=False)
    out_path = tmp_path / "salida.csv"
    argv = ["predict", "--config", str(write_config(session_config)), "--input", str(in_path),
            "--output", str(out_path), "--alias", "challenger"]
    monkeypatch.setattr("sys.argv", argv)
    predict_module.main()
    assert patched_registry == ["challenger"]
    assert len(pd.read_csv(out_path)) == 5


def test_verify_needs_neither_metadata_nor_tracking_db(config, exported_folder):
    """La verificación solo necesita la carpeta y la referencia: sin metadatos ni base de MLflow."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri("http://127.0.0.1:9")      # servidor inexistente: cualquier consulta fallaría
    try:
        assert verify_reference(config) == 0.0
    finally:
        mlflow.set_tracking_uri(previous)
    assert not exported_folder["metadata"].exists()
