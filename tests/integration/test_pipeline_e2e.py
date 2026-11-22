"""Flujo extremo a extremo: carga -> limpieza -> features -> entrenamiento -> Registry -> predicción -> métricas."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features import load_features, model_input_columns, temporal_split
from src.modeling.pipeline import evaluate
from src.modeling.predict import load_registered_model, predict
from src.tracking import calculate_sha256
from tests.integration.conftest import E2E_MODELS, E2E_TRIALS

pytestmark = pytest.mark.integration

METRIC_COLUMNS = [f"{p}_{m}" for p in ("val", "test") for m in ("mae", "rmse", "r2", "poisson_deviance")]


def runs_of(e2e, experiment_key: str, filter_string: str = ""):
    experiment = e2e.client.get_experiment_by_name(e2e.config["mlflow"][experiment_key])
    return e2e.client.search_runs([experiment.experiment_id], filter_string)


# --------------------------------------------------------------------------- datos
def test_data_versions_are_logged_with_hashes(e2e):
    runs = {r.data.tags["dataset_stage"]: r for r in runs_of(e2e, "data_experiment")}
    assert set(runs) == {"raw", "interim", "processed"}
    cfg = e2e.config
    files = {
        "raw": cfg.file("raw_dir", "raw_trips_file"),
        "interim": cfg.file("interim_dir", "interim_file"),
        "processed": cfg.file("processed_dir", "features_file"),
    }
    for stage, path in files.items():
        assert runs[stage].data.params["sha256_digest"] == calculate_sha256(path), stage
    # Las anomalías inyectadas por make_raw_fixture quedan registradas en la versión interim.
    interim = runs["interim"].data.metrics
    assert interim["initial_rows"] == 3002
    assert interim["dropped_duplicates"] == 2
    assert interim["final_rows"] == 3002 - 42
    assert runs["processed"].data.metrics["num_stations"] == 4


# --------------------------------------------------------------------------- entrenamiento y comparación
def test_comparison_table(e2e):
    table = e2e.table
    assert sorted(table["model"]) == sorted(E2E_MODELS)
    assert table["val_mae"].is_monotonic_increasing
    assert np.isfinite(table[METRIC_COLUMNS].to_numpy()).all()
    assert (table[["val_mae", "test_mae", "val_rmse", "test_rmse"]] >= 0).all().all()
    saved = pd.read_csv(e2e.config.path("reports_dir") / "model_comparison.csv")
    assert saved["model"].tolist() == table["model"].tolist()
    assert (e2e.config.path("figures_dir") / "06_comparacion_modelos.png").exists()


def test_best_model_beats_dummy(e2e):
    """La demanda tiene patrón horario y rezagos informativos: el mejor modelo supera a la media constante."""
    by_model = e2e.table.set_index("model")
    best = e2e.table.iloc[0]["model"]
    assert best != "dummy_mean"
    assert by_model.loc[best, "val_mae"] < by_model.loc["dummy_mean", "val_mae"]
    assert by_model.loc[best, "test_mae"] < by_model.loc["dummy_mean", "test_mae"]
    assert by_model.loc[best, "test_r2"] > by_model.loc["dummy_mean", "test_r2"]


def test_training_runs_are_tracked(e2e):
    sha = calculate_sha256(e2e.config.file("processed_dir", "features_file"))
    for name in E2E_MODELS:
        (run,) = runs_of(e2e, "training_experiment", f"attributes.run_name = '{name}'")
        params, metrics = run.data.params, run.data.metrics
        assert params["model"] == name
        assert params["dataset_sha256"] == sha
        assert params["seed"] == "42"
        assert params["features"] == ",".join(model_input_columns(e2e.config))
        assert int(params["rows_train"]) > int(params["rows_val"]) > 0
        assert set(METRIC_COLUMNS) <= set(metrics)
        assert all(math.isfinite(metrics[k]) for k in METRIC_COLUMNS)
        artifacts = [a.path for a in e2e.client.list_artifacts(run.info.run_id, "plots")]
        assert artifacts == [f"plots/perfil_horario_{name}.png"]


@pytest.mark.parametrize("name", ["poisson", "hist_gradient_boosting"])
def test_optuna_trials_are_nested_runs(e2e, name):
    (parent,) = runs_of(e2e, "training_experiment", f"attributes.run_name = '{name}'")
    trials = runs_of(e2e, "training_experiment", f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(trials) == E2E_TRIALS
    assert parent.data.metrics["tuning_trials_completed"] == E2E_TRIALS
    # Se buscó con una muestra de train (train_sample_rows) y aun así cada trial evaluó en val.
    assert all("val_mae" in t.data.metrics for t in trials)
    assert parent.data.params["tuning_timeout_seconds"] == "none"


def test_comparison_run_artifacts(e2e):
    (comparison,) = runs_of(e2e, "training_experiment", "tags.stage = 'comparison'")
    artifacts = {a.path for a in e2e.client.list_artifacts(comparison.info.run_id)}
    assert {"config.yaml", "model_comparison.csv", "plots"} <= artifacts
    assert comparison.data.tags["best_model"] == e2e.table.iloc[0]["model"]


# --------------------------------------------------------------------------- Model Registry
def test_registry_versions_and_aliases(e2e):
    versions = e2e.client.search_model_versions(f"name = '{e2e.model_name}'")
    registered = {v.tags["algorithm"]: v for v in versions}
    assert set(registered) == {"poisson", "hist_gradient_boosting"}     # dummy_mean: register: false

    ranked = e2e.table[e2e.table["model"] != "dummy_mean"]["model"].tolist()
    champion = e2e.client.get_model_version_by_alias(e2e.model_name, "champion")
    challenger = e2e.client.get_model_version_by_alias(e2e.model_name, "challenger")
    assert champion.tags["algorithm"] == ranked[0]
    assert challenger.tags["algorithm"] == ranked[1]
    for v in (champion, challenger):
        assert {"val_mae", "test_mae", "test_rmse", "test_r2", "hyperparameters", "dataset_sha256",
                "training_run_id"} <= set(v.tags)
        assert v.description and "test MAE=" in v.description


def test_reloaded_champion_reproduces_registered_metrics(e2e):
    model, uri = load_registered_model(e2e.config)
    assert uri == f"models:/{e2e.model_name}@champion"
    df = load_features(e2e.config)
    test = df[temporal_split(df, e2e.config) == "test"]
    metrics = evaluate(test["demand_count"], model.predict(test[model_input_columns(e2e.config)]), "test")

    champion = e2e.client.get_model_version_by_alias(e2e.model_name, "champion")
    row = e2e.table.set_index("model").loc[champion.tags["algorithm"]]
    assert metrics["test_mae"] == pytest.approx(row["test_mae"], rel=1e-12)
    assert metrics["test_rmse"] == pytest.approx(row["test_rmse"], rel=1e-12)
    assert f"{metrics['test_mae']:.4f}" == champion.tags["test_mae"]


def test_predict_with_champion_writes_non_negative_predictions(e2e):
    out = predict(e2e.config)
    df = load_features(e2e.config)
    assert len(out) == int((temporal_split(df, e2e.config) == "test").sum())
    assert list(out.columns) == ["Origen_Id", "fecha_hora", "demand_count", "demand_pred"]
    assert (out["demand_pred"] >= 0).all()
    assert np.isfinite(out["demand_pred"]).all()
    saved = pd.read_csv(e2e.config.path("processed_dir") / "predicciones_champion.csv")
    assert len(saved) == len(out)


def test_challenger_alias_also_predicts(e2e, tmp_path):
    out = predict(e2e.config, alias="challenger", output_path=str(tmp_path / "challenger.csv"))
    champion = predict(e2e.config, output_path=str(tmp_path / "champion.csv"))
    assert len(out) == len(champion)
    assert not np.allclose(out["demand_pred"], champion["demand_pred"])   # son modelos distintos
