"""Construcción de estimadores y pipelines, traducción del espacio de búsqueda y reproducibilidad."""

from __future__ import annotations

import numpy as np
import optuna
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline

from src.modeling import train as train_module
from src.modeling.pipeline import build_estimator, build_pipeline, suggest_params
from src.modeling.train import ModelTrainer

MODEL_NAMES = ["dummy_mean", "poisson", "hist_gradient_boosting", "random_forest"]


class FakeTrial:
    """Imita un trial de Optuna devolviendo el límite inferior de cada rango."""

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_int(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


# --------------------------------------------------------------------------- build_estimator
def test_seed_is_injected(session_config):
    rf = build_estimator(session_config["models"]["random_forest"], seed=123)
    assert rf.get_params()["random_state"] == 123


def test_explicit_random_state_is_not_overwritten():
    est = build_estimator({"class": "sklearn.ensemble.RandomForestRegressor", "params": {"random_state": 7}}, seed=1)
    assert est.random_state == 7


def test_estimator_without_random_state_receives_no_seed():
    est = build_estimator({"class": "sklearn.linear_model.PoissonRegressor", "params": {"max_iter": 50}}, seed=3)
    assert isinstance(est, PoissonRegressor)
    assert "random_state" not in est.get_params()
    assert est.max_iter == 50


def test_tuned_params_override_fixed_params():
    cfg = {"class": "sklearn.ensemble.HistGradientBoostingRegressor", "params": {"max_iter": 100, "loss": "poisson"}}
    est = build_estimator(cfg, seed=0, params={"max_iter": 10})
    assert est.max_iter == 10
    assert est.loss == "poisson"


@pytest.mark.parametrize("cls, error", [
    ("sklearn.linear_model.NoExiste", AttributeError),
    ("sklearn.modulo_inexistente.Regresor", ModuleNotFoundError),
])
def test_unknown_class_raises(cls, error):
    with pytest.raises(error):
        build_estimator({"class": cls}, seed=0)


# --------------------------------------------------------------------------- build_pipeline
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_configured_model_trains_and_predicts(session_config, model_data, name):
    params = suggest_params(FakeTrial(), session_config["models"][name].get("search_space", {}))
    pipeline = build_pipeline(session_config, name, params)
    assert isinstance(pipeline, Pipeline)
    assert [s for s, _ in pipeline.steps] == ["preprocessor", "model"]
    preds = pipeline.fit(model_data["X_train"], model_data["y_train"]).predict(model_data["X_test"])
    assert len(preds) == len(model_data["X_test"])
    assert np.isfinite(preds).all()


def test_dummy_predicts_train_mean(session_config, model_data):
    pipeline = build_pipeline(session_config, "dummy_mean").fit(model_data["X_train"], model_data["y_train"])
    assert isinstance(pipeline.named_steps["model"], DummyRegressor)
    preds = pipeline.predict(model_data["X_test"])
    assert preds == pytest.approx(np.full(len(preds), model_data["y_train"].mean()))


# --------------------------------------------------------------------------- suggest_params
SPACE = {
    "alpha": {"type": "float", "low": 1.0e-6, "high": 1.0, "log": True},
    "max_iter": {"type": "int", "low": 100, "high": 500},
    "loss": {"type": "categorical", "choices": ["poisson", "squared_error"]},
}


def test_suggest_params_with_fake_trial():
    assert suggest_params(FakeTrial(), SPACE) == {"alpha": 1.0e-6, "max_iter": 100, "loss": "poisson"}


def test_suggest_params_with_fixed_optuna_trial():
    trial = optuna.trial.FixedTrial({"alpha": 0.01, "max_iter": 250, "loss": "squared_error"})
    assert suggest_params(trial, SPACE) == {"alpha": 0.01, "max_iter": 250, "loss": "squared_error"}


def test_suggest_params_types_and_ranges_with_real_study():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(10):
        params = suggest_params(study.ask(), SPACE)
        assert isinstance(params["alpha"], float) and 1e-6 <= params["alpha"] <= 1.0
        assert isinstance(params["max_iter"], int) and 100 <= params["max_iter"] <= 500
        assert params["loss"] in SPACE["loss"]["choices"]


def test_suggest_params_passes_log_flag():
    seen = {}

    class Recorder(FakeTrial):
        def suggest_float(self, name, low, high, log=False):
            seen[name] = log
            return low

        def suggest_int(self, name, low, high, log=False):
            seen[name] = log
            return low

    suggest_params(Recorder(), {"a": {"type": "float", "low": 1, "high": 2, "log": True},
                                "b": {"type": "int", "low": 1, "high": 9}})
    assert seen == {"a": True, "b": False}


def test_suggest_params_unsupported_type():
    with pytest.raises(ValueError, match="no soportado: uniforme"):
        suggest_params(FakeTrial(), {"x": {"type": "uniforme", "low": 0, "high": 1}})


def test_empty_search_space():
    assert suggest_params(FakeTrial(), {}) == {}


# --------------------------------------------------------------------------- reproducibilidad
def test_training_is_reproducible(session_config, model_data, trained_model):
    again = build_pipeline(session_config, "hist_gradient_boosting").fit(model_data["X_train"], model_data["y_train"])
    np.testing.assert_array_equal(trained_model.predict(model_data["X_test"]), again.predict(model_data["X_test"]))


def test_random_forest_depends_only_on_seed(session_config, model_data):
    params = {"n_estimators": 5}

    def fit_predict(seed):
        cfg = {**session_config, "seed": seed}
        return build_pipeline(type(session_config)(cfg), "random_forest", params).fit(
            model_data["X_train"], model_data["y_train"]).predict(model_data["X_test"])

    np.testing.assert_array_equal(fit_predict(1), fit_predict(1))
    assert not np.array_equal(fit_predict(1), fit_predict(2))


# --------------------------------------------------------------------------- ModelTrainer (sin MLflow)
def test_trainer_rejects_unknown_models(session_config):
    with pytest.raises(ValueError, match="no_existe"):
        ModelTrainer(session_config, model_names=["poisson", "no_existe"])


def test_trainer_defaults_to_enabled_models(config):
    config["models"]["random_forest"]["enabled"] = False
    assert ModelTrainer(config).model_names == ["dummy_mean", "poisson", "hist_gradient_boosting"]


def test_tune_without_search_space_returns_empty(session_config):
    assert ModelTrainer(session_config).tune("dummy_mean", parts={}) == {}


def test_register_models_without_candidates_does_nothing(session_config, monkeypatch):
    monkeypatch.setattr(train_module.mlflow, "register_model", lambda *a, **k: pytest.fail("no debe registrar"))
    ModelTrainer(session_config).register_models([{"register": False, "val_mae": 1.0}])


def test_train_main_passes_cli_arguments(session_config, write_config, monkeypatch):
    captured = {}

    class SpyTrainer:
        def __init__(self, config, model_names, n_trials, register):
            captured.update(seed=config["seed"], models=model_names, n_trials=n_trials, register=register)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(train_module, "ModelTrainer", SpyTrainer)
    argv = ["train", "--config", str(write_config(session_config)), "--models", "poisson", "dummy_mean",
            "--n-trials", "3", "--no-register"]
    monkeypatch.setattr("sys.argv", argv)
    train_module.main()
    assert captured == {"seed": 42, "models": ["poisson", "dummy_mean"], "n_trials": 3, "register": False,
                        "ran": True}
