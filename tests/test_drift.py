"""Pruebas del módulo de monitoreo de data drift."""

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import DEFAULTS, DriftMonitor, assess, build_scenarios, psi


def test_psi_identical_distributions_is_zero():
    x = pd.Series(np.random.default_rng(0).poisson(2, 5000).astype(float))
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_mean_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi(ref, pd.Series(rng.normal(0, 1, 5000))) < DEFAULTS["psi_warning"]
    assert psi(ref, pd.Series(rng.normal(1.5, 1, 5000))) >= DEFAULTS["psi_alert"]


def test_psi_categorical_and_missing_values():
    ref = pd.Series([0, 1] * 500, dtype=float)
    cur = pd.Series([1] * 900 + [np.nan] * 100, dtype=float)
    assert psi(ref, cur) >= DEFAULTS["psi_alert"]


def test_scenarios(config, splits):
    test = splits["test"]
    scenarios = build_scenarios(test, config, DEFAULTS)
    assert "base_prueba" in scenarios
    missing = scenarios["rezagos_faltantes_40pct"][0]
    assert 0.3 < missing["demand_lag_24h"].isna().mean() < 0.5
    inflated = scenarios["rezagos_inflados_x2"][0]
    np.testing.assert_allclose(inflated["demand_lag_24h"], test["demand_lag_24h"] * 2)
    dropped = scenarios["caida_demanda_40pct"][0]
    assert dropped["demand_count"].sum() < test["demand_count"].sum()
    pd.testing.assert_series_equal(scenarios["base_prueba"][0]["demand_count"], test["demand_count"])


BASE = {"mae_change_pct": 0.0, "bias_pct": 0.0, "missing_pct": 0.0, "drift_share": 0.0,
        "max_feature_psi": 0.01, "target_psi": 0.0}


@pytest.mark.parametrize("change, level, keyword", [
    ({}, "OK", "Sin acción"),
    ({"missing_pct": 40.0}, "ALERTA", "feature pipeline"),
    ({"max_feature_psi": 2.0, "drift_share": 0.3, "mae_change_pct": 40.0}, "ALERTA", "feature pipeline"),
    ({"bias_pct": 70.0, "target_psi": 0.1}, "ALERTA", "reentrenar"),
    ({"mae_change_pct": 12.0}, "ADVERTENCIA", "monitoreo"),
])
def test_alert_rules(change, level, keyword):
    got_level, action = assess({**BASE, **change}, DEFAULTS)
    assert got_level == level
    assert keyword in action


def test_drift_monitor_end_to_end(config, features, bundle_dir):
    summary = DriftMonitor(config, bundle_dir).run(use_mlflow=False)
    assert summary["scenario"].iloc[0] == "base_prueba"
    assert summary["mae_change_pct"].iloc[0] == pytest.approx(0.0)
    assert summary.set_index("scenario").loc["rezagos_faltantes_40pct", "alert_level"] == "ALERTA"
    out = config["monitoring"]["output_dir"]
    alerts = json.loads(open(f"{out}/alertas.json", encoding="utf-8").read())
    assert {"thresholds", "scenarios", "weekly"} <= set(alerts)
