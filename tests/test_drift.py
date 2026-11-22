"""Pruebas del monitoreo de drift (src/monitoring): métricas, escenarios, reglas e integración sintética.

Autocontenidas: no usan datos reales, ni el modelo exportado, ni MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.features import MiBiciFeatureEngineer, lag_columns, model_input_columns, temporal_split  # noqa: E402
from src.modeling.pipeline import build_pipeline  # noqa: E402
from src.monitoring import metrics as M  # noqa: E402
from src.monitoring import rules as R  # noqa: E402
from src.monitoring import scenarios as S  # noqa: E402
from src.monitoring.drift import PRED_COL, Monitor, run_monitoring  # noqa: E402
from tests.conftest_data import make_test_config  # noqa: E402

THRESHOLDS = {
    "psi": {"warning": 0.10, "alert": 0.25},
    "ks_statistic": {"warning": 0.10, "alert": 0.20},
    "wasserstein_norm": {"warning": 0.10, "alert": 0.25},
    "missing_rate": {"warning": 0.01, "alert": 0.05},
    "count_domain_violation_rate": {"warning": 0.01, "alert": 0.05},
    "unseen_category_rate": {"warning": 0.01, "alert": 0.05},
    "hour_profile_shift_hours": {"warning": 1, "alert": 2},
    "mae_degradation": {"warning": 0.10, "alert": 0.25},
    "wape_degradation": {"warning": 0.10, "alert": 0.25},
}


# --------------------------------------------------------------------------- datos sintéticos
def synthetic_trips(n_stations: int = 20, days: int = 60, seed: int = 0):
    """Viajes estacionarios (patrón por hora y estación, sin tendencia) y catálogo de estaciones."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n_stations + 1)
    stations = pd.DataFrame({"id": ids, "name": [f"E{i}" for i in ids],
                             "latitude": 20.6 + rng.random(n_stations) / 10,
                             "longitude": -103.4 + rng.random(n_stations) / 10,
                             "capacity": rng.integers(10, 30, n_stations)})
    hours = pd.date_range("2024-06-03", periods=days * 24, freq="h")
    hour_shape = 0.2 + 2.0 * np.exp(-((hours.hour.to_numpy() - 8) ** 2) / 4) \
        + 2.5 * np.exp(-((hours.hour.to_numpy() - 18) ** 2) / 4)
    station_scale = rng.uniform(0.4, 1.6, n_stations)
    counts = rng.poisson(np.outer(station_scale, hour_shape))  # estaciones x horas
    st_idx, h_idx = np.nonzero(counts)
    reps = counts[st_idx, h_idx]
    start = np.repeat(hours.to_numpy()[h_idx], reps) + pd.to_timedelta(
        rng.integers(0, 3600, reps.sum()), unit="s").to_numpy()
    trips = pd.DataFrame({"Inicio_del_viaje": start, "Origen_Id": np.repeat(ids[st_idx], reps)})
    return trips.sort_values("Inicio_del_viaje").reset_index(drop=True), stations


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("drift")
    config = make_test_config(tmp)
    trips, stations = synthetic_trips()
    features = MiBiciFeatureEngineer(config).build_features(trips, stations)
    split = temporal_split(features, config)
    train_val = features[split != "test"]
    model = build_pipeline(config, "poisson", {"alpha": 1e-4}).fit(
        train_val[model_input_columns(config)], train_val[config["features"]["target"]])
    return {"tmp": tmp, "config": config, "trips": trips, "stations": stations, "features": features,
            "split": split, "model": model}


# --------------------------------------------------------------------------- métricas
def test_psi_zero_for_identical_and_grows_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000)
    assert M.psi(ref, ref.copy()) == pytest.approx(0.0, abs=1e-12)
    values = [M.psi(ref, ref + shift) for shift in (0.1, 0.5, 1.0, 2.0)]
    assert all(np.diff(values) > 0)
    assert values[0] < 0.10 < values[2]


def test_psi_finite_with_empty_bins_and_roughly_symmetric():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 5000), rng.normal(0.5, 1, 5000)
    concentrated = np.full(1000, 100.0)  # todo en el último intervalo
    assert np.isfinite(M.psi(a, concentrated)) and M.psi(a, concentrated) > 0.25
    forward, backward = M.psi(a, b), M.psi(b, a)
    assert np.isfinite(forward) and np.isfinite(backward)
    assert forward == pytest.approx(backward, rel=0.5)


def test_psi_discrete_and_missing_as_own_category():
    ref = np.array([0] * 900 + [1] * 100, dtype=float)
    assert M.is_discrete(ref, 30)
    assert M.psi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    # p = (0.9, 0.1), q = (0.5, 0.5): PSI exacto sin épsilon en las categorías observadas
    cur = np.array([0] * 500 + [1] * 500, dtype=float)
    expected = (0.5 - 0.9) * np.log(0.5 / 0.9) + (0.5 - 0.1) * np.log(0.5 / 0.1)
    assert M.psi(ref, cur) == pytest.approx(expected, rel=1e-3)
    with_nan = ref.copy()
    with_nan[:200] = np.nan
    assert M.psi(ref, with_nan) > 0.25
    assert M.missing_rate(with_nan) == pytest.approx(0.2)


def test_ks_and_wasserstein_known_cases():
    ref = np.array([0.0, 1.0, 2.0, 3.0])
    stat, _ = M.ks_test(ref, ref)
    assert stat == pytest.approx(0.0)
    stat, _ = M.ks_test(ref, ref + 10)
    assert stat == pytest.approx(1.0)
    # Desplazar todo 1 unidad: Wasserstein = 1, normalizada entre la desv. est. poblacional (sqrt(1.25)).
    assert M.wasserstein_normalized(ref, ref + 1) == pytest.approx(1 / np.sqrt(1.25))
    assert M.wasserstein_normalized(ref, ref) == pytest.approx(0.0)
    big = np.random.default_rng(3).normal(size=200000)
    stat_a, _ = M.ks_test(big, big + 0.5, sample_rows=5000, seed=7)
    stat_b, _ = M.ks_test(big, big + 0.5, sample_rows=5000, seed=7)
    assert stat_a == stat_b  # submuestra con semilla: reproducible


def test_count_domain_violation_rate():
    values = np.array([[0, 1], [2, 3], [1.5, 0], [np.nan, 4], [-1, 0]], dtype=float)
    assert M.count_domain_violation_rate(values) == pytest.approx(2 / 5)  # 1.5 y -1; NaN no cuenta
    assert M.count_domain_violation_rate(np.array([0.0, 3.0, 7.0])) == 0.0


def test_unseen_rate_and_hour_profile_shift():
    assert M.unseen_category_rate(np.array([1, 2, 3, 99]), [1, 2, 3]) == pytest.approx(0.25)
    hours = np.arange(24)
    assert np.array_equal(M.hour_from_cyclic(np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)),
                          hours)
    profile = np.exp(-((hours - 8) ** 2) / 4) + np.exp(-((hours - 18) ** 2) / 4)
    assert M.hour_profile_shift(profile, profile)["shift_hours"] == 0
    shifted = np.roll(profile, 6)  # la hora codificada va 6 h adelantada
    assert M.hour_profile_shift(profile, shifted)["shift_hours"] == 6
    assert M.hour_profile_shift(profile, np.roll(profile, -3))["shift_hours"] == -3


# --------------------------------------------------------------------------- escenarios
def _test_rows(synthetic):
    feats = synthetic["features"]
    return feats[synthetic["split"] == "test"].reset_index(drop=True)


def test_missing_lags_only_touches_lags(synthetic):
    base = _test_rows(synthetic)
    lags = lag_columns(synthetic["config"])
    out = S.missing_lags(base, lags, 0.3, seed=42)
    other = [c for c in base.columns if c not in lags]
    pd.testing.assert_frame_equal(out[other], base[other])
    rate = out[lags[0]].isna().mean()
    assert 0.2 < rate < 0.4
    assert out[lags].isna().all(axis=1).sum() == out[lags[0]].isna().sum()  # todos los rezagos de la fila
    assert base[lags].notna().all().all()  # sin mutar la entrada


def test_scale_lags_and_clock_shift(synthetic):
    base = _test_rows(synthetic)
    lags = lag_columns(synthetic["config"])
    scaled = S.scale_lags(base, lags, 1.5)
    np.testing.assert_allclose(scaled[lags].to_numpy(), base[lags].to_numpy() * 1.5)
    target = synthetic["config"]["features"]["target"]
    shifted = S.clock_shift(base, 6)
    pd.testing.assert_series_equal(shifted[target], base[target])
    pd.testing.assert_frame_equal(shifted[lags], base[lags])
    assert (shifted["hour"] == (base["hour"] + 6) % 24).all()
    # La distribución marginal de la hora no cambia (por eso hace falta el perfil horario).
    assert M.psi(base["sin_hour"], shifted["sin_hour"]) < 0.01


def test_rebuild_identity_and_demand_increase_coherent(synthetic):
    config, base = synthetic["config"], _test_rows(synthetic)
    engineer = MiBiciFeatureEngineer(config)
    target = config["features"]["target"]
    start, end = base["fecha_hora"].min(), base["fecha_hora"].max()
    ids = sorted(base["Origen_Id"].unique())
    rebuilt = S.rebuild_features(engineer, synthetic["trips"], synthetic["stations"], ids, start, end)
    expected = base.sort_values(["fecha_hora", "Origen_Id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt[expected.columns], expected, check_dtype=False)

    more = S.change_demand(synthetic["trips"], 1.4, seed=42)
    assert len(more) == len(synthetic["trips"]) + round(0.4 * len(synthetic["trips"]))
    up = S.rebuild_features(engineer, more, synthetic["stations"], ids, start, end)
    assert up[target].sum() / rebuilt[target].sum() == pytest.approx(1.4, rel=0.05)
    assert up["demand_lag_24h"].sum() / rebuilt["demand_lag_24h"].sum() == pytest.approx(1.4, rel=0.05)
    # Coherencia: el rezago de 24 h es el objetivo de 24 h antes en la misma estación.
    wide = up.pivot(index="fecha_hora", columns="Origen_Id", values=target)
    lag = up.pivot(index="fecha_hora", columns="Origen_Id", values="demand_lag_24h")
    np.testing.assert_array_equal(lag.iloc[24:].to_numpy(), wide.iloc[:-24].to_numpy())

    fewer = S.change_demand(synthetic["trips"], 0.6, seed=42)
    assert len(fewer) / len(synthetic["trips"]) == pytest.approx(0.6, abs=0.02)


def test_new_stations_unseen_and_without_history(synthetic):
    config, base = synthetic["config"], _test_rows(synthetic)
    start = base["fecha_hora"].min()
    ids = sorted(base["Origen_Id"].unique())
    trips, catalog, grid_ids, mapping = S.new_stations(synthetic["trips"], synthetic["stations"], ids, 0.15,
                                                       start, 100000, seed=42)
    assert len(mapping) == round(len(ids) * 0.15)
    new_ids = set(mapping.values())
    assert new_ids.isdisjoint(ids) and new_ids <= set(catalog["id"]) and new_ids <= set(grid_ids)
    assert (trips.loc[trips["Origen_Id"].isin(new_ids), "Inicio_del_viaje"] >= start).all()
    rebuilt = S.rebuild_features(MiBiciFeatureEngineer(config), trips, catalog, grid_ids, start,
                                 base["fecha_hora"].max())
    first_day = rebuilt[rebuilt["Origen_Id"].isin(new_ids) & (rebuilt["fecha_hora"] < start + pd.Timedelta(hours=24))]
    assert (first_day[lag_columns(config)] == 0).all().all()
    assert M.unseen_category_rate(rebuilt["Origen_Id"], ids) == pytest.approx(len(new_ids) / len(ids))


# --------------------------------------------------------------------------- reglas
@pytest.mark.parametrize("value, expected", [
    (0.0999, "OK"), (0.10, "AVISO"), (0.2499, "AVISO"), (0.25, "ALERTA"), (3.0, "ALERTA"), (None, None),
    (float("nan"), None),
])
def test_level_from_value_boundaries(value, expected):
    assert R.level_from_value(value, 0.10, 0.25) == expected


@pytest.mark.parametrize("metrics, expected", [
    ({"psi": 0.09, "ks_statistic": 0.05, "wasserstein_norm": 0.05}, "OK"),
    ({"psi": 0.10, "ks_statistic": 0.05, "wasserstein_norm": 0.05}, "AVISO"),
    ({"psi": 0.01, "ks_statistic": 0.20, "wasserstein_norm": 0.05}, "ALERTA"),
    ({"psi": 0.01, "ks_statistic": 0.01, "wasserstein_norm": 0.25}, "ALERTA"),
    # En discretas la Wasserstein normalizada no decide.
    ({"psi": 0.01, "ks_statistic": 0.01, "wasserstein_norm": 0.90, "discrete": True}, "OK"),
])
def test_variable_level(metrics, expected):
    assert R.variable_level(metrics, THRESHOLDS) == expected


@pytest.mark.parametrize("mae, wape, expected", [
    (0.0999, 0.0, "OK"), (0.10, 0.0, "AVISO"), (0.25, 0.0, "ALERTA"), (-0.2, 0.10, "AVISO"),
    (-0.2, 0.25, "ALERTA"), (None, None, None),
])
def test_performance_level_boundaries(mae, wape, expected):
    assert R.performance_level(mae, wape, THRESHOLDS) == expected


@pytest.mark.parametrize("missing, unseen, shift, expected", [
    (0.0, 0.0, 0, "OK"), (0.01, 0.0, 0, "AVISO"), (0.05, 0.0, 0, "ALERTA"), (0.0, 0.0099, 0, "OK"),
    (0.0, 0.01, 0, "AVISO"), (0.0, 0.05, 0, "ALERTA"), (0.0, 0.0, 1, "AVISO"), (0.0, 0.0, -2, "ALERTA"),
])
def test_quality_level_boundaries(missing, unseen, shift, expected):
    assert R.quality_level(missing, unseen, shift, THRESHOLDS)["level"] == expected


@pytest.mark.parametrize("domain, expected", [(None, "OK"), (0.0099, "OK"), (0.01, "AVISO"), (0.05, "ALERTA")])
def test_quality_domain_boundaries(domain, expected):
    quality = R.quality_level(0.0, 0.0, 0, THRESHOLDS, domain)
    assert quality["level"] == expected
    if expected != "OK":
        assert R.decide(quality, "ALERTA", "AVISO")["rule"] == "R1_pipeline"


Q_OK = {"missing": "OK", "unseen": "OK", "hour_shift": "OK", "domain": "OK", "level": "OK"}


@pytest.mark.parametrize("quality, drift, perf, level, rule", [
    (Q_OK, "OK", "OK", "OK", "R0_ok"),
    ({**Q_OK, "domain": "AVISO", "level": "AVISO"}, "ALERTA", "AVISO", "ALERTA", "R1_pipeline"),
    ({**Q_OK, "missing": "ALERTA", "level": "ALERTA"}, "ALERTA", "OK", "ALERTA", "R1_pipeline"),
    ({**Q_OK, "hour_shift": "AVISO", "level": "AVISO"}, "OK", "ALERTA", "ALERTA", "R1_pipeline"),
    ({**Q_OK, "unseen": "AVISO", "level": "AVISO"}, "OK", "OK", "AVISO", "R2_catalogo"),
    (Q_OK, "AVISO", "ALERTA", "ALERTA", "R3_reentrenar"),
    (Q_OK, "ALERTA", "AVISO", "ALERTA", "R4_programar"),
    (Q_OK, "OK", "AVISO", "AVISO", "R5_concepto"),
    (Q_OK, "ALERTA", "OK", "ALERTA", "R6_vigilar"),
    (Q_OK, "AVISO", None, "AVISO", "R6_vigilar"),          # sin etiquetas
    (Q_OK, "ALERTA", None, "ALERTA", "R7_sin_etiquetas"),  # sin etiquetas
    (Q_OK, "OK", None, "OK", "R0_ok"),                     # sin etiquetas
])
def test_decide_rules(quality, drift, perf, level, rule):
    decision = R.decide(quality, drift, perf)
    assert decision["level"] == level and decision["rule"] == rule
    assert decision["action"] == R.ACTIONS[rule]


# --------------------------------------------------------------------------- control e integración
def test_control_against_itself_is_ok(synthetic):
    config, base = synthetic["config"], _test_rows(synthetic)
    monitor = Monitor(config, synthetic["model"], base, base)
    result = monitor.assess(monitor.baseline)
    assert result["decision"]["level"] == "OK" and result["decision"]["rule"] == "R0_ok"
    assert result["degradation"]["mae"] == pytest.approx(0.0)
    assert PRED_COL in result["variables"]


def test_end_to_end_synthetic(synthetic, tmp_path):
    config = synthetic["config"]
    test_rows = _test_rows(synthetic)
    config["monitoring"]["scenarios"]["vacaciones_diciembre"].update(
        start=str(test_rows["fecha_hora"].min() + pd.Timedelta(days=2)), end=str(test_rows["fecha_hora"].max()))
    out_dir, fig_dir = tmp_path / "drift", tmp_path / "figures"
    report = run_monitoring(config, use_mlflow=False, features=synthetic["features"], trips=synthetic["trips"],
                            stations=synthetic["stations"], model=synthetic["model"], output_dir=out_dir,
                            figures_dir=fig_dir)
    for name in ("drift_report.json", "drift_report.md", "drift_summary.csv"):
        assert (out_dir / name).stat().st_size > 0
    for name in ("07_drift_distribuciones.png", "08_drift_psi_por_variable.png", "09_drift_desempeno.png",
                 "10_drift_ventanas_tiempo.png"):
        assert (fig_dir / name).stat().st_size > 0

    saved = json.loads((out_dir / "drift_report.json").read_text(encoding="utf-8"))
    by_name = {s["name"]: s for s in saved["scenarios"]}
    assert set(by_name) == set(config["monitoring"]["scenarios"])
    assert saved["reconstruction_check"]["ok"]
    assert by_name["control"]["decision"]["level"] == "OK"
    assert by_name["rezagos_faltantes"]["decision"]["rule"] == "R1_pipeline"
    assert by_name["reloj_desfasado"]["quality"]["hour_profile"]["shift_hours"] == 6
    assert by_name["reloj_desfasado"]["decision"]["rule"] == "R1_pipeline"
    assert by_name["estaciones_nuevas"]["decision"]["rule"] == "R2_catalogo"
    assert by_name["rezagos_escala"]["levels"]["drift"] == "ALERTA"
    assert by_name["rezagos_escala"]["decision"]["rule"] == "R1_pipeline"  # rezagos no enteros
    assert by_name["control"]["quality"]["count_domain_violation_rate"] == 0.0
    summary = pd.read_csv(out_dir / "drift_summary.csv")
    assert list(summary["escenario"]) == list(config["monitoring"]["scenarios"])
    md = (out_dir / "drift_report.md").read_text(encoding="utf-8")
    assert "| control |" in md and "Umbrales" in md
    assert report["windows"]


def test_unknown_scenario_raises(synthetic):
    with pytest.raises(ValueError):
        run_monitoring(synthetic["config"], use_mlflow=False, scenario_names=["no_existe"],
                       features=synthetic["features"], model=synthetic["model"])
