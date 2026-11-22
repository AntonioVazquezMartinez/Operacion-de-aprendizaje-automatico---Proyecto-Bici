"""Métricas de evaluación (`src.modeling.pipeline.evaluate`) con valores calculados a mano."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.modeling.pipeline import evaluate


def test_hand_computed_example():
    """y = [1, 2, 3, 4], ŷ = [1, 2, 3, 6]: un solo error de 2 unidades.

    MAE = 2/4 = 0.5; MSE = 4/4 = 1 -> RMSE = 1; SST = 2.25 + 0.25 + 0.25 + 2.25 = 5 -> R² = 1 - 4/5 = 0.2.
    Devianza de Poisson: solo el último término es distinto de cero, 2·(4·ln(4/6) - 4 + 6), entre 4.
    """
    m = evaluate(pd.Series([1, 2, 3, 4]), np.array([1.0, 2.0, 3.0, 6.0]), "test")
    assert m["test_mae"] == pytest.approx(0.5)
    assert m["test_rmse"] == pytest.approx(1.0)
    assert m["test_r2"] == pytest.approx(0.2)
    assert m["test_poisson_deviance"] == pytest.approx(2 * (4 * math.log(4 / 6) + 2) / 4)


def test_second_hand_computed_example():
    """y = [0, 2, 4], ŷ = [1, 1, 4]: MAE = 2/3, RMSE = sqrt(2/3), SST = 8 -> R² = 1 - 2/8 = 0.75."""
    m = evaluate(np.array([0, 2, 4]), np.array([1.0, 1.0, 4.0]), "val")
    assert m["val_mae"] == pytest.approx(2 / 3)
    assert m["val_rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert m["val_r2"] == pytest.approx(0.75)
    # Con y = 0 el término de Poisson es 2·ŷ.
    expected_dev = (2 * 1.0 + 2 * (2 * math.log(2 / 1) - 2 + 1) + 0.0) / 3
    assert m["val_poisson_deviance"] == pytest.approx(expected_dev)


def test_perfect_prediction():
    metrics = evaluate(pd.Series([0, 1, 2]), np.array([0.0, 1.0, 2.0]), "val")
    assert metrics["val_mae"] == pytest.approx(0.0)
    assert metrics["val_rmse"] == pytest.approx(0.0)
    assert metrics["val_r2"] == pytest.approx(1.0)
    assert set(metrics) == {"val_mae", "val_rmse", "val_r2", "val_poisson_deviance"}


def test_negative_predictions_are_clipped_to_zero():
    """Una demanda negativa no existe: -3 se recorta a 0 y el error queda en 0."""
    m = evaluate(np.array([0, 1]), np.array([-3.0, 1.0]), "test")
    assert m["test_mae"] == pytest.approx(0.0)
    assert m["test_rmse"] == pytest.approx(0.0)


def test_zero_prediction_keeps_poisson_deviance_finite():
    m = evaluate(np.array([1, 2]), np.array([0.0, 2.0]), "test")
    assert math.isfinite(m["test_poisson_deviance"])
    assert m["test_mae"] == pytest.approx(0.5)


@pytest.mark.parametrize("prefix", ["val", "test", "pred"])
def test_keys_are_prefixed_and_float(prefix):
    m = evaluate([1, 2, 3], [1.5, 2.0, 2.5], prefix)
    assert set(m) == {f"{prefix}_{k}" for k in ("mae", "rmse", "r2", "poisson_deviance")}
    assert all(isinstance(v, float) for v in m.values())
    assert m[f"{prefix}_rmse"] >= m[f"{prefix}_mae"]
