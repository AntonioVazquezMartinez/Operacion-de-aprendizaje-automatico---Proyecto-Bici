"""Transformador de preprocesamiento: se ajusta solo con train y nunca entrega NaN."""

from __future__ import annotations

import numpy as np
import pytest

from src.features import build_preprocessor, feature_columns, model_input_columns


@pytest.fixture(scope="module")
def fitted(session_config, model_data):
    pre = build_preprocessor(session_config).fit(model_data["X_train"], model_data["y_train"])
    return pre


def test_output_columns(fitted, session_config, model_data):
    names = list(fitted.get_feature_names_out())
    assert names == model_input_columns(session_config)
    assert fitted.transform(model_data["X_test"]).shape == (len(model_data["X_test"]), len(names))


def test_scaler_statistics_come_only_from_train(fitted, session_config, model_data):
    numeric = feature_columns(session_config)["numeric"]
    scaler = fitted.named_transformers_["num"].named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, model_data["X_train"][numeric].mean().to_numpy())
    # Con todo el conjunto (train + test) la media del rezago sería distinta: el ajuste no ve test.
    lag = numeric.index("demand_lag_24h")
    full_mean = np.concatenate([model_data["X_train"]["demand_lag_24h"], model_data["X_test"]["demand_lag_24h"]]).mean()
    assert scaler.mean_[lag] != pytest.approx(full_mean)


def test_preprocessor_fit_only_on_train(fitted, model_data):
    names = list(fitted.get_feature_names_out())
    lag_idx = names.index("demand_lag_24h")
    train_t = fitted.transform(model_data["X_train"])
    assert float(np.mean(train_t[:, lag_idx])) == pytest.approx(0.0, abs=1e-6)
    assert float(np.std(train_t[:, lag_idx])) == pytest.approx(1.0, abs=1e-6)
    assert fitted.transform(model_data["X_test"]).shape[1] == len(names)


def test_no_nan_in_output_even_with_nan_inputs(fitted, model_data):
    X = model_data["X_test"].copy().astype({"capacity": float})
    X.iloc[0, X.columns.get_loc("demand_lag_24h")] = np.nan
    X.iloc[1, X.columns.get_loc("capacity")] = np.nan
    X.iloc[2, X.columns.get_loc("sin_hour")] = np.nan
    out = fitted.transform(X)
    assert not np.isnan(out).any()


def test_unknown_station_is_encoded_without_nan(fitted, model_data):
    X = model_data["X_test"].head(3).copy()
    X["Origen_Id"] = 987654                   # estación que no estaba en train
    assert np.isfinite(fitted.transform(X)).all()


def test_binary_columns_pass_through(fitted, model_data):
    names = list(fitted.get_feature_names_out())
    out = fitted.transform(model_data["X_test"])
    for col in ("is_weekend", "is_holiday"):
        np.testing.assert_array_equal(out[:, names.index(col)], model_data["X_test"][col].to_numpy())
