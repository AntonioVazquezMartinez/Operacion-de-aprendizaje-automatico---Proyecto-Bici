"""Prueba de integración de extremo a extremo (pytest):

    carga de datos -> limpieza -> características -> partición -> entrenamiento
    -> predicción -> métricas -> paquete de despliegue -> servicio -> verificación

No usa MLflow, por lo que corre en cualquier entorno (incluido el contenedor y GitHub Actions).
"""

import numpy as np
import pandas as pd

from src.features import model_input_columns
from src.modeling.pipeline import build_pipeline, evaluate
from src.serving.service import ModelService
from src.serving.verify import verify


def test_pipeline_end_to_end(config, features, splits):
    target = config["features"]["target"]
    cols = model_input_columns(config)
    assert not features[cols].isna().any().any(), "la matriz de características no debe tener faltantes"

    model = build_pipeline(config, "hist_gradient_boosting", {"max_iter": 50})
    model.fit(splits["train"][cols], splits["train"][target])
    preds = model.predict(splits["test"][cols])
    metrics = evaluate(splits["test"][target], preds, "test")

    assert len(preds) == len(splits["test"])
    assert np.isfinite(preds).all()
    assert set(metrics) == {"test_mae", "test_rmse", "test_r2", "test_poisson_deviance"}
    assert 0 <= metrics["test_mae"] <= metrics["test_rmse"]

    dummy = build_pipeline(config, "dummy_mean").fit(splits["train"][cols], splits["train"][target])
    dummy_mae = evaluate(splits["test"][target], dummy.predict(splits["test"][cols]), "d")["d_mae"]
    # Con datos sintéticos aleatorios no se exige ganarle al dummy, solo un error del mismo orden.
    assert metrics["test_mae"] < 2 * dummy_mae


def test_served_predictions_match_offline_pipeline(config, features, trained_model, bundle_dir):
    """Sin training-serving skew: la API calcula las mismas variables y predicciones que el pipeline."""
    service = ModelService(bundle_dir)
    window = service.predictable_window()
    rows = features[(features["fecha_hora"] >= pd.Timestamp(window["desde"]))
                    & (features["fecha_hora"] <= pd.Timestamp(window["hasta"]))].head(50)
    assert len(rows) > 0
    instances = [{"station_id": int(r.Origen_Id), "fecha_hora": r.fecha_hora} for r in rows.itertuples()]

    served = service.build_features(instances)
    cols = model_input_columns(config)
    np.testing.assert_allclose(served[cols].to_numpy(dtype=float), rows[cols].to_numpy(dtype=float), atol=1e-12)

    online = [p["demand_pred"] for p in service.predict(instances)]
    offline = np.clip(trained_model.predict(rows[cols]), 0, None)
    np.testing.assert_allclose(online, offline, rtol=1e-12)


def test_bundle_is_reproducible(bundle_dir):
    result = verify(bundle_dir)
    assert result["model_hash_ok"]
    assert result["predictions_ok"], result["reference_max_abs_diff"]
    assert result["reproducible"]
