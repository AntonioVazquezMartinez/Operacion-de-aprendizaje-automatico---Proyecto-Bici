"""Paquete de despliegue (serving bundle) del modelo champion.

El servicio no consulta la base de MLflow en producción: usa una copia exportada del modelo
registrado, identificada por su URI del Model Registry (`models:/<nombre>/<versión>`) y su hash.
El paquete vive en `models/champion/` y contiene:

    model.joblib                 Pipeline de scikit-learn (preprocesamiento + modelo)
    model_card.json              Origen del modelo: URI, versión, run, métricas, hashes, versiones
    feature_store.csv.gz         Demanda horaria reciente por estación (para calcular rezagos)
    stations.csv                 Catálogo de estaciones: ubicación y capacidad
    reference_predictions.csv    Entradas fijas y su predicción esperada (prueba de reproducibilidad)

Estas funciones no dependen de MLflow para poder usarlas en pruebas y en el contenedor.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import Config
from src.features import model_input_columns

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"
STORE_FILE = "feature_store.csv.gz"
STATIONS_FILE = "stations.csv"
REFERENCE_FILE = "reference_predictions.csv"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def serving_settings(config: Config) -> Dict[str, Any]:
    """Valores de la sección `serving` de config.yaml con valores por defecto."""
    defaults = {"bundle_dir": "models/champion", "feature_store_hours": 360, "reference_rows": 1000,
                "max_instances": 1000}
    return {**defaults, **(config.get("serving") or {})}


def build_feature_store(df_features: pd.DataFrame, target: str, hours: int) -> pd.DataFrame:
    """Últimas `hours` horas de demanda por estación (la cuadrícula es horaria y continua)."""
    end = df_features["fecha_hora"].max()
    start = end - pd.Timedelta(hours=hours - 1)
    store = df_features.loc[df_features["fecha_hora"] >= start, ["Origen_Id", "fecha_hora", target]]
    store = store.rename(columns={target: "demand_count"})
    store["Origen_Id"] = store["Origen_Id"].astype(int)
    return store.sort_values(["Origen_Id", "fecha_hora"]).reset_index(drop=True)


def export_bundle(
    config: Config,
    model: Any,
    df_features: pd.DataFrame,
    df_stations: pd.DataFrame,
    out_dir: str | Path,
    model_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Escribe el paquete de despliegue y regresa la ruta de model_card.json."""
    settings = serving_settings(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = config["features"]["target"]
    inputs = model_input_columns(config)

    # 1. Modelo
    model_path = out_dir / MODEL_FILE
    joblib.dump(model, model_path, compress=3)

    # 2. Feature store con la historia reciente que necesitan los rezagos
    store = build_feature_store(df_features, target, int(settings["feature_store_hours"]))
    store.to_csv(out_dir / STORE_FILE, index=False, date_format="%Y-%m-%d %H:%M:%S")

    # 3. Catálogo de estaciones activas
    active = set(store["Origen_Id"].unique())
    stations = df_stations.loc[df_stations["id"].isin(active), ["id", "latitude", "longitude", "capacity"]]
    stations.sort_values("id").to_csv(out_dir / STATIONS_FILE, index=False)

    # 4. Predicciones de referencia: filas fijas (semilla) y su salida esperada
    n_ref = min(int(settings["reference_rows"]), len(df_features))
    reference = df_features.sample(n=n_ref, random_state=config["seed"]).sort_index()
    reference = reference[["Origen_Id", "fecha_hora", target] + [c for c in inputs if c != "Origen_Id"]].copy()
    reference["expected_pred"] = np.clip(model.predict(reference[inputs]), 0, None)
    # Precisión completa: la hora 15 y la 21 tienen el mismo seno salvo por ~1e-16, y redondear
    # el CSV cambiaría la predicción (pandas lee los floats con precisión completa).
    reference.to_csv(out_dir / REFERENCE_FILE, index=False)

    # 5. Ficha del modelo
    store_end = store["fecha_hora"].max()
    card = {
        **(model_meta or {}),
        "model_file": MODEL_FILE,
        "model_sha256": sha256_file(model_path),
        "feature_store_sha256": sha256_file(out_dir / STORE_FILE),
        "reference_sha256": sha256_file(out_dir / REFERENCE_FILE),
        "input_columns": inputs,
        # Tipos con los que se entrenó el modelo: el servicio convierte las entradas a estos tipos.
        "input_dtypes": {c: str(df_features[c].dtype) for c in inputs},
        "target": target,
        "lag_hours": config["features"]["lag_hours"],
        "same_hour_mean_days": config["features"].get("same_hour_mean_days"),
        "feature_store_start": str(store["fecha_hora"].min()),
        "feature_store_end": str(store_end),
        "num_stations": int(len(stations)),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "export_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    card_path = out_dir / CARD_FILE
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return card_path


def load_card(bundle_dir: str | Path) -> Dict[str, Any]:
    return json.loads((Path(bundle_dir) / CARD_FILE).read_text(encoding="utf-8"))
