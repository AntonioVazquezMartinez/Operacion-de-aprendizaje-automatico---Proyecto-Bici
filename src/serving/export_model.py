"""Exporta una versión del MLflow Model Registry como paquete de despliegue (models/champion/).

Carga `models:/mibici-demand-forecaster@champion` (o la versión indicada), lo guarda junto con
el feature store, el catálogo de estaciones y las predicciones de referencia, y registra el
paquete como artefactos de una corrida del experimento de despliegue en MLflow.

Uso:
    python -m src.serving.export_model                 # alias champion
    python -m src.serving.export_model --version 1     # versión específica
    python -m src.serving.export_model --no-mlflow-log # no crear la corrida de despliegue
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn  # noqa: F401  (carga explícita del flavor de scikit-learn)
import pandas as pd

from src.config import PROJ_ROOT, Config, base_arg_parser, load_config
from src.features import load_features
from src.serving.bundle import export_bundle, load_card, serving_settings
from src.tracking import setup_mlflow


def export_registered_model(config: Config, alias: Optional[str] = None, version: Optional[str] = None,
                            log_to_mlflow: bool = True) -> Path:
    reg = config["mlflow"]
    name = reg["registered_model_name"]
    mlflow.set_tracking_uri(config.tracking_uri)
    client = mlflow.MlflowClient()
    if version:
        mv = client.get_model_version(name, str(version))
        alias = None
    else:
        alias = alias or reg["champion_alias"]
        mv = client.get_model_version_by_alias(name, alias)

    model_uri = f"models:/{name}/{mv.version}"
    print(f"Cargando {model_uri}" + (f" (alias {alias})" if alias else ""))
    model = mlflow.sklearn.load_model(model_uri)

    meta = {
        "registered_model_name": name,
        "model_version": str(mv.version),
        "model_alias": alias,
        "model_uri": model_uri,
        "model_uri_alias": f"models:/{name}@{alias}" if alias else None,
        "training_run_id": mv.run_id,
        "mlflow_source": mv.source,
        "algorithm": mv.tags.get("algorithm"),
        "hyperparameters": mv.tags.get("hyperparameters"),
        "training_dataset_sha256": mv.tags.get("dataset_sha256"),
        "registry_metrics": {k: float(mv.tags[k]) for k in ("val_mae", "test_mae", "test_rmse", "test_r2")
                             if k in mv.tags},
    }

    settings = serving_settings(config)
    out_dir = Path(settings["bundle_dir"])
    out_dir = out_dir if out_dir.is_absolute() else PROJ_ROOT / out_dir
    df_features = load_features(config)
    df_stations = pd.read_csv(config.file("raw_dir", "stations_file"))
    card_path = export_bundle(config, model, df_features, df_stations, out_dir, meta)
    card = load_card(out_dir)
    print(f"Paquete de despliegue en {out_dir}")
    print(f"  model.joblib sha256 = {card['model_sha256']}")
    print(f"  feature store: {card['feature_store_start']} a {card['feature_store_end']}, "
          f"{card['num_stations']} estaciones")

    if log_to_mlflow:
        setup_mlflow(config, (config.get("serving") or {}).get("experiment", "MiBici_Despliegue"))
        with mlflow.start_run(run_name=f"05_Export_{name}_v{mv.version}"):
            mlflow.set_tags({"phase": "Entrega Final", "stage": "deployment", "role": "ML Engineer",
                             "model_uri": model_uri})
            mlflow.log_params({"registered_model_name": name, "model_version": mv.version,
                               "model_alias": alias or "", "model_sha256": card["model_sha256"],
                               "feature_store_sha256": card["feature_store_sha256"],
                               "feature_store_end": card["feature_store_end"]})
            mlflow.log_artifacts(str(out_dir), artifact_path="serving_bundle")
        client.set_model_version_tag(name, mv.version, "serving_bundle_sha256", card["model_sha256"])
        print("Paquete registrado en MLflow (experimento de despliegue) y etiquetado en el Model Registry.")
    return card_path


def main() -> None:
    parser = base_arg_parser("Exporta el modelo registrado como paquete de despliegue")
    parser.add_argument("--alias", help="Alias del Model Registry (por defecto: champion)")
    parser.add_argument("--version", help="Versión específica del Model Registry")
    parser.add_argument("--no-mlflow-log", action="store_true", help="No registrar el paquete en MLflow")
    args = parser.parse_args()
    export_registered_model(load_config(args.config), args.alias, args.version, not args.no_mlflow_log)


if __name__ == "__main__":
    main()
