"""Predicción por lotes con el modelo registrado en el MLflow Model Registry.

Por defecto carga `models:/mibici-demand-forecaster@champion` y predice el periodo de prueba,
con lo que se comprueba que el modelo registrado reproduce las métricas reportadas.

Uso:
    python -m src.modeling.predict
    python -m src.modeling.predict --alias challenger
    python -m src.modeling.predict --input data/external/nuevas_horas.csv --output predicciones.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import mlflow
import mlflow.sklearn  # noqa: F401  (carga explícita del flavor de scikit-learn)
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import Config, base_arg_parser, load_config
from src.features import load_features, model_input_columns, temporal_split
from src.modeling.pipeline import evaluate


def load_registered_model(config: Config, alias: Optional[str] = None) -> Tuple[Pipeline, str]:
    mlflow.set_tracking_uri(config.tracking_uri)
    alias = alias or config["mlflow"]["champion_alias"]
    uri = f"models:/{config['mlflow']['registered_model_name']}@{alias}"
    return mlflow.sklearn.load_model(uri), uri


def predict(config: Config, input_path: Optional[str] = None, output_path: Optional[str] = None,
            alias: Optional[str] = None) -> pd.DataFrame:
    model, uri = load_registered_model(config, alias)
    target = config["features"]["target"]

    if input_path:
        df = pd.read_csv(input_path)
    else:
        df = load_features(config)
        df = df[temporal_split(df, config) == "test"].reset_index(drop=True)

    missing = set(model_input_columns(config)) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas de entrada: {sorted(missing)}")

    predictions = np.clip(model.predict(df[model_input_columns(config)]), 0, None)
    keep = [c for c in ("Origen_Id", "fecha_hora", target) if c in df.columns]
    out = df[keep].copy()
    out["demand_pred"] = predictions

    if target in df.columns:
        metrics = evaluate(df[target], predictions, "pred")
        print(f"{uri}: MAE {metrics['pred_mae']:.4f} | RMSE {metrics['pred_rmse']:.4f} | R2 {metrics['pred_r2']:.4f}")

    alias_name = alias or config["mlflow"]["champion_alias"]
    output = Path(output_path) if output_path else config.path("processed_dir") / f"predicciones_{alias_name}.csv"
    out.to_csv(output, index=False)
    print(f"{len(out)} predicciones guardadas en {output}")
    return out


def main() -> None:
    parser = base_arg_parser("Predicción por lotes con el modelo registrado")
    parser.add_argument("--alias", help="Alias del Model Registry (por defecto: champion)")
    parser.add_argument("--input", help="CSV con las columnas de entrada; por defecto, el periodo de prueba")
    parser.add_argument("--output", help="Ruta del CSV de salida")
    args = parser.parse_args()
    predict(load_config(args.config), args.input, args.output, args.alias)


if __name__ == "__main__":
    main()
