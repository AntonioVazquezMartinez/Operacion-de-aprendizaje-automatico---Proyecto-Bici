"""Exportación del modelo champion del Model Registry a una carpeta autocontenida para el servicio.

Genera (rutas en la sección `serving` de config.yaml):
    - models/champion/                  carpeta de modelo MLflow (MLmodel + modelo serializado) que se
                                        carga con `mlflow.sklearn.load_model` sin la base mlruns.db;
                                        incluye el catálogo mínimo de estaciones que usa el servicio y el
                                        feature store (feature_store.csv.gz: demanda de las últimas
                                        `serving.feature_store_hours` horas por estación).
    - models/champion_metadata.json     versión registrada, URI, hiperparámetros, hashes, métricas,
                                        columnas de entrada y versiones de librerías.
    - models/reference_predictions.csv  muestra fija del conjunto de prueba con la predicción de
                                        referencia (patrón oro para verificar la reproducibilidad).
    - models/reference_feature_store_predictions.csv
                                        muestra fija de (estación, hora) de la ventana predecible del
                                        feature store con los rezagos y la predicción calculados fuera
                                        de línea (patrón oro de /predict sin rezagos).

La carpeta models/champion/ se versiona en Git (unos 6 MB), así que `docker build` funciona en un clon limpio.

Uso:
    python -m src.modeling.export                 # exporta el alias champion
    python -m src.modeling.export --alias challenger
    python -m src.modeling.export --verify-only   # solo recarga la carpeta y compara con la referencia
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn  # noqa: F401  (carga explícita del flavor de scikit-learn)
import numpy as np
import pandas as pd
import sklearn
import yaml

from src.config import PROJ_ROOT, Config, base_arg_parser, load_config, set_global_seeds
from src.features import MiBiciFeatureEngineer, lag_columns, load_features, model_input_columns, temporal_split
from src.serving.feature_store import (
    STATION_KEY,
    TIME_KEY,
    FeatureStore,
    build_feature_store,
    compute_lags,
    lag_offsets_hours,
    write_feature_store,
)
from src.tracking import calculate_sha256

# Los CSV se escriben con el formato por omisión de pandas (repr más corto que reproduce exactamente
# cada float64), así que la lectura recupera los mismos valores bit a bit (precisión completa).
STATION_COLUMNS = ["id", "latitude", "longitude", "capacity"]


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJ_ROOT / path


def serving_paths(config: Config) -> Dict[str, Path]:
    """Rutas absolutas de los artefactos de servicio definidos en `serving`."""
    serving = config["serving"]
    model_dir = _resolve(serving["model_dir"])
    return {
        "model_dir": model_dir,
        "metadata": _resolve(serving["metadata_file"]),
        "reference": _resolve(serving["reference_predictions_file"]),
        "stations": model_dir / serving.get("stations_file", "stations.csv"),
        "feature_store": model_dir / serving["feature_store_file"],
        "feature_store_reference": _resolve(serving["feature_store_reference_file"]),
    }


def reference_columns(config: Config) -> List[str]:
    """Columnas del CSV de referencia: identificadores, entradas del modelo (sin repetir), real y predicción."""
    target = config["features"]["target"]
    inputs = [c for c in model_input_columns(config) if c not in ("Origen_Id", "fecha_hora")]
    return ["Origen_Id", "fecha_hora", *inputs, target, "demand_pred"]


def predict_clipped(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predicción con el mismo posprocesamiento que src.modeling.predict (sin valores negativos)."""
    return np.clip(np.asarray(model.predict(X), dtype=float), 0, None)


def _pickled_model_file(model_dir: Path) -> Path:
    """Archivo serializado del modelo según la sección `flavors.sklearn` del archivo MLmodel."""
    with open(model_dir / "MLmodel", encoding="utf-8") as f:
        mlmodel = yaml.safe_load(f)
    return model_dir / mlmodel["flavors"]["sklearn"]["pickled_model"]


def normalize_text_files(model_dir: Path, skip: Path) -> List[str]:
    """Convierte CRLF -> LF en los archivos de texto de la carpeta MLflow (MLmodel, YAML, JSON, requirements).

    MLflow los escribe con el fin de línea de la plataforma (CRLF en Windows). La carpeta se versiona en Git con
    `eol=lf` (.gitattributes) y DVC guarda su hash en dvc.lock: con LF el contenido es el mismo byte a byte en
    Windows y en Linux. El modelo serializado (`skip`) no se toca. Regresa los archivos modificados.
    """
    changed = []
    for path in sorted(p for p in model_dir.rglob("*") if p.is_file() and p != skip):
        data = path.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue  # binario
        if b"\r\n" in data:
            path.write_bytes(data.replace(b"\r\n", b"\n"))
            changed.append(path.relative_to(model_dir).as_posix())
    return changed


def feature_store_reference(config: Config, model: Any, df: pd.DataFrame, df_stations: pd.DataFrame,
                            store: FeatureStore) -> pd.DataFrame:
    """Muestra fija de (estación, hora) de la ventana predecible con rezagos y predicción fuera de línea.

    - Horas dentro del periodo de datos: la fila de la matriz procesada (pipeline de entrenamiento, rezagos de
      `add_lag_features` sobre el historial completo), sin pasar por el feature store.
    - Horas posteriores al último dato (hasta +24 h): las mismas funciones del pipeline (calendario, estación y
      `add_lag_features`) sobre la serie de la matriz procesada, con la demanda futura desconocida.
    """
    target = config["features"]["target"]
    inputs = model_input_columns(config)
    times = pd.to_datetime(df[TIME_KEY])
    in_data = df.loc[(times >= store.window_start) & (times <= store.window_end)].copy()
    in_data["in_data_period"] = 1

    end = times.max()
    since = end - pd.Timedelta(hours=max(lag_offsets_hours(config)))
    future = compute_lags(df.loc[times >= since, [STATION_KEY, TIME_KEY, target]], config, store.window_end)
    future = future[future[TIME_KEY] > end].reset_index(drop=True)
    future = MiBiciFeatureEngineer.add_calendar_features(future)
    future = MiBiciFeatureEngineer.add_station_features(future, df_stations)
    future[target] = np.nan
    future["in_data_period"] = 0

    keep = [*inputs, TIME_KEY, target, "in_data_period"]
    candidates = pd.concat([in_data[keep], future[keep]], ignore_index=True)
    n_rows = min(int(config["serving"]["feature_store_reference_rows"]), len(candidates))
    sample = candidates.sample(n=n_rows, random_state=config["seed"]).sort_values([TIME_KEY, STATION_KEY])
    sample = sample.reset_index(drop=True)
    sample["demand_pred"] = predict_clipped(model, sample[inputs])
    sample[target] = sample[target].astype("Int64")  # vacío en las horas posteriores al último dato
    return sample[[STATION_KEY, TIME_KEY, *lag_columns(config), target, "in_data_period", "demand_pred"]]


def export_feature_store(config: Config, model: Any, df: pd.DataFrame, df_stations: pd.DataFrame,
                         paths: Dict[str, Path]) -> Dict[str, Any]:
    """Escribe feature_store.csv.gz y su predicción de referencia; regresa sus entradas de los metadatos."""
    hours = int(config["serving"]["feature_store_hours"])
    store_df = build_feature_store(df, config, hours)
    write_feature_store(store_df, paths["feature_store"])
    store = FeatureStore(store_df, config)

    reference = feature_store_reference(config, model, df, df_stations, store)
    paths["feature_store_reference"].parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(paths["feature_store_reference"], index=False, lineterminator="\n")
    info = store.info()
    return {
        "feature_store": {
            "file": paths["feature_store"].relative_to(paths["model_dir"]).as_posix(),
            "sha256": calculate_sha256(paths["feature_store"]),
            **info,
            "lag_columns": store.lag_names,
            "lag_offsets_hours": sorted(set(lag_offsets_hours(config))),
        },
        "feature_store_reference_predictions": {
            "file": paths["feature_store_reference"].name,
            "rows": int(len(reference)),
            "rows_in_data_period": int(reference["in_data_period"].sum()),
            "rows_after_data_period": int((reference["in_data_period"] == 0).sum()),
            "seed": int(config["seed"]),
            "sha256": calculate_sha256(paths["feature_store_reference"]),
        },
    }


def _library_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "mlflow": mlflow.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def export_champion(config: Config, alias: Optional[str] = None) -> Dict[str, Any]:
    """Descarga la versión del Registry apuntada por `alias` y escribe carpeta, metadatos y referencia."""
    mlflow.set_tracking_uri(config.tracking_uri)
    reg = config["mlflow"]
    name = reg["registered_model_name"]
    alias = alias or reg["champion_alias"]
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    version_uri = f"models:/{name}/{mv.version}"
    paths = serving_paths(config)
    print(f"Exportando {name}@{alias} -> v{mv.version} ({version_uri})")

    # 1. Carpeta de modelo MLflow (copia exacta de los artefactos registrados).
    model_dir = paths["model_dir"]
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=version_uri, dst_path=tmp))
        if not (local / "MLmodel").exists():
            raise FileNotFoundError(f"La descarga de {version_uri} no contiene un archivo MLmodel: {local}")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local, model_dir)
    normalized = normalize_text_files(model_dir, _pickled_model_file(model_dir))
    if normalized:
        print(f"Fin de línea LF en: {', '.join(normalized)}")

    # 2. Catálogo mínimo de estaciones conocidas por el modelo (las que aparecen en la matriz procesada).
    df = load_features(config)
    raw_stations = pd.read_csv(config.file("raw_dir", "stations_file"))
    known = set(df["Origen_Id"].unique())
    stations = raw_stations.loc[raw_stations["id"].isin(known), STATION_COLUMNS]
    stations = stations.sort_values("id").reset_index(drop=True)
    # Fin de línea fijo "\n": con el de la plataforma (CRLF en Windows) el SHA-256 de los metadatos no
    # coincide con el archivo que Git entrega en Linux (core.autocrlf normaliza a LF).
    stations.to_csv(paths["stations"], index=False, lineterminator="\n")

    # 3. Muestra fija del conjunto de prueba con la predicción de referencia del modelo exportado.
    target = config["features"]["target"]
    inputs = model_input_columns(config)
    test = df[temporal_split(df, config) == "test"].reset_index(drop=True)
    n_rows = min(int(config["serving"]["reference_sample_rows"]), len(test))
    sample = test.sample(n=n_rows, random_state=config["seed"]).sort_values(["fecha_hora", "Origen_Id"])
    sample = sample.reset_index(drop=True)
    model = mlflow.sklearn.load_model(str(model_dir))
    sample["demand_pred"] = predict_clipped(model, sample[inputs])
    reference = sample[reference_columns(config)]
    paths["reference"].parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(paths["reference"], index=False, lineterminator="\n")

    # 4. Feature store (últimas horas de demanda por estación) y su predicción de referencia.
    feature_store_md = export_feature_store(config, model, df, raw_stations, paths)

    # 5. Metadatos.
    run = client.get_run(mv.run_id)
    run_params, run_metrics = run.data.params, run.data.metrics
    data_path = config.file("processed_dir", "features_file")
    dataset_sha = calculate_sha256(data_path)
    registered_sha = mv.tags.get("dataset_sha256")
    if registered_sha and registered_sha != dataset_sha:
        print(f"Aviso: el SHA-256 del dataset actual ({dataset_sha}) no coincide con el registrado "
              f"en el modelo ({registered_sha}).")
    model_file = _pickled_model_file(model_dir)
    aliases = sorted(client.get_model_version(name, mv.version).aliases)
    metadata = {
        "registered_model_name": name,
        "version": int(mv.version),
        "alias": alias,
        "aliases": aliases,
        "model_uri": version_uri,
        "alias_uri": f"models:/{name}@{alias}",
        "run_id": mv.run_id,
        "algorithm": mv.tags.get("algorithm", run_params.get("model")),
        "estimator_class": run_params.get("estimator_class"),
        "hyperparameters": json.loads(mv.tags.get("hyperparameters", "{}")),
        "model_file": model_file.relative_to(model_dir).as_posix(),
        "model_sha256": calculate_sha256(model_file),
        "dataset_file": data_path.name,
        "dataset_sha256": dataset_sha,
        "dataset_sha256_matches_registry": registered_sha == dataset_sha if registered_sha else None,
        "stations_file": paths["stations"].relative_to(model_dir).as_posix(),
        "stations_sha256": calculate_sha256(paths["stations"]),
        "num_stations": int(len(stations)),
        "metrics": {k: run_metrics[k] for k in sorted(run_metrics) if k.startswith(("val_", "test_"))},
        "input_columns": inputs,
        "target": target,
        "prediction_postprocessing": "clip(prediccion, 0, None)",
        "data_period": {
            "start": run_params.get("period_start"),
            "end": run_params.get("period_end"),
            "train_val_end": run_params.get("train_val_end"),
            "test_start": run_params.get("test_start"),
            "rows_train": int(run_params["rows_train"]) if "rows_train" in run_params else None,
            "rows_val": int(run_params["rows_val"]) if "rows_val" in run_params else None,
            "rows_test": int(run_params["rows_test"]) if "rows_test" in run_params else None,
            "months": list(config["data"]["months"]),
        },
        "reference_predictions": {
            "file": paths["reference"].name,
            "rows": int(len(reference)),
            "seed": int(config["seed"]),
            "sha256": calculate_sha256(paths["reference"]),
        },
        **feature_store_md,
        "library_versions": _library_versions(),
        "platform": platform.platform(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # newline="\n": el mismo archivo (y el mismo hash en dvc.lock) en Windows y en Linux.
    with open(paths["metadata"], "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    fs = metadata["feature_store"]
    print(f"Modelo: {model_dir} (SHA-256 {metadata['model_sha256']})")
    print(f"Metadatos: {paths['metadata']}")
    print(f"Referencia: {paths['reference']} ({len(reference)} filas)")
    print(f"Feature store: {fs['file']} {fs['start']} a {fs['end']} ({fs['hours']} h, {fs['num_stations']} "
          f"estaciones, SHA-256 {fs['sha256']}); ventana predecible {fs['predictable_window']['start']} a "
          f"{fs['predictable_window']['end']}")
    print(f"Referencia del feature store: {paths['feature_store_reference']} "
          f"({metadata['feature_store_reference_predictions']['rows']} filas)")
    return metadata


def verify_reference(config: Config) -> float:
    """Recarga la carpeta exportada (sin MLflow Registry) y compara con las predicciones de referencia.

    Regresa la diferencia máxima absoluta entre la predicción actual y la de referencia.
    """
    paths = serving_paths(config)
    model = mlflow.sklearn.load_model(str(paths["model_dir"]))
    reference = pd.read_csv(paths["reference"])
    predictions = predict_clipped(model, reference[model_input_columns(config)])
    max_abs_diff = float(np.max(np.abs(predictions - reference["demand_pred"].to_numpy(dtype=float))))
    print(f"Verificación: {len(reference)} filas, diferencia máxima absoluta = {max_abs_diff:.3e} "
          f"(Python {platform.python_version()}, scikit-learn {sklearn.__version__}, {sys.platform})")
    return max_abs_diff


def main() -> None:
    parser = base_arg_parser("Exporta el modelo champion del Model Registry para el servicio")
    parser.add_argument("--alias", help="Alias del Model Registry (por defecto: champion)")
    parser.add_argument("--verify-only", action="store_true",
                        help="No exporta; recarga models/champion y compara con la referencia")
    args = parser.parse_args()
    config = load_config(args.config)
    set_global_seeds(config["seed"])
    if not args.verify_only:
        export_champion(config, args.alias)
    verify_reference(config)


if __name__ == "__main__":
    main()
