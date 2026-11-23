"""Compara el reentrenamiento en Linux (volumen /work) con los artefactos del entrenamiento en Windows.

Entradas de Windows (copiadas a la imagen desde el repositorio): reports/model_comparison.csv,
models/champion_metadata.json, models/reference_predictions.csv y models/champion/.
Entradas de Linux: <work>/reports/model_comparison.csv, <work>/models/... generados por run_pipeline_linux.sh.

Salidas en --output: metricas_linux_vs_windows.csv (diferencia por modelo y métrica) y
comparacion_pipeline.json (hashes de datos y modelo, hiperparámetros del champion, predicciones de las
500 filas de referencia con el modelo reentrenado en Linux y con el modelo de Windows ejecutado en Linux).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path
from typing import Any, Dict

import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn

APP = Path(__file__).resolve().parents[1]
METRICS = ["val_mae", "val_rmse", "val_r2", "val_poisson_deviance",
           "test_mae", "test_rmse", "test_r2", "test_poisson_deviance"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_diff(pred: np.ndarray, ref: np.ndarray) -> Dict[str, Any]:
    diff = np.abs(pred - ref)
    same = sum(struct.pack("<d", float(a)) == struct.pack("<d", float(b)) for a, b in zip(pred, ref))
    return {"rows": int(len(ref)), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "rows_bitwise_identical": int(same)}


def predict(model_dir: Path, reference: pd.DataFrame, columns: list) -> np.ndarray:
    model = mlflow.sklearn.load_model(str(model_dir))
    return np.clip(np.asarray(model.predict(reference[columns]), dtype=float), 0, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", default="/work")
    parser.add_argument("--output", default="/work/comparison")
    args = parser.parse_args()
    work, out = Path(args.work), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Tabla de métricas por modelo.
    win = pd.read_csv(APP / "reports" / "model_comparison.csv").set_index("model")
    lin = pd.read_csv(work / "reports" / "model_comparison.csv").set_index("model")
    rows = []
    for model in win.index:
        for metric in METRICS:
            w, l = float(win.at[model, metric]), float(lin.at[model, metric])
            rows.append({"model": model, "metric": metric, "windows": w, "linux": l, "abs_diff": abs(l - w),
                         "rel_diff": abs(l - w) / abs(w) if w else None, "identical": w == l})
    table = pd.DataFrame(rows)
    table.to_csv(out / "metricas_linux_vs_windows.csv", index=False, float_format="%.17g")

    # 2. Hashes, hiperparámetros y predicciones de referencia.
    win_md = json.loads((APP / "models" / "champion_metadata.json").read_text(encoding="utf-8"))
    lin_md = json.loads((work / "models" / "champion_metadata.json").read_text(encoding="utf-8"))
    cols = win_md["input_columns"]
    ref = pd.read_csv(APP / "models" / "reference_predictions.csv")
    lin_ref = pd.read_csv(work / "models" / "reference_predictions.csv")
    expected = ref["demand_pred"].to_numpy(dtype=float)
    same_sample = bool(ref.drop(columns="demand_pred").equals(lin_ref.drop(columns="demand_pred")))

    result = {
        "linux_environment": {"platform": platform.platform(), "python": platform.python_version(),
                              "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                              "pandas": pd.__version__},
        "windows_environment": {"platform": win_md.get("platform"), **win_md.get("library_versions", {})},
        "dataset_sha256": {"windows": win_md["dataset_sha256"], "linux": lin_md["dataset_sha256"],
                           "identical": win_md["dataset_sha256"] == lin_md["dataset_sha256"]},
        "stations_sha256": {"windows": win_md["stations_sha256"], "linux": lin_md["stations_sha256"],
                            "identical": win_md["stations_sha256"] == lin_md["stations_sha256"]},
        "data_period": {"windows": win_md["data_period"], "linux": lin_md["data_period"],
                        "identical": win_md["data_period"] == lin_md["data_period"]},
        "champion": {"windows": {k: win_md[k] for k in ("algorithm", "version", "hyperparameters")},
                     "linux": {k: lin_md[k] for k in ("algorithm", "version", "hyperparameters")},
                     "same_algorithm": win_md["algorithm"] == lin_md["algorithm"],
                     "same_hyperparameters": win_md["hyperparameters"] == lin_md["hyperparameters"]},
        "champion_metrics": {"windows": win_md["metrics"], "linux": lin_md["metrics"],
                             "identical": win_md["metrics"] == lin_md["metrics"]},
        "model_pickle_sha256": {"windows": win_md["model_sha256"], "linux": lin_md["model_sha256"],
                                "identical": win_md["model_sha256"] == lin_md["model_sha256"]},
        "reference_sample": {"same_rows_and_inputs": same_sample,
                             "reference_csv_sha256": {"windows": sha256(APP / "models" / "reference_predictions.csv"),
                                                      "linux": sha256(work / "models" / "reference_predictions.csv")}},
        "predictions_on_windows_reference": {
            "linux_retrained_model": prediction_diff(predict(work / "models" / "champion", ref, cols), expected),
            "windows_model_run_on_linux": prediction_diff(predict(APP / "models" / "champion", ref, cols), expected),
        },
        "metrics_table": {"cells": int(len(table)), "identical_cells": int(table["identical"].sum()),
                          "max_abs_diff": float(table["abs_diff"].max()),
                          "max_rel_diff": float(table["rel_diff"].max())},
    }
    (out / "comparacion_pipeline.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ("data_period", "champion_metrics")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
