"""Binario original de la Fase 2 contra el champion reentrenado en la fase final.

`models/fase2_original/` es una copia sin modificar del paquete que la rama entrega-final (Jesús Rincón) exportó
del Model Registry de la Fase 2 (`model.joblib`: Pipeline de scikit-learn 1.9.1 serializado con joblib, entrenado
en otra máquina con Python 3.12). El script:

1. Verifica los SHA-256 del paquete contra su `model_card.json`.
2. Comprueba que el binario reproduce sus propias predicciones de referencia en este entorno (mismo binario,
   otro sistema y otra versión de Python): diferencia máxima y filas idénticas bit a bit.
3. Compara sus predicciones con las del champion reentrenado (`models:/mibici-demand-forecaster/1` de esta fase)
   sobre las 500 filas de `models/reference_predictions.csv`: diferencia máxima y media, correlación y MAE/RMSE de
   cada modelo contra la demanda real.

Uso:
    python scripts/compare_fase2_binary.py
    python scripts/compare_fase2_binary.py --output-dir reports/fase_final/reproducibilidad
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import sklearn

PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FASE2_DIR = PROJ_ROOT / "models" / "fase2_original"
DEFAULT_REFERENCE = PROJ_ROOT / "models" / "reference_predictions.csv"
DEFAULT_METADATA = PROJ_ROOT / "models" / "champion_metadata.json"
DEFAULT_OUTPUT_DIR = PROJ_ROOT / "reports" / "fase_final" / "reproducibilidad"
OUTPUT_NAME = "fase2_binario_vs_reentrenado"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJ_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    d = np.abs(a - b)
    return {"rows": int(len(d)), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "rows_bitwise_identical": int((a == b).sum())}


def error_metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    err = pred - actual
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def check_hashes(fase2_dir: Path, card: Dict[str, Any]) -> Dict[str, Any]:
    """SHA-256 de los archivos contra model_card.json (el CSV de referencia también con CRLF, ver la nota)."""
    model = (fase2_dir / card["model_file"]).read_bytes()
    reference = (fase2_dir / "reference_predictions.csv").read_bytes()
    lf = reference.replace(b"\r\n", b"\n")
    return {
        "model_file": card["model_file"],
        "model_sha256_card": card["model_sha256"],
        "model_sha256_file": sha256_bytes(model),
        "model_sha256_ok": sha256_bytes(model) == card["model_sha256"],
        "reference_sha256_card": card.get("reference_sha256"),
        "reference_sha256_file": sha256_bytes(reference),
        "reference_sha256_file_crlf": sha256_bytes(lf.replace(b"\n", b"\r\n")),
        "reference_sha256_ok": card.get("reference_sha256") in {sha256_bytes(lf), sha256_bytes(lf.replace(b"\n",
                                                                                                          b"\r\n"))},
        "reference_line_endings_in_git": "CRLF" if b"\r\n" in reference else "LF",
        "note": ("model_card.json guarda el SHA-256 del CSV de referencia escrito en Windows (CRLF); Git lo "
                 "almacena normalizado a LF. El archivo copiado es el de Git sin cambios: su contenido con CRLF "
                 "coincide con el hash de la ficha."),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    fase2_dir = Path(args.fase2_dir)
    card = json.loads((fase2_dir / "model_card.json").read_text(encoding="utf-8"))
    hashes = check_hashes(fase2_dir, card)
    model = joblib.load(fase2_dir / card["model_file"])
    inputs = list(card["input_columns"])
    # Tipos del entrenamiento de la Fase 2: el TargetEncoder aprendió Origen_Id como float64; con enteros cada
    # estación se trataría como categoría desconocida (lo documenta src/serving/service.py de entrega-final).
    dtypes = card["input_dtypes"]

    # 1. El binario reproduce sus propias predicciones de referencia en este entorno.
    own = pd.read_csv(fase2_dir / "reference_predictions.csv")
    own_pred = np.clip(model.predict(own[inputs].astype(dtypes)), 0, None)
    own_cmp = diff_stats(own_pred, own["expected_pred"].to_numpy(dtype=float))
    own_cmp["pass_tolerance_1e-9"] = own_cmp["max_abs_diff"] <= 1e-9

    # 2. Binario de la Fase 2 contra el champion reentrenado sobre el patrón oro de la fase final.
    ref = pd.read_csv(args.reference)
    actual = ref[card["target"]].to_numpy(dtype=float)
    champion_pred = ref["demand_pred"].to_numpy(dtype=float)
    fase2_pred = np.clip(model.predict(ref[inputs].astype(dtypes)), 0, None)
    comparison = diff_stats(fase2_pred, champion_pred)
    comparison["pearson_correlation"] = float(np.corrcoef(fase2_pred, champion_pred)[0, 1])
    comparison["rows_abs_diff_gt_0.1"] = int((np.abs(fase2_pred - champion_pred) > 0.1).sum())
    comparison["mean_diff_fase2_minus_retrained"] = float(np.mean(fase2_pred - champion_pred))

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8")) if Path(args.metadata).exists() else {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                        "joblib": joblib.__version__},
        "fase2_binary": {
            "folder": rel(fase2_dir),
            "source": "rama origin/entrega-final, models/champion/ (copia sin modificar)",
            "model_uri": card.get("model_uri"), "training_run_id": card.get("training_run_id"),
            "algorithm": card.get("algorithm"), "hyperparameters": json.loads(card.get("hyperparameters") or "{}"),
            "registry_metrics": card.get("registry_metrics"), "export_environment": card.get("export_environment"),
            "training_dataset_sha256": card.get("training_dataset_sha256"),
        },
        "retrained_champion": {
            "model_uri": metadata.get("model_uri"), "run_id": metadata.get("run_id"),
            "model_sha256": metadata.get("model_sha256"), "hyperparameters": metadata.get("hyperparameters"),
            "dataset_sha256": metadata.get("dataset_sha256"),
            "test_mae_registry": (metadata.get("metrics") or {}).get("test_mae"),
            "library_versions": metadata.get("library_versions"),
        },
        "hashes": hashes,
        "fase2_reproduces_own_reference": {"file": rel(fase2_dir / "reference_predictions.csv"), **own_cmp},
        "fase2_vs_retrained_on_reference": {
            "file": rel(Path(args.reference)),
            **comparison,
            "metrics_vs_actual": {"fase2_binary": error_metrics(fase2_pred, actual),
                                  "retrained_champion": error_metrics(champion_pred, actual)},
        },
        "same_hyperparameters": (json.loads(card.get("hyperparameters") or "{}")
                                 == (metadata.get("hyperparameters") or {})),
    }


def markdown(r: Dict[str, Any]) -> str:
    own, cmp_ = r["fase2_reproduces_own_reference"], r["fase2_vs_retrained_on_reference"]
    m2, mr = cmp_["metrics_vs_actual"]["fase2_binary"], cmp_["metrics_vs_actual"]["retrained_champion"]
    h, env, exp = r["hashes"], r["environment"], r["fase2_binary"]["export_environment"] or {}
    return "\n".join([
        "# Binario original de la Fase 2 contra el champion reentrenado",
        "",
        f"Generado con `python scripts/compare_fase2_binary.py` ({r['generated_at']}). Datos completos en "
        f"`{OUTPUT_NAME}.json`.",
        "",
        "`models/fase2_original/` es una copia sin modificar del paquete exportado en la rama `entrega-final` (Jesús "
        f"Rincón): `{r['fase2_binary']['model_uri']}`, corrida `{r['fase2_binary']['training_run_id']}`, exportado "
        f"con Python {exp.get('python')} en {exp.get('platform')} (scikit-learn {exp.get('scikit_learn')}).",
        "",
        "## 1. Integridad",
        "",
        f"- SHA-256 de `model.joblib`: `{h['model_sha256_file']}` (ficha: coincide = {h['model_sha256_ok']}).",
        f"- SHA-256 del CSV de referencia: coincide con la ficha = {h['reference_sha256_ok']} (Git lo guarda con "
        f"{h['reference_line_endings_in_git']}; la ficha tiene el hash de la versión CRLF escrita en Windows).",
        "",
        "## 2. El mismo binario en otro entorno",
        "",
        f"Entorno de esta verificación: Python {env['python']}, {env['platform']}, "
        f"scikit-learn {env['scikit-learn']}, "
        f"numpy {env['numpy']}, pandas {env['pandas']}.",
        "",
        "| Filas | Diferencia máxima | Diferencia media | Idénticas bit a bit | Tolerancia 1e-9 |",
        "|---|---|---|---|---|",
        f"| {own['rows']} | {own['max_abs_diff']!r} | {own['mean_abs_diff']!r} | {own['rows_bitwise_identical']}/"
        f"{own['rows']} | {'PASS' if own['pass_tolerance_1e-9'] else 'FAIL'} |",
        "",
        "## 3. Binario de la Fase 2 contra el champion reentrenado (500 filas de `models/reference_predictions.csv`)",
        "",
        f"Mismos hiperparámetros: {r['same_hyperparameters']}. Los dos modelos se entrenaron con el mismo código y "
        "semilla, pero con datos del portal descargados en fechas distintas (ver "
        "`reports/fase_final/comparacion_fase2.md`), así que no son el mismo binario.",
        "",
        "| Medida | Valor |",
        "|---|---|",
        f"| Diferencia máxima entre predicciones | {cmp_['max_abs_diff']!r} |",
        f"| Diferencia media absoluta | {cmp_['mean_abs_diff']!r} |",
        f"| Diferencia media (Fase 2 - reentrenado) | {cmp_['mean_diff_fase2_minus_retrained']!r} |",
        f"| Filas con diferencia > 0.1 viajes | {cmp_['rows_abs_diff_gt_0.1']} de {cmp_['rows']} |",
        f"| Correlación de Pearson | {cmp_['pearson_correlation']!r} |",
        f"| MAE contra la demanda real: binario Fase 2 | {m2['mae']!r} |",
        f"| MAE contra la demanda real: champion reentrenado | {mr['mae']!r} |",
        f"| RMSE: binario Fase 2 / reentrenado | {m2['rmse']!r} / {mr['rmse']!r} |",
        "",
        "El MAE de esta tabla es sobre la muestra de 500 filas, no sobre el conjunto de prueba completo (MAE de prueba "
        f"registrado: Fase 2 {(r['fase2_binary']['registry_metrics'] or {}).get('test_mae')!r}, reentrenado "
        f"{r['retrained_champion']['test_mae_registry']!r}).",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara el binario de la Fase 2 con el champion reentrenado")
    parser.add_argument("--fase2-dir", default=str(DEFAULT_FASE2_DIR))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = run(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{OUTPUT_NAME}.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    text = markdown(result)
    with open(out / f"{OUTPUT_NAME}.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    print(f"Resultados: {rel(out / (OUTPUT_NAME + '.json'))} y .md")
    ok = (result["hashes"]["model_sha256_ok"] and result["hashes"]["reference_sha256_ok"]
          and result["fase2_reproduces_own_reference"]["pass_tolerance_1e-9"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
