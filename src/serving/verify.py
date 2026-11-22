"""Verificación de reproducibilidad del modelo desplegable en cualquier entorno.

Compara, en la máquina o contenedor donde se ejecute, las predicciones del paquete de
despliegue contra las predicciones de referencia guardadas al exportarlo, y opcionalmente
recalcula las métricas del periodo de prueba contra las registradas en el Model Registry.

Uso:
    python -m src.serving.verify                 # solo predicciones de referencia (no requiere datos)
    python -m src.serving.verify --full          # además, métricas sobre todo el periodo de prueba
    python -m src.serving.verify --tag contenedor --output reports/reproducibility
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import sklearn

from src.config import PROJ_ROOT, load_config
from src.serving.bundle import MODEL_FILE, REFERENCE_FILE, sha256_file
from src.serving.service import ModelService

# Tolerancia en bicicletas por hora. Entre Windows/Python 3.12 y Linux/Python 3.13 la diferencia
# medida fue de ~1e-15 (redondeo de punto flotante), así que 1e-6 deja margen sin ocultar cambios reales.
TOLERANCE = 1e-6


def read_reference(path: Path) -> pd.DataFrame:
    """Lee las predicciones de referencia con precisión completa de punto flotante.

    pandas 3.0 necesita float_precision="round_trip"; pandas 3.1 ya lo hace por defecto y marca el
    argumento como obsoleto, así que se usa sin mostrar el aviso y se omite si llega a eliminarse.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.read_csv(path, parse_dates=["fecha_hora"], float_precision="round_trip")
    except TypeError:
        return pd.read_csv(path, parse_dates=["fecha_hora"])


def verify(bundle_dir: str | Path, full: bool = False, config_path: Optional[str] = None) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    service = ModelService(bundle_dir)
    card = service.card

    reference = read_reference(bundle_dir / REFERENCE_FILE)
    preds = service.predict_frame(reference)
    diff = np.abs(preds - reference["expected_pred"].to_numpy())
    result: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "export_environment": card.get("export_environment"),
        "model_uri": card.get("model_uri"),
        "model_sha256_expected": card.get("model_sha256"),
        "model_sha256_actual": sha256_file(bundle_dir / MODEL_FILE),
        "reference_rows": int(len(reference)),
        "reference_max_abs_diff": float(diff.max()),
        "reference_mean_abs_diff": float(diff.mean()),
    }
    result["model_hash_ok"] = result["model_sha256_expected"] == result["model_sha256_actual"]
    result["tolerance"] = TOLERANCE
    result["predictions_identical"] = result["reference_max_abs_diff"] <= 1e-12
    result["predictions_ok"] = result["reference_max_abs_diff"] <= TOLERANCE

    if full:
        from src.features import load_features, temporal_split
        from src.modeling.pipeline import evaluate

        config = load_config(config_path)
        df = load_features(config)
        test = df[temporal_split(df, config) == "test"]
        metrics = evaluate(test[config["features"]["target"]], service.predict_frame(test), "test")
        expected = card.get("registry_metrics", {})
        result["test_metrics"] = metrics
        result["test_metrics_registry"] = expected
        result["test_mae_abs_diff"] = abs(metrics["test_mae"] - expected["test_mae"]) if "test_mae" in expected \
            else None
        # Las métricas del registro se guardan con 4 decimales.
        result["metrics_ok"] = result["test_mae_abs_diff"] is not None and result["test_mae_abs_diff"] < 1e-4

    checks = [result["model_hash_ok"], result["predictions_ok"]] + ([result["metrics_ok"]] if full else [])
    result["reproducible"] = bool(all(checks))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica la reproducibilidad del modelo desplegable")
    parser.add_argument("--bundle", default=str(PROJ_ROOT / "models" / "champion"))
    parser.add_argument("--full", action="store_true", help="Recalcular métricas del periodo de prueba")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tag", default=None, help="Nombre del entorno (p. ej. windows-jesus, contenedor)")
    parser.add_argument("--output", default=None, help="Carpeta donde guardar el resultado en JSON")
    args = parser.parse_args()

    result = verify(args.bundle, args.full, args.config)
    result["tag"] = args.tag
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nREPRODUCIBLE" if result["reproducible"] else "\nNO REPRODUCIBLE: revisar diferencias")
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        name = f"verify_{args.tag or result['host']}.json"
        (out / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resultado guardado en {out / name}")
    raise SystemExit(0 if result["reproducible"] else 1)


if __name__ == "__main__":
    main()
