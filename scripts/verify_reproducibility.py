"""Verificación de reproducibilidad del servicio de predicción contra el patrón oro del entrenamiento.

Envía las filas de `models/reference_predictions.csv` (500 filas del conjunto de prueba con la predicción
`demand_pred` calculada en el entorno de entrenamiento) a un servicio en ejecución por dos caminos:

- `POST /predict/raw`: las 13 columnas de entrada del modelo tal como están en el CSV.
- `POST /predict`: solo estación, hora y rezagos; el servidor deriva calendario y atributos de estación.

y, desde la versión 1.1.0 del servicio, las filas de `models/reference_feature_store_predictions.csv` (horas de
la ventana del feature store, con rezagos y predicción calculados fuera de línea en el entorno de entrenamiento)
por un tercer camino:

- `POST /predict` sin rezagos: solo estación y hora; el servidor toma los rezagos de su feature store. Se comparan
  la predicción y también los tres rezagos que reporta la respuesta (deben ser idénticos).

Compara cada predicción con la de referencia (diferencia máxima y media absoluta, filas idénticas bit a bit),
recalcula MAE/RMSE contra el valor real y comprueba que el SHA-256 del modelo cargado sea el de
`models/champion_metadata.json`. Escribe un JSON con el entorno del cliente, el del servidor (reportado por
`/model-info`) y el veredicto PASS/FAIL; el código de salida es 0 solo con PASS.

Solo usa la biblioteca estándar (Python 3.10+): corre en cualquier máquina o runner sin instalar nada.

Uso:
    python scripts/verify_reproducibility.py --url http://localhost:8000 --label contenedor-local \
        --output reports/fase_final/reproducibilidad/contenedor_local.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import struct
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PROJ_ROOT / "models" / "reference_predictions.csv"
DEFAULT_METADATA = PROJ_ROOT / "models" / "champion_metadata.json"
DEFAULT_FS_REFERENCE = PROJ_ROOT / "models" / "reference_feature_store_predictions.csv"
# Columnas del CSV que no son entradas de /predict/raw ni del cálculo de /predict.
STATION_COLUMN, TIME_COLUMN, PRED_COLUMN = "Origen_Id", "fecha_hora", "demand_pred"
LAG_FIELDS = ("demand_lag_24h", "demand_lag_168h", "demand_same_hour_mean_7d")


# --------------------------------------------------------------------------- HTTP
def http_json(url: str, body: Any = None, timeout: float = 120.0) -> Tuple[int, Any]:
    """GET (sin cuerpo) o POST JSON; regresa (código HTTP, cuerpo decodificado)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, text = resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        status, text = err.code, err.read().decode("utf-8", errors="replace")
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


def wait_for_health(base: str, attempts: int, delay: float) -> Tuple[bool, Any, int]:
    """Reintenta GET /health hasta obtener 200; regresa (listo, último cuerpo, intentos usados)."""
    last: Any = None
    for attempt in range(1, attempts + 1):
        try:
            status, body = http_json(base + "/health", timeout=10)
            last = {"status_code": status, "body": body}
            if status == 200:
                return True, last, attempt
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last = {"error": f"{type(exc).__name__}: {exc}"}
        time.sleep(delay)
    return False, last, attempts


# --------------------------------------------------------------------------- datos de referencia
def load_reference(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_number(text: str) -> float | int:
    """Entero si el texto es un entero literal; si no, float (float() es exacto para el repr de float64)."""
    stripped = text.strip()
    if stripped.lstrip("+-").isdigit():
        return int(stripped)
    return float(stripped)


def raw_rows(rows: Sequence[Dict[str, str]], input_columns: Sequence[str]) -> List[Dict[str, float | int]]:
    return [{c: parse_number(r[c]) for c in input_columns} for r in rows]


def predict_instances(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Entradas de /predict derivadas del CSV: estación, hora local (ISO-8601) y los tres rezagos."""
    return [{"station_id": int(r[STATION_COLUMN]), "datetime": r[TIME_COLUMN].strip().replace(" ", "T"),
             **{lag: float(r[lag]) for lag in LAG_FIELDS}} for r in rows]


def feature_store_instances(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Entradas de /predict SIN rezagos: el servidor los calcula con su feature store."""
    return [{"station_id": int(r[STATION_COLUMN]), "datetime": r[TIME_COLUMN].strip().replace(" ", "T")}
            for r in rows]


def chunks(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def call_predictions(base: str, path: str, key: str, payload: Sequence[Any], batch: int,
                     records: Optional[List[Any]] = None) -> Tuple[List[float], Any]:
    """Envía el lote en trozos de `batch` y regresa las predicciones en orden (o el error del servidor).

    Si se pasa `records`, ahí se agregan los objetos completos de la respuesta (rezagos usados, lags_source).
    """
    preds: List[float] = []
    for part in chunks(payload, batch):
        status, body = http_json(base + path, {key: list(part)})
        if status != 200:
            return preds, {"status_code": status, "body": body}
        values = body["predictions"]
        preds.extend(float(v["demand_pred"]) if isinstance(v, dict) else float(v) for v in values)
        if records is not None:
            records.extend(values)
    return preds, None


# --------------------------------------------------------------------------- comparación
def bits(x: float) -> bytes:
    return struct.pack("<d", x)


def compare(pred: Sequence[float], ref: Sequence[float], tolerance: float) -> Dict[str, Any]:
    diffs = [abs(p - r) for p, r in zip(pred, ref)]
    worst = max(range(len(diffs)), key=diffs.__getitem__) if diffs else None
    return {
        "rows": len(pred),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": math.fsum(diffs) / len(diffs) if diffs else None,
        "rows_bitwise_identical": sum(bits(p) == bits(r) for p, r in zip(pred, ref)),
        "rows_within_tolerance": sum(d <= tolerance for d in diffs),
        "worst_row_index": worst,
        "worst_row": {"pred": pred[worst], "reference": ref[worst]} if worst is not None else None,
        "pass": len(pred) == len(ref) and bool(diffs) and max(diffs) <= tolerance,
    }


def error_metrics(pred: Sequence[float], actual: Sequence[float]) -> Dict[str, float]:
    errors = [p - a for p, a in zip(pred, actual)]
    return {"mae": math.fsum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(math.fsum(e * e for e in errors) / len(errors))}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Ruta relativa al repositorio cuando es posible (el JSON no expone rutas personales del equipo)."""
    try:
        return path.resolve().relative_to(PROJ_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def client_environment() -> Dict[str, Any]:
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "release": platform.release(), "platform": platform.platform(),
            "machine": platform.machine()}


# --------------------------------------------------------------------------- principal
def run(args: argparse.Namespace) -> Dict[str, Any]:
    base = args.url.rstrip("/")
    reference_path = Path(args.reference)
    report: Dict[str, Any] = {
        "label": args.label,
        "url": base,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tolerance": args.tolerance,
        "client_environment": client_environment(),
        "reference": {"file": display_path(reference_path), "sha256": sha256_file(reference_path)},
        "checks": {},
    }
    checks = report["checks"]

    ready, health, attempts = wait_for_health(base, args.retries, args.retry_delay)
    report["health"] = {"ready": ready, "attempts": attempts, "last": health}
    checks["health"] = ready
    if not ready:
        return report

    status, info = http_json(base + "/model-info")
    if status != 200:
        report["model_info_error"] = {"status_code": status, "body": info}
        checks["model_info"] = False
        return report
    report["server_environment"] = {
        "service_version": info.get("service_version"),
        "model": info.get("model"),
        "run_id": info.get("run_id"),
        "model_sha256_computed": info.get("model_sha256_computed"),
        "model_sha256_verified": info.get("model_sha256_verified"),
        "runtime_library_versions": info.get("runtime_library_versions"),
        "training_library_versions": info.get("training_library_versions"),
        "max_batch_size": info.get("max_batch_size"),
        "feature_store": info.get("feature_store"),
    }
    checks["model_sha256_verified_by_server"] = bool(info.get("model_sha256_verified"))

    # SHA-256 del modelo cargado por el servidor contra los metadatos locales (si están disponibles).
    metadata_path = Path(args.metadata)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("model_sha256")
        report["model_sha256"] = {"expected_from_metadata": expected,
                                  "served": info.get("model_sha256_computed"),
                                  "metadata_file": display_path(metadata_path)}
        checks["model_sha256_matches_metadata"] = expected == info.get("model_sha256_computed")
        report["reference_sha256_matches_metadata"] = (
            metadata.get("reference_predictions", {}).get("sha256") == report["reference"]["sha256"])
    else:
        report["model_sha256"] = {"expected_from_metadata": None, "served": info.get("model_sha256_computed"),
                                  "note": f"No existe {display_path(metadata_path)}; no se comparó."}

    rows = load_reference(reference_path)
    input_columns = list(info["input_columns"])
    target = info.get("target") or "demand_count"
    reference = [float(r[PRED_COLUMN]) for r in rows]
    actual = [float(r[target]) for r in rows]
    batch = int(info.get("max_batch_size") or 1000)
    ref_metrics = error_metrics(reference, actual)
    report["reference_metrics"] = ref_metrics

    endpoints = {
        "predict_raw": ("/predict/raw", "rows", raw_rows(rows, input_columns)),
        "predict": ("/predict", "instances", predict_instances(rows)),
    }
    report["endpoints"] = {}
    records: Dict[str, List[Any]] = {name: [] for name in endpoints}
    for name, (path, key, payload) in endpoints.items():
        start = time.perf_counter()
        preds, error = call_predictions(base, path, key, payload, batch, records[name])
        elapsed_ms = (time.perf_counter() - start) * 1000
        if error is not None or len(preds) != len(reference):
            report["endpoints"][name] = {"path": path, "error": error, "rows_received": len(preds)}
            checks[f"{name}_predictions"] = False
            continue
        result = compare(preds, reference, args.tolerance)
        metrics = error_metrics(preds, actual)
        result.update({
            "path": path,
            "elapsed_ms": round(elapsed_ms, 2),
            "metrics": metrics,
            "metrics_abs_diff_vs_reference": {k: abs(metrics[k] - ref_metrics[k]) for k in metrics},
        })
        report["endpoints"][name] = result
        checks[f"{name}_predictions"] = result["pass"]
        checks[f"{name}_metrics"] = all(v <= args.tolerance for v in result["metrics_abs_diff_vs_reference"].values())
    sources = {r.get("lags_source") for r in records["predict"] if isinstance(r, dict)}
    if sources and sources != {None}:  # servicio >= 1.1.0: con rezagos enviados el origen debe ser "request"
        checks["predict_lags_source_request"] = sources == {"request"}

    verify_feature_store(args, base, info, batch, report, checks)
    return report


def verify_feature_store(args: argparse.Namespace, base: str, info: Dict[str, Any], batch: int,
                         report: Dict[str, Any], checks: Dict[str, bool]) -> None:
    """Camino del feature store: /predict sin rezagos contra la referencia calculada fuera de línea."""
    fs_path = Path(args.fs_reference)
    server_fs = info.get("feature_store") or {}
    if not server_fs.get("enabled"):
        report["feature_store"] = {"skipped": "El servicio no tiene feature store (versión anterior a 1.1.0)."}
        return
    if not fs_path.exists():
        report["feature_store"] = {"skipped": f"No existe {display_path(fs_path)}."}
        return
    rows = load_reference(fs_path)
    expected = [float(r[PRED_COLUMN]) for r in rows]
    section: Dict[str, Any] = {"reference": {"file": display_path(fs_path), "sha256": sha256_file(fs_path),
                                             "rows": len(rows)},
                               "server_feature_store_sha256_verified": server_fs.get("sha256_verified"),
                               "predictable_window": server_fs.get("predictable_window")}
    metadata_path = Path(args.metadata)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        section["reference_sha256_matches_metadata"] = (
            metadata.get("feature_store_reference_predictions", {}).get("sha256") == section["reference"]["sha256"])
        expected_fs = (metadata.get("feature_store") or {}).get("sha256")
        section["feature_store_sha256"] = {"expected_from_metadata": expected_fs, "served": server_fs.get("sha256")}
        checks["feature_store_sha256_matches_metadata"] = expected_fs == server_fs.get("sha256")
    checks["feature_store_sha256_verified_by_server"] = bool(server_fs.get("sha256_verified"))

    returned: List[Any] = []
    start = time.perf_counter()
    preds, error = call_predictions(base, "/predict", "instances", feature_store_instances(rows), batch, returned)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if error is not None or len(preds) != len(expected):
        section["predict_feature_store"] = {"path": "/predict (sin rezagos)", "error": error,
                                            "rows_received": len(preds)}
        checks["predict_feature_store_predictions"] = False
        report["feature_store"] = section
        return
    result = compare(preds, expected, args.tolerance)
    lag_diffs = {lag: max(abs(float(r[lag]) - float(row[lag])) for r, row in zip(returned, rows)) for lag in LAG_FIELDS}
    lag_identical = sum(all(bits(float(r[lag])) == bits(float(row[lag])) for lag in LAG_FIELDS)
                        for r, row in zip(returned, rows))
    in_data = [i for i, row in enumerate(rows) if row.get("in_data_period", "1").strip() == "1"]
    result.update({
        "path": "/predict (sin rezagos)",
        "elapsed_ms": round(elapsed_ms, 2),
        "rows_in_data_period": len(in_data),
        "rows_after_data_period": len(rows) - len(in_data),
        "lags_max_abs_diff": lag_diffs,
        "rows_lags_bitwise_identical": lag_identical,
        "lags_source": sorted({str(r.get("lags_source")) for r in returned}),
    })
    section["predict_feature_store"] = result
    checks["predict_feature_store_predictions"] = result["pass"]
    checks["predict_feature_store_lags"] = lag_identical == len(rows)
    checks["predict_feature_store_lags_source"] = result["lags_source"] == ["feature_store"]
    report["feature_store"] = section


def summary(report: Dict[str, Any]) -> str:
    lines = [f"Verificación de reproducibilidad [{report['label']}] contra {report['url']}",
             f"  Cliente: {report['client_environment']['platform']} (Python {report['client_environment']['python']})"]
    server = report.get("server_environment")
    if server:
        versions = ", ".join(f"{k} {v}" for k, v in (server.get("runtime_library_versions") or {}).items())
        lines.append(f"  Servidor: servicio {server.get('service_version')}, modelo {server['model'].get('uri')}; "
                     f"{versions}")
        sha = report.get("model_sha256", {})
        lines.append(f"  SHA-256 servido {sha.get('served')} | esperado {sha.get('expected_from_metadata')}")
    for name, res in report.get("endpoints", {}).items():
        if "error" in res:
            lines.append(f"  {res['path']}: ERROR {res['error']}")
            continue
        lines.append(f"  {res['path']:<13} filas {res['rows']} | dif. máx. {res['max_abs_diff']!r} | dif. media "
                     f"{res['mean_abs_diff']!r} | idénticas bit a bit {res['rows_bitwise_identical']}/{res['rows']}")
        lines.append(f"  {'':<13} MAE {res['metrics']['mae']!r} RMSE {res['metrics']['rmse']!r} "
                     f"(referencia MAE {report['reference_metrics']['mae']!r} "
                     f"RMSE {report['reference_metrics']['rmse']!r})")
    fs = report.get("feature_store") or {}
    if "skipped" in fs:
        lines.append(f"  Feature store: omitido ({fs['skipped']})")
    elif fs.get("predict_feature_store"):
        res = fs["predict_feature_store"]
        if "error" in res:
            lines.append(f"  {res['path']}: ERROR {res['error']}")
        else:
            window = fs.get("predictable_window") or {}
            lines.append(f"  /predict sin rezagos (feature store, ventana {window.get('start')} a "
                         f"{window.get('end')}):")
            lines.append(f"  {'':<13} filas {res['rows']} | dif. máx. {res['max_abs_diff']!r} | idénticas bit a bit "
                         f"{res['rows_bitwise_identical']}/{res['rows']} | rezagos idénticos "
                         f"{res['rows_lags_bitwise_identical']}/{res['rows']}")
    failed = [k for k, ok in report["checks"].items() if not ok]
    lines.append(f"  Tolerancia {report['tolerance']!r} | veredicto {report['verdict']}"
                 + (f" (fallan: {', '.join(failed)})" if failed else ""))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica que el servicio reproduce las predicciones de referencia")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base del servicio")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="CSV de predicciones de referencia")
    parser.add_argument("--fs-reference", default=str(DEFAULT_FS_REFERENCE),
                        help="CSV de referencia del feature store (camino /predict sin rezagos)")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA),
                        help="champion_metadata.json para comparar el SHA-256 del modelo (se omite si no existe)")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="Diferencia absoluta máxima aceptada por fila (por omisión 1e-9)")
    parser.add_argument("--output", help="Ruta del JSON de resultados")
    parser.add_argument("--label", default=platform.system().lower(), help="Nombre del entorno verificado")
    parser.add_argument("--retries", type=int, default=40, help="Intentos de GET /health antes de fallar")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="Segundos entre intentos de /health")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # acentos legibles también al redirigir la salida en Windows
        sys.stdout.reconfigure(encoding="utf-8")

    report = run(args)
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["verdict"] = "PASS" if report["checks"] and all(report["checks"].values()) else "FAIL"
    print(summary(report))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        # json.dumps escribe cada float con repr (precisión completa de float64).
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  Resultado: {out.as_posix()}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
