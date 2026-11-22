"""Prueba de humo de la API en ejecución: llama a todos los endpoints (peticiones válidas e inválidas) y
guarda cada petición y respuesta en un log. Solo usa la biblioteca estándar.

Además envía las filas de models/reference_predictions.csv por /predict y /predict/raw y reporta la
diferencia máxima absoluta contra `demand_pred`.

Uso (con el servicio levantado):
    python scripts/api_smoke.py --url http://127.0.0.1:8000 --output reports/fase_final/logs/20_api_smoke.log
    python scripts/api_smoke.py --url http://127.0.0.1:8001 --only-unavailable   # servicio sin modelo
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJ_ROOT = Path(__file__).resolve().parents[1]
LAGS = ("demand_lag_24h", "demand_lag_168h", "demand_same_hour_mean_7d")
INT_COLUMNS = ("capacity", "Origen_Id", "is_weekend", "is_holiday")
MAX_BODY_CHARS = 4000


def call(base: str, method: str, path: str, body: Any = None, raw: Optional[str] = None) -> Tuple[int, Any, float]:
    data = raw.encode("utf-8") if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, text = resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        status, text = err.code, err.read().decode("utf-8")
    elapsed = (time.perf_counter() - start) * 1000
    try:
        return status, json.loads(text), elapsed
    except json.JSONDecodeError:
        return status, text, elapsed


def _short(obj: Any) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2)
    return text if len(text) <= MAX_BODY_CHARS else text[:MAX_BODY_CHARS] + f"\n... ({len(text)} caracteres en total)"


def load_reference() -> Tuple[List[Dict[str, str]], List[str]]:
    with open(PROJ_ROOT / "models" / "reference_predictions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    metadata = json.loads((PROJ_ROOT / "models" / "champion_metadata.json").read_text(encoding="utf-8"))
    return rows, metadata["input_columns"]


def instance(row: Dict[str, str]) -> Dict[str, Any]:
    return {"station_id": int(row["Origen_Id"]), "datetime": row["fecha_hora"].replace(" ", "T"),
            **{lag: float(row[lag]) for lag in LAGS}}


def raw_row(row: Dict[str, str], cols: List[str]) -> Dict[str, Any]:
    return {c: (int(row[c]) if c in INT_COLUMNS else float(row[c])) for c in cols}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de humo de la API de MiBici")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default=None, help="Log de salida (se agrega al final si ya existe)")
    parser.add_argument("--only-unavailable", action="store_true", help="Casos para un servicio sin modelo (503)")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    lines: List[str] = [f"===== {datetime.now().isoformat(timespec='seconds')} | {base} ====="]

    def log_case(name: str, method: str, path: str, body: Any = None, raw: Optional[str] = None,
                 show_request: bool = True) -> Tuple[int, Any]:
        status, resp, ms = call(base, method, path, body, raw)
        lines.append(f"\n--- {name}\n>>> {method} {path}")
        if show_request and (body is not None or raw is not None):
            lines.append(_short(raw if raw is not None else body))
        lines.append(f"<<< HTTP {status} ({ms:.1f} ms)\n{_short(resp)}")
        print(f"{status} {method} {path} - {name}")
        return status, resp

    rows, cols = load_reference()
    first = instance(rows[0])
    if args.only_unavailable:
        log_case("health sin modelo (503)", "GET", "/health")
        log_case("model-info sin modelo (503)", "GET", "/model-info")
        log_case("predict sin modelo (503)", "POST", "/predict", {"instances": [first]})
    else:
        log_case("raíz", "GET", "/")
        log_case("health", "GET", "/health")
        log_case("model-info", "GET", "/model-info")
        log_case("predict: una instancia (fila 1 de reference_predictions.csv)", "POST", "/predict",
                 {"instances": [first]})
        log_case("predict: lote de 3 instancias (filas 1-3 de la referencia)", "POST", "/predict",
                 {"instances": [instance(r) for r in rows[:3]]})
        log_case("predict: misma hora con zona horaria UTC (16:30Z = 10:30 local, se trunca a 10:00)", "POST",
                 "/predict", {"instances": [{**first, "datetime": "2025-11-08T16:30:00Z"}]})
        log_case("predict/raw: fila 1 de la referencia con las columnas ya construidas", "POST", "/predict/raw",
                 {"rows": [raw_row(rows[0], cols)]})
        log_case("422: rezago negativo", "POST", "/predict", {"instances": [{**first, "demand_lag_24h": -1}]})
        log_case("422: NaN e Infinity", "POST", "/predict",
                 raw=json.dumps({"instances": [{**first, "demand_lag_24h": float("nan"),
                                                "demand_lag_168h": float("inf")}]}))
        log_case("422: fecha inválida", "POST", "/predict", {"instances": [{**first, "datetime": "2025-13-45T99:00"}]})
        log_case("422: tipo incorrecto en station_id y campo extra", "POST", "/predict",
                 {"instances": [{**first, "station_id": "24", "otro": 1}]})
        log_case("422: campo faltante", "POST", "/predict",
                 {"instances": [{k: v for k, v in first.items() if k != "demand_lag_168h"}]})
        log_case("422: lote vacío", "POST", "/predict", {"instances": []})
        log_case("422: lote mayor que serving.max_batch_size (1001 instancias)", "POST", "/predict",
                 {"instances": [first] * 1001}, show_request=False)
        log_case("422: JSON inválido", "POST", "/predict", raw="{no es json")
        log_case("422: predict/raw sin una columna", "POST", "/predict/raw",
                 {"rows": [{k: v for k, v in raw_row(rows[0], cols).items() if k != "capacity"}]})
        log_case("404: estaciones desconocidas", "POST", "/predict",
                 {"instances": [first, {**first, "station_id": 99999}, {**first, "station_id": 5000}]})
        log_case("404: ruta inexistente", "GET", "/no-existe")
        log_case("405: método no permitido", "GET", "/predict")

        expected = [float(r["demand_pred"]) for r in rows]
        for path, body, getter in (
            ("/predict", {"instances": [instance(r) for r in rows]},
             lambda b: [p["demand_pred"] for p in b["predictions"]]),
            ("/predict/raw", {"rows": [raw_row(r, cols) for r in rows]}, lambda b: b["predictions"]),
        ):
            status, resp = log_case(f"reproducibilidad: {len(rows)} filas de la referencia por {path}", "POST", path,
                                    body, show_request=False)
            if status == 200:
                preds = getter(resp)
                diffs = [abs(a - b) for a, b in zip(preds, expected)]
                exact = sum(a == b for a, b in zip(preds, expected))
                lines.append(f"*** {path}: {len(preds)} predicciones; diferencia máxima absoluta = {max(diffs)!r}; "
                             f"iguales bit a bit = {exact}/{len(expected)}")
                print(lines[-1])

    text = "\n".join(lines) + "\n"
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Log: {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
