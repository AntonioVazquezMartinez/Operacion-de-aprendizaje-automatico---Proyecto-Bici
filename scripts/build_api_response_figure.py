"""Figura 14: petición y respuesta 200 reales de POST /predict, tomadas del log de la prueba de humo.

No es una captura del navegador: la imagen se dibuja a partir del texto registrado por scripts/api_smoke.py en
reports/fase_final/logs/20_api_smoke.log (sección B: contenedor ml-service:latest, servicio 1.1.0).

Uso:
    python scripts/build_api_response_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJ_ROOT / "reports" / "fase_final" / "logs" / "20_api_smoke.log"
DEFAULT_OUTPUT = PROJ_ROOT / "reports" / "figures" / "14_api_respuesta_200.png"
SECTION = "##### B)"
CASE = "--- predict: lote mixto (con rezagos y sin rezagos)"


def extract_case(log: Path) -> tuple[str, str, str]:
    """Regresa (petición, línea de estado, respuesta) del caso CASE dentro de la sección SECTION."""
    text = log.read_text(encoding="utf-8")
    section = text[text.index(SECTION):]
    block = section[section.index(CASE):].split("\n--- ", 1)[0]
    request = block.split(">>> POST /predict\n", 1)[1].split("\n<<< ", 1)[0]
    status, response = block.split("\n<<< ", 1)[1].split("\n", 1)
    return request.strip(), status.strip(), response.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Figura de la respuesta 200 real de POST /predict")
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    request, status, response = extract_case(Path(args.log))

    fig = plt.figure(figsize=(14, 7.4), dpi=110)
    fig.patch.set_facecolor("white")
    fig.text(0.03, 0.97, "POST /predict - respuesta real del contenedor ml-service:latest (servicio 1.1.0)",
             fontsize=15, weight="bold", va="top")
    fig.text(0.03, 0.92, "Instancia 1 con sus tres rezagos (lags_source = request); instancia 2 sin rezagos: el "
             "servicio los calcula con el feature store (lags_source = feature_store).", fontsize=10.5, va="top")
    style = dict(family="monospace", fontsize=9.6, va="top")
    box = dict(boxstyle="round,pad=0.6", facecolor="#f4f6f8", edgecolor="#c9d1d9")
    fig.text(0.03, 0.855, "Petición (JSON)", fontsize=11.5, weight="bold", va="top")
    fig.text(0.03, 0.81, request, bbox=box, **style)
    fig.text(0.47, 0.855, f"Respuesta: {status}", fontsize=11.5, weight="bold", va="top", color="#1a7f37")
    fig.text(0.47, 0.81, response, bbox=box, **style)
    fig.text(0.03, 0.025, "Imagen generada con scripts/build_api_response_figure.py a partir de la petición y la "
             "respuesta registradas en reports/fase_final/logs/20_api_smoke.log (sección B); no es una captura del "
             "navegador.", fontsize=8.5, color="#57606a")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Figura: {out}")


if __name__ == "__main__":
    main()
