"""Exporta el esquema OpenAPI del servicio de predicción a docs/openapi.json.

No necesita el modelo: el esquema se genera a partir de la aplicación (rutas y modelos de Pydantic).

Uso:
    python scripts/export_openapi.py
    python scripts/export_openapi.py --output docs/openapi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.config import load_config  # noqa: E402
from src.serving.app import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta el esquema OpenAPI de la API de MiBici")
    parser.add_argument("--config", default=None, help="Ruta al config.yaml (por defecto, el del proyecto)")
    parser.add_argument("--output", default=str(PROJ_ROOT / "docs" / "openapi.json"), help="Archivo de salida")
    args = parser.parse_args()
    spec = create_app(load_config(args.config)).openapi()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OpenAPI {spec['info']['version']} ({len(spec['paths'])} rutas) -> {output}")


if __name__ == "__main__":
    main()
