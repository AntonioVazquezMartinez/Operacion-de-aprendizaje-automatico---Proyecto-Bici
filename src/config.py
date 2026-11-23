"""Carga de la configuración central (config.yaml) y resolución de rutas del proyecto.

Las rutas se resuelven contra la raíz del repositorio, por lo que los módulos funcionan
igual desde la terminal, desde notebooks/ o desde las pruebas.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJ_ROOT / "config.yaml"


class Config(dict):
    """Diccionario de configuración con utilidades para rutas y MLflow."""

    def path(self, key: str) -> Path:
        """Ruta absoluta de una carpeta definida en `paths` (la crea si no existe)."""
        folder = Path(self["paths"][key])
        if not folder.is_absolute():
            folder = PROJ_ROOT / folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def file(self, dir_key: str, file_key: str) -> Path:
        """Ruta absoluta de un archivo de datos: carpeta de `paths` + nombre de `data`."""
        return self.path(dir_key) / self["data"][file_key]

    @property
    def tracking_uri(self) -> str:
        """URI de MLflow; una base SQLite relativa se ancla a la raíz del proyecto."""
        uri = self["mlflow"]["tracking_uri"]
        prefix = "sqlite:///"
        if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
            return prefix + (PROJ_ROOT / uri[len(prefix):]).as_posix()
        return uri

    @property
    def artifact_location(self) -> str:
        folder = Path(self["mlflow"]["artifact_dir"])
        if not folder.is_absolute():
            folder = PROJ_ROOT / folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder.as_uri()


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Lee config.yaml (o el archivo indicado) y aplica sobrescrituras opcionales por sección."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = copy.deepcopy(data)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            data.setdefault(section, {}).update(values)
        else:
            data[section] = values
    return Config(data)


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """Parser común de línea de comandos: todos los módulos aceptan --config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Ruta al archivo de configuración YAML")
    return parser
