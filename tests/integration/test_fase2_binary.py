"""Binario original de la Fase 2 (models/fase2_original/, copiado sin cambios de la rama entrega-final).

Evidencia de que el mismo binario produce resultados consistentes en otro entorno: se entrenó en otra máquina con
Python 3.12 y aquí (Windows o el runner Linux de CI, Python 3.13) debe reproducir sus propias predicciones de
referencia. Se omite si falta la carpeta.
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import pytest

from src.config import PROJ_ROOT

pytestmark = pytest.mark.integration

FASE2_DIR = PROJ_ROOT / "models" / "fase2_original"
SCRIPT = PROJ_ROOT / "scripts" / "compare_fase2_binary.py"


@pytest.fixture(scope="module")
def compare_module():
    if not (FASE2_DIR / "model.joblib").exists():
        pytest.skip("No existe models/fase2_original/ (binario de la Fase 2)")
    spec = importlib.util.spec_from_file_location("compare_fase2_binary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result(compare_module):
    args = argparse.Namespace(fase2_dir=str(FASE2_DIR), reference=str(compare_module.DEFAULT_REFERENCE),
                              metadata=str(compare_module.DEFAULT_METADATA))
    return compare_module.run(args)


def test_hashes_coinciden_con_la_ficha(result):
    assert result["hashes"]["model_sha256_ok"] is True
    assert result["hashes"]["reference_sha256_ok"] is True


def test_binario_reproduce_su_referencia(result):
    own = result["fase2_reproduces_own_reference"]
    assert own["rows"] == 1000
    # En Windows la diferencia medida es 0; en Linux el exp() final puede variar 1 ulp (ver docs/docker.md).
    assert own["max_abs_diff"] <= 1e-9


def test_comparacion_con_el_reentrenado(result):
    cmp_ = result["fase2_vs_retrained_on_reference"]
    assert cmp_["rows"] == 500
    assert cmp_["pearson_correlation"] > 0.99
    metrics = cmp_["metrics_vs_actual"]
    assert set(metrics) == {"fase2_binary", "retrained_champion"}
    assert result["same_hyperparameters"] is True


def test_script_escribe_json_y_md(compare_module, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["compare", "--output-dir", str(tmp_path)])
    assert compare_module.main() == 0
    data = json.loads((tmp_path / "fase2_binario_vs_reentrenado.json").read_text(encoding="utf-8"))
    assert data["hashes"]["model_sha256_ok"] is True
    assert "Binario original de la Fase 2" in (tmp_path / "fase2_binario_vs_reentrenado.md").read_text(
        encoding="utf-8")
