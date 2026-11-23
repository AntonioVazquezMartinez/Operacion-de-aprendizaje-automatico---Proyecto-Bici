"""Gráficas del EDA y de evaluación: se generan archivos PNG válidos en un directorio temporal."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src import plots

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def assert_png(path) -> None:
    assert path.exists()
    assert path.read_bytes()[:8] == PNG_SIGNATURE


def test_generate_eda_figures(clean_trips, raw_frames, tmp_path):
    paths = plots.generate_eda_figures(clean_trips, raw_frames["stations"], tmp_path)
    assert [p.name for p in paths] == [
        "01_distribucion_horaria.png", "02_viajes_dia_semana.png", "03_top_estaciones_demanda.png",
        "04_distribucion_duracion.png", "05_demografia_usuarios.png",
    ]
    for p in paths:
        assert_png(p)


def test_plot_hourly_profile(tmp_path):
    ts = pd.Series(pd.date_range("2025-01-01", periods=48, freq="h"))
    y = np.arange(48, dtype=float)
    path = plots.plot_hourly_profile(ts, y, y + 1, tmp_path / "perfil.png", title="prueba")
    assert_png(path)


def test_plot_model_comparison(tmp_path):
    table = pd.DataFrame({"model": ["a", "b"], "test_mae": [0.9, 0.7]})
    assert_png(plots.plot_model_comparison(table, tmp_path / "comparacion.png"))


@pytest.mark.parametrize("available", [True, False])
def test_style_falls_back_to_default(monkeypatch, available):
    used = []
    monkeypatch.setattr(plots.plt.style, "use", used.append)
    if not available:
        monkeypatch.setattr(plots.plt.style, "available", [])
    plots._style()
    assert used == ["seaborn-v0_8-whitegrid" if available else "default"]


def test_plots_main(session_config, cleaning_result, write_config, monkeypatch, tmp_path):
    cfg = copy.deepcopy(session_config)
    cfg["paths"]["figures_dir"] = str(tmp_path / "figuras")
    monkeypatch.setattr("sys.argv", ["plots", "--config", str(write_config(cfg))])
    plots.main()
    assert len(list((tmp_path / "figuras").glob("*.png"))) == 5
