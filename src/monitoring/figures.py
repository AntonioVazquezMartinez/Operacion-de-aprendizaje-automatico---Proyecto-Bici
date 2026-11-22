"""Figuras del monitoreo de drift (estilo de src/plots.py; textos en español).

    07_drift_distribuciones.png     referencia vs. escenarios en las variables clave y perfil horario.
    08_drift_psi_por_variable.png   mapa de calor escenario x variable coloreado por nivel de PSI.
    09_drift_desempeno.png          MAE y WAPE por escenario contra la línea base y bandas de aviso/alerta.
    10_drift_ventanas_tiempo.png    MAE, WAPE y PSI por ventana semanal del conjunto de prueba.

El nivel siempre se escribe como texto junto al color (no depende solo del color).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.plots import _save, _style

STATUS = {"OK": "#0ca30c", "AVISO": "#fab219", "ALERTA": "#d03b3b"}
STATUS_TEXT = {"OK": "white", "AVISO": "#0b0b0b", "ALERTA": "white"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100"]
REFERENCE_COLOR = "#8c8c8c"

VAR_LABELS = {
    "sin_hour": "sen(hora)", "cos_hour": "cos(hora)", "sin_dayofweek": "sen(día)", "cos_dayofweek": "cos(día)",
    "latitude": "latitud", "longitude": "longitud", "capacity": "capacidad",
    "demand_lag_24h": "rezago 24 h", "demand_lag_168h": "rezago 168 h",
    "demand_same_hour_mean_7d": "media misma\nhora 7 d", "is_weekend": "fin de\nsemana", "is_holiday": "festivo",
    "prediccion": "predicción",
}
SCENARIO_LABELS = {
    "control": "Control (sin alterar)", "rezagos_faltantes": "Rezagos faltantes", "rezagos_escala": "Rezagos ×1.5",
    "reloj_desfasado": "Reloj desfasado", "aumento_demanda": "Demanda +40 %", "caida_demanda": "Demanda −40 %",
    "vacaciones_diciembre": "Vacaciones dic.", "estaciones_nuevas": "Estaciones nuevas",
}


def _label(name: str) -> str:
    return SCENARIO_LABELS.get(name, name.replace("_", " "))


def _var(name: str) -> str:
    return VAR_LABELS.get(name, name)


def _why(scenario: Dict[str, Any]) -> str:
    """Nivel global y componentes que lo alcanzan (p. ej. "ALERTA: calidad, desempeño")."""
    level = scenario["decision"]["level"]
    if level == "OK":
        return "OK"
    names = {"quality": "calidad", "drift": "drift", "performance": "desempeño"}
    parts = [names[k] for k, v in scenario["levels"].items() if v == level]
    return f"{level}: {', '.join(parts)}"


# --------------------------------------------------------------------------- 07
def plot_distributions(samples: Dict[str, Dict[str, Any]], path: Path) -> Path:
    """Histogramas (proporción) de referencia vs. escenarios y perfil horario del rezago de 24 h."""
    _style()
    preferred = ["control", "rezagos_escala", "aumento_demanda", "caida_demanda", "vacaciones_diciembre"]
    shown = [s for s in preferred if s in samples] or [s for s in samples if s != "referencia"][:5]
    panels = [
        ("demand_lag_24h", np.arange(0, 13) - 0.5, "Rezago de 24 h (viajes/hora)"),
        ("demand_same_hour_mean_7d", np.linspace(0, 8, 33), "Media de la misma hora, 7 días"),
        ("prediccion", np.linspace(0, 8, 33), "Predicción del modelo (viajes/hora)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (var, bins, title) in zip(axes.flat, panels):
        ref = samples["referencia"]["values"][var]
        ref = ref[~np.isnan(ref)]
        ax.hist(np.clip(ref, bins[0], bins[-1] - 1e-9), bins=bins, weights=np.full(ref.size, 1 / ref.size),
                color=REFERENCE_COLOR, alpha=0.35, label="Referencia (validación)")
        for color, name in zip(SERIES, shown):
            vals = samples[name]["values"][var]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            ax.hist(np.clip(vals, bins[0], bins[-1] - 1e-9), bins=bins, weights=np.full(vals.size, 1 / vals.size),
                    histtype="step", linewidth=2, color=color, label=_label(name))
        ax.set_yscale("log")
        ax.set(title=title, xlabel="Valor (el último intervalo agrupa los mayores)", ylabel="Proporción (escala log)")
    axes.flat[0].legend(fontsize=9)

    ax = axes.flat[3]
    ax.plot(range(24), samples["referencia"]["profile"], color=REFERENCE_COLOR, linewidth=4, alpha=0.6,
            label="Referencia (validación)")
    for color, name in zip(SERIES, ["control", "reloj_desfasado", "vacaciones_diciembre"]):
        if name in samples:
            ax.plot(range(24), samples[name]["profile"], marker="o", markersize=4, linewidth=2, color=color,
                    label=_label(name))
    ax.set(title="Perfil horario: rezago de 24 h medio según la hora codificada",
           xlabel="Hora del día (según sen/cos de las variables)", ylabel="Rezago 24 h medio")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9)
    fig.suptitle("Distribuciones de referencia vs. escenarios de monitoreo", fontsize=14)
    return _save(fig, path)


# --------------------------------------------------------------------------- 08
def plot_psi_heatmap(report: Dict[str, Any], path: Path) -> Path:
    """Mapa de calor escenario x variable; color = nivel según el PSI y número = PSI."""
    _style()
    scenarios = report["scenarios"]
    t = report["thresholds"]["psi"]
    variables = list(scenarios[0]["variables"])
    values = np.array([[s["variables"][v]["psi"] for v in variables] for s in scenarios])
    informative = {v for v in variables if scenarios[0]["variables"][v].get("informative")}
    fig, ax = plt.subplots(figsize=(max(10, 1.05 * len(variables) + 4), 0.65 * len(scenarios) + 2.6))
    for i in range(len(scenarios)):
        for j, var in enumerate(variables):
            val = values[i, j]
            level = "ALERTA" if val >= t["alert"] else ("AVISO" if val >= t["warning"] else "OK")
            faded = level == "OK" or var in informative
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=STATUS[level], edgecolor="white", linewidth=2,
                                       alpha=0.3 if faded else 1.0, hatch="//" if var in informative else None))
            text = "≥10" if val >= 10 else (f"{val:.2f}" if val >= 1 else f"{val:.3f}")
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=9,
                    color="#0b0b0b" if faded else STATUS_TEXT[level])
    ax.set_xlim(0, len(variables))
    ax.set_ylim(len(scenarios), 0)
    ax.set_xticks(np.arange(len(variables)) + 0.5)
    ax.set_xticklabels([_var(v) + ("\n(i)" if v in informative else "") for v in variables], fontsize=9)
    for tick, v in zip(ax.get_xticklabels(), variables):
        if v in informative:
            tick.set_color("#6b6b6b")
    ax.set_yticks(np.arange(len(scenarios)) + 0.5)
    ax.set_yticklabels([f"{_label(s['name'])}  [{_why(s)}]" for s in scenarios], fontsize=10)
    ax.grid(False)
    ax.set_title("PSI por variable y escenario (referencia: validación)\n"
                 "Color = nivel según PSI; entre corchetes, nivel global y qué lo provoca; "
                 "(i) rayada = calendario, informativa (no decide)", fontsize=12)
    handles = [Patch(facecolor=STATUS["OK"], alpha=0.35, label=f"OK (PSI < {t['warning']:.2f})"),
               Patch(facecolor=STATUS["AVISO"], label=f"AVISO ({t['warning']:.2f} ≤ PSI < {t['alert']:.2f})"),
               Patch(facecolor=STATUS["ALERTA"], label=f"ALERTA (PSI ≥ {t['alert']:.2f})")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return _save(fig, path)


# --------------------------------------------------------------------------- 09
def _bands(ax, base: float, t: Dict[str, float], vertical: bool = True) -> None:
    span = ax.axvspan if vertical else ax.axhspan
    line = ax.axvline if vertical else ax.axhline
    span(base * (1 + t["warning"]), base * (1 + t["alert"]), color=STATUS["AVISO"], alpha=0.18,
         label=f"Aviso (+{t['warning']:.0%} a +{t['alert']:.0%})")
    span(base * (1 + t["alert"]), base * 3, color=STATUS["ALERTA"], alpha=0.12, label=f"Alerta (≥ +{t['alert']:.0%})")
    line(base, color="black", linestyle="--", linewidth=1.5, label=f"Línea base (prueba) = {base:.3f}")


def plot_performance(report: Dict[str, Any], path: Path) -> Path:
    """MAE y WAPE por escenario (dos paneles, un eje cada uno) contra la línea base."""
    _style()
    scenarios = report["scenarios"]
    base = report["baseline"]["performance"]
    th = report["thresholds"]
    names = [_label(s["name"]) for s in scenarios]
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.55 * len(scenarios) + 3.2))
    for ax, metric, key, title in [(axes[0], "mae", "mae_degradation", "MAE (bicicletas/hora)"),
                                   (axes[1], "wape", "wape_degradation", "WAPE (MAE / demanda media)")]:
        t = th[key]
        vals = np.array([s["performance"][metric] for s in scenarios])
        degr = vals / base[metric] - 1
        levels = ["ALERTA" if d >= t["alert"] else ("AVISO" if d >= t["warning"] else "OK") for d in degr]
        y = np.arange(len(scenarios))
        _bands(ax, base[metric], t)
        ax.barh(y, vals, color=[STATUS[lv] for lv in levels], edgecolor="white", height=0.65)
        xmax = max(vals.max(), base[metric] * (1 + t["alert"])) * 1.38
        for yi, v, d, lv in zip(y, vals, degr, levels):
            ax.text(v + xmax * 0.01, yi, f"{v:.3f} ({d:+.1%}) {lv}", va="center", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set(title=title, xlabel=title.split(" (")[0])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Desempeño del champion por escenario vs. línea base (color = nivel de esa métrica)", fontsize=13)
    return _save(fig, path)


# --------------------------------------------------------------------------- 10
def plot_windows(report: Dict[str, Any], path: Path) -> Path:
    """MAE, WAPE y PSI por ventana semanal del conjunto de prueba (tres paneles, un eje cada uno)."""
    _style()
    windows = report["windows"]
    base = report["baseline"]["performance"]
    th = report["thresholds"]
    x = np.arange(len(windows))
    labels = [f"{w['start'][5:10]}\n{w['decision']['level']}" + ("*" if w["rows"] < max(v["rows"] for v in windows)
                                                                  else "") for w in windows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, metric, key, ylabel in [(axes[0], "mae", "mae_degradation", "MAE (bicicletas/hora)"),
                                    (axes[1], "wape", "wape_degradation", "WAPE")]:
        t = th[key]
        vals = np.array([w["performance"][metric] for w in windows])
        _bands(ax, base[metric], t, vertical=False)
        ax.plot(x, vals, marker="o", markersize=8, linewidth=2, color=SERIES[0], label="Ventana semanal")
        lo = min(vals.min(), base[metric]) * 0.9
        hi = max(vals.max(), base[metric] * (1 + t["alert"])) * 1.08
        ax.set_ylim(lo, hi)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[0].set_title("Monitoreo continuo: conjunto de prueba en ventanas de 7 días (sin alterar)")

    ax = axes[2]
    t = th["psi"]
    series = [("demand_lag_24h", "rezago 24 h"), ("demand_same_hour_mean_7d", "media misma hora 7 d"),
              ("prediccion", "predicción")]
    for color, (var, lab) in zip(SERIES[1:], series):
        ax.plot(x, [w["psi"][var] for w in windows], marker="o", markersize=7, linewidth=2, color=color, label=lab)
    informative = set(report["settings"].get("informative_variables") or [])
    max_psi = [max(v for k, v in w["psi"].items() if k not in informative) for w in windows]
    ax.plot(x, max_psi, linestyle=":", linewidth=2, color="black",
            label="máximo de las variables que deciden (sin calendario)")
    ax.axhline(t["warning"], color=STATUS["AVISO"], linewidth=2, label=f"Aviso PSI = {t['warning']:.2f}")
    ax.axhline(t["alert"], color=STATUS["ALERTA"], linewidth=2, label=f"Alerta PSI = {t['alert']:.2f}")
    ax.set_ylabel("PSI vs. validación")
    ax.set_ylim(0, max(t["alert"] * 1.2, max(max_psi) * 1.1))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Inicio de la ventana (mes-día) y nivel global; * = ventana incompleta")
    return _save(fig, path)


def generate_figures(report: Dict[str, Any], samples: Dict[str, Dict[str, Any]], figures_dir: Path) -> List[Path]:
    figures_dir = Path(figures_dir)
    paths = [plot_distributions(samples, figures_dir / "07_drift_distribuciones.png")]
    if report["scenarios"]:
        paths += [plot_psi_heatmap(report, figures_dir / "08_drift_psi_por_variable.png"),
                  plot_performance(report, figures_dir / "09_drift_desempeno.png")]
    if report["windows"]:
        paths.append(plot_windows(report, figures_dir / "10_drift_ventanas_tiempo.png"))
    return paths
