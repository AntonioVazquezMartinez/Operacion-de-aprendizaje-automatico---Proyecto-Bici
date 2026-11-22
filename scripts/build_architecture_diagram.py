"""Diagramas de la solución para el README y el reporte final.

Genera:
    reports/figures/00_arquitectura.png   flujo de punta a punta con componentes, herramientas y rol responsable.
    reports/figures/00b_ciclo_mlops.png    ciclo de vida MLOps con las etapas, los roles y la fase en que se construyó.

Las dos cifras que aparecen en el diagrama (número de pruebas y filas de la matriz) se leen de los logs de la
fase final; si un log no existe se muestra "s/d".

Uso:
    python scripts/build_architecture_diagram.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
LOG_DIR = ROOT / "reports" / "fase_final" / "logs"

# Un color por rol (mismos colores en ambas figuras y en el reporte).
ROLE_COLORS = {
    "DE": "#1f5fa8",
    "DS": "#2e8b57",
    "MLE": "#0f8c8c",
    "SWE": "#7b4fb3",
    "SRE": "#d2691e",
}
ROLE_NAMES = {
    "DE": "Data Engineer",
    "DS": "Data Scientist",
    "MLE": "ML Engineer",
    "SWE": "Software Engineer",
    "SRE": "Site Reliability Engineer",
}
INK = "#1f2933"
MUTED = "#52606d"
LANE_BG = "#f4f6f8"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _facts() -> dict:
    """Cifras del diagrama leídas de los logs (sin valores escritos a mano)."""
    tests = re.findall(r"(\d+) passed", _read(LOG_DIR / "10_pytest_cobertura.log"))
    shape = re.search(r"\((\d+), (\d+)\)", _read(LOG_DIR / "02_features.log"))
    return {
        "tests": tests[-1] if tests else "s/d",
        "rows": f"{int(shape.group(1)) / 1e6:.2f} M filas" if shape else "s/d",
    }


def _box(ax, x, y, w, h, title, lines, roles, face="#ffffff", edge="#9aa5b1"):
    """Caja con título, líneas de detalle e insignias de rol (abajo a la derecha)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.8",
                                linewidth=1.3, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x + 0.9, y + h - 1.2, title, ha="left", va="top", fontsize=10, fontweight="bold", color=INK, zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 0.9, y + h - 3.3 - i * 1.75, line, ha="left", va="top", fontsize=7.9, color=MUTED, zorder=4)
    bx = x + w - 0.6
    for role in reversed(roles):
        width = 1.25 + 0.95 * len(role)
        bx -= width
        ax.add_patch(FancyBboxPatch((bx, y + 0.55), width - 0.3, 1.75, boxstyle="round,pad=0.02,rounding_size=0.5",
                                    linewidth=0, facecolor=ROLE_COLORS[role], zorder=4))
        ax.text(bx + (width - 0.3) / 2, y + 1.42, role, ha="center", va="center", fontsize=7.4, color="white",
                fontweight="bold", zorder=5)


def _arrow(ax, p1, p2, color=INK, ls="-", rad=0.0, lw=1.6, label=None, label_xy=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        lx, ly = label_xy if label_xy else ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.8, color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))


def architecture(path: Path) -> None:
    facts = _facts()
    fig, ax = plt.subplots(figsize=(17, 11.8))
    ax.set_xlim(0, 136)
    ax.set_ylim(0, 94)
    ax.axis("off")

    ax.text(68, 93.2, "Arquitectura de la solución MLOps: demanda horaria por estación de MiBici (Equipo 19)",
            ha="center", va="top", fontsize=15, fontweight="bold", color=INK)
    ax.text(68, 89.9, "Flujo de punta a punta con módulos del repositorio, herramientas y rol responsable de cada bloque",
            ha="center", va="top", fontsize=10.5, color=MUTED)
    lx = 9.0
    for role, color in ROLE_COLORS.items():
        ax.add_patch(FancyBboxPatch((lx, 84.6), 1.6, 1.4, boxstyle="round,pad=0.02,rounding_size=0.3", linewidth=0,
                                    facecolor=color))
        ax.text(lx + 2.2, 85.3, f"{role} = {ROLE_NAMES[role]}", ha="left", va="center", fontsize=8.8, color=INK)
        lx += 25.0

    lanes = [(66.0, 16.0, "1. Datos"), (47.0, 16.0, "2. Modelado\ny registro"), (28.0, 16.0, "3. Servicio y\ndespliegue"),
             (9.0, 16.0, "4. Monitoreo\ny retroali-\nmentación")]
    for y, h, label in lanes:
        ax.add_patch(FancyBboxPatch((0.6, y), 134.8, h, boxstyle="round,pad=0,rounding_size=1.2", linewidth=0,
                                    facecolor=LANE_BG, zorder=0))
        ax.text(1.8, y + h / 2, label, ha="left", va="center", fontsize=9.8, fontweight="bold", color=INK)

    W, H = 26.0, 12.0
    xs = [13.5, 43.0, 72.5, 102.0]

    def row(y):
        for a, b in zip(xs[:-1], xs[1:]):
            _arrow(ax, (a + W, y + H / 2), (b, y + H / 2))

    # --- Carril 1: datos
    y1 = 68.0
    _box(ax, xs[0], y1, W, H, "Portal de datos abiertos",
         ["mibici.net: 12 CSV mensuales de 2025", "+ catálogo nomenclatura_2026_10", "URL en config.yaml (data.trip_urls)"],
         ["DE"], face="#eef4fb")
    _box(ax, xs[1], y1, W, H, "Ingesta y limpieza",
         ["src/dataset.py (requests, pandas)", "UTF-8 / latin-1, reglas de calidad", "→ data/raw y data/interim"], ["DE"])
    _box(ax, xs[2], y1, W, H, "Características",
         ["src/features.py: cuadrícula", "estación-hora, calendario cíclico,", "festivos, rezagos 24 h y 168 h"],
         ["DS", "DE"])
    _box(ax, xs[3], y1, W, H, "Versionado de datos",
         ["SHA-256 + mlflow.log_input", "(experimento MiBici_Datos)", "DVC: dvc.yaml + dvc.lock"], ["DE", "MLE"])
    row(y1)

    # --- Carril 2: modelado
    y2 = 49.0
    _box(ax, xs[0], y2, W, H, "Partición y preproceso",
         ["70/15/15 cronológica (sin fuga)", "ColumnTransformer: imputador,", "escalador y TargetEncoder"], ["DS", "MLE"])
    _box(ax, xs[1], y2, W, H, "Entrenamiento y ajuste",
         ["src/modeling/train.py", "Optuna TPE (semilla 42), 4 modelos", "HGB Poisson, RF, Poisson, Dummy"],
         ["DS", "MLE"])
    _box(ax, xs[2], y2, W, H, "MLflow Tracking y Registry",
         ["experimento MiBici_Fase2_Modelos", "mibici-demand-forecaster v1-v3", "alias champion / challenger"], ["MLE"],
         face="#eaf6f6")
    _box(ax, xs[3], y2, W, H, "Exportación del champion",
         ["src/modeling/export.py", "models/champion (en Git)", "+ feature store y patrones oro"],
         ["MLE", "SRE"])
    row(y2)
    _arrow(ax, (xs[3] + W / 2, y1), (xs[0] + W / 2, y2 + H), color=MUTED, ls="--",
           label=f"matriz procesada ({facts['rows']})", label_xy=(71.0, 64.6))

    # --- Carril 3: servicio (de derecha a izquierda)
    y3 = 30.0
    _box(ax, xs[3], y3, W, H, "API REST FastAPI 1.1.0",
         ["POST /predict (rezagos opcionales)", "feature store: últimas 360 h", "Pydantic v2, /health, /model-info"],
         ["SWE"], face="#f3eefa")
    _box(ax, xs[2], y3, W, H, "Contenedor Docker",
         ["Dockerfile multi-etapa, no root", "build desde clon limpio (CI)", "Docker o Podman, HEALTHCHECK"],
         ["SRE"], face="#fbf1e8")
    _box(ax, xs[1], y3, W, H, "DockerHub",
         ["gusrivasmna/mibici-demand-service", "1.1.0, 1.1.0-model-v1, latest", "docker pull / docker run"],
         ["SRE"], face="#fbf1e8")
    _box(ax, xs[0], y3, W, H, "Consumo operativo",
         ["planeación del rebalanceo", "por estación y hora con 1 a 24 h", "de anticipación"], ["DS"])
    _arrow(ax, (xs[3] + W / 2, y2), (xs[3] + W / 2, y3 + H))
    for a, b in zip(xs[1:][::-1], xs[:-1][::-1]):
        _arrow(ax, (a, y3 + H / 2), (b + W, y3 + H / 2))

    # --- Carril 4: monitoreo
    y4 = 11.0
    _box(ax, xs[0], y4, W, H, "Datos de monitoreo",
         ["escenarios alterados + control", "ventanas semanales de prueba", "src/monitoring/scenarios.py"],
         ["DS", "SRE"])
    _box(ax, xs[1], y4, W, H, "Detección de drift",
         ["PSI, Kolmogorov-Smirnov,", "Wasserstein, calidad de datos", "MAE y WAPE vs. línea base"], ["DS"])
    _box(ax, xs[2], y4, W, H, "Umbrales y alertas",
         ["OK / AVISO / ALERTA (config.yaml)", "reglas R0-R7, con y sin etiquetas", "MLflow: MiBici_Monitoreo"],
         ["SRE", "DS"], face="#fdecec")
    _box(ax, xs[3], y4, W, H, "Acción propuesta",
         ["corregir el pipeline de variables", "o reentrenar y promover", "challenger → champion"], ["MLE", "SRE"])
    _arrow(ax, (xs[0] + W / 2, y3), (xs[0] + W / 2, y4 + H), label="predicciones y datos nuevos",
           label_xy=(xs[0] + W / 2, 26.5))
    row(y4)

    # Retroalimentación: de "Acción propuesta" al entrenamiento, por el margen derecho y el pasillo entre carriles.
    red = "#c0392b"
    gx, gy = 133.2, 45.5
    tx = xs[1] + W / 2
    ax.plot([xs[3] + W, gx, gx, tx], [y4 + H / 2, y4 + H / 2, gy, gy], color=red, lw=1.8, ls="--", zorder=1)
    _arrow(ax, (tx, gy), (tx, y2), color=red, ls="--", lw=1.8)
    ax.text(85.0, gy, "reentrenamiento (regla R3)", ha="center", va="center", fontsize=8, color=red, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))

    # --- Banda transversal
    ax.add_patch(FancyBboxPatch((0.6, 0.4), 134.8, 7.4, boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.0,
                                edgecolor="#9aa5b1", facecolor="#ffffff", zorder=0))
    ax.text(1.8, 4.1, "Transversal", ha="left", va="center", fontsize=9.8, fontweight="bold", color=INK)
    items = [
        ("Git / GitHub", "ramas y pull requests", "SWE"),
        ("GitHub Actions", "pruebas, build e imagen", "SRE"),
        (f"pytest ({facts['tests']} pruebas)", "unitarias + integración", "SWE"),
        ("config.yaml", "parámetros únicos", "SWE"),
        ("Semillas y versiones", "seed 42, deps con ==", "SRE"),
        ("DVC + MLflow + Git", "datos y modelos", "DE"),
    ]
    x = 13.5
    for title, sub, role in items:
        ax.add_patch(FancyBboxPatch((x, 1.2), 19.0, 5.8, boxstyle="round,pad=0.02,rounding_size=0.6", linewidth=1.2,
                                    edgecolor=ROLE_COLORS[role], facecolor="#ffffff", zorder=3))
        ax.text(x + 9.5, 5.3, title, ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK, zorder=4)
        ax.text(x + 9.5, 2.8, f"{sub}  [{role}]", ha="center", va="center", fontsize=7.5, color=MUTED, zorder=4)
        x += 19.8

    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mlops_cycle(path: Path) -> None:
    stages = [
        ("Datos", "ingesta, limpieza, características,\nversionado (SHA-256, DVC)", ["DE", "DS"], "F1 → F2 → FF"),
        ("Experimentación", "Optuna, 4 modelos,\nMLflow Tracking", ["DS", "MLE"], "F1 → F2"),
        ("Empaquetado", "Model Registry, exportación,\npruebas pytest, CI", ["MLE", "SWE"], "F2 → FF"),
        ("Despliegue", "FastAPI, Docker,\nDockerHub", ["SWE", "SRE"], "FF"),
        ("Monitoreo", "PSI/KS/Wasserstein, MAE/WAPE,\numbrales y alertas", ["DS", "SRE"], "FF"),
        ("Retroalimentación", "reentrenar y promover challenger\no corregir el pipeline", ["MLE", "DE"], "FF (propuesta)"),
    ]
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12.2, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")
    r = 7.2
    n = len(stages)
    centers = [(r * math.cos(math.pi / 2 - 2 * math.pi * i / n), r * math.sin(math.pi / 2 - 2 * math.pi * i / n))
               for i in range(n)]
    for i in range(n):
        a = math.pi / 2 - 2 * math.pi * i / n - 0.40
        b = math.pi / 2 - 2 * math.pi * (i + 1) / n + 0.40
        _arrow(ax, (r * math.cos(a), r * math.sin(a)), (r * math.cos(b), r * math.sin(b)), color="#9aa5b1",
               rad=-0.25, lw=2.4)
    for (cx, cy), (title, detail, roles, phase) in zip(centers, stages):
        w, h = 5.9, 3.3
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.5",
                                    linewidth=1.6, edgecolor=ROLE_COLORS[roles[0]], facecolor="white", zorder=3))
        ax.text(cx, cy + 1.1, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK, zorder=4)
        ax.text(cx, cy + 0.0, detail, ha="center", va="center", fontsize=8.2, color=MUTED, zorder=4,
                linespacing=1.25)
        ax.text(cx - 0.2, cy - 1.18, "  ".join(roles), ha="right", va="center", fontsize=8.6, fontweight="bold",
                color=ROLE_COLORS[roles[0]], zorder=4)
        ax.text(cx + 0.2, cy - 1.18, phase, ha="left", va="center", fontsize=8.2, color=INK, zorder=4, style="italic")
    ax.text(0, 0.9, "Ciclo de vida MLOps", ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(0, -0.35, "del proyecto MiBici", ha="center", va="center", fontsize=11, color=MUTED)
    ax.text(0, -1.6, "F1 = Fase 1   F2 = Fase 2\nFF = Fase final", ha="center", va="center", fontsize=9,
            color=MUTED, linespacing=1.3)
    for k, (role, color) in enumerate(ROLE_COLORS.items()):
        lx, ly = (-11.0 + 7.6 * (k % 3), -11.0 - 1.0 * (k // 3))
        ax.add_patch(FancyBboxPatch((lx, ly - 0.3), 0.6, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    linewidth=0, facecolor=color))
        ax.text(lx + 0.85, ly, f"{role}: {ROLE_NAMES[role]}", ha="left", va="center", fontsize=9, color=INK)
    ax.text(0, 10.8, "Etapas del ciclo, roles responsables (el primero rinde cuentas) y fase en que se construyó",
            ha="center", va="center", fontsize=10, color=MUTED)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    architecture(FIG_DIR / "00_arquitectura.png")
    mlops_cycle(FIG_DIR / "00b_ciclo_mlops.png")
    print(f"Figuras guardadas en {FIG_DIR}: 00_arquitectura.png, 00b_ciclo_mlops.png")


if __name__ == "__main__":
    main()
