"""Gráficas del análisis exploratorio y de evaluación de modelos.

Uso:
    python -m src.plots      # genera las figuras del EDA en reports/figures a partir de data/interim
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz gráfica (terminal, CI y MLflow)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config, base_arg_parser, load_config  # noqa: E402

DAY_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def _style() -> None:
    style = "seaborn-v0_8-whitegrid"
    plt.style.use(style if style in plt.style.available else "default")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- EDA
def plot_hourly_distribution(df_trips: pd.DataFrame, path: Path) -> Path:
    hours = df_trips["Inicio_del_viaje"].dt.hour.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hours.index, hours.values, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.axvspan(7.5, 9.5, color="orange", alpha=0.25, label="Pico matutino")
    ax.axvspan(17.5, 19.5, color="red", alpha=0.20, label="Pico vespertino")
    ax.set(title="Distribución de viajes por hora del día", xlabel="Hora del día", ylabel="Viajes")
    ax.set_xticks(range(24))
    ax.legend()
    return _save(fig, path)


def plot_weekday_volume(df_trips: pd.DataFrame, path: Path) -> Path:
    counts = df_trips["Inicio_del_viaje"].dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(DAY_NAMES, counts.values, color=["#2ca02c"] * 5 + ["#d62728"] * 2, edgecolor="black")
    ax.bar_label(bars, fmt="%d")
    ax.set(title="Volumen de viajes: días laborales vs. fin de semana", ylabel="Viajes")
    return _save(fig, path)


def plot_top_stations(df_trips: pd.DataFrame, df_stations: pd.DataFrame, path: Path, top: int = 10) -> Path:
    top_df = df_trips["Origen_Id"].value_counts().head(top).rename_axis("id").reset_index(name="viajes")
    top_df = top_df.merge(df_stations[["id", "name"]], on="id", how="left")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["name"].astype(str), top_df["viajes"], color="#9467bd", edgecolor="black")
    ax.invert_yaxis()
    ax.set(title=f"Top {top} estaciones con mayor demanda (origen)", xlabel="Extracciones")
    return _save(fig, path)


def plot_duration_distribution(df_trips: pd.DataFrame, path: Path) -> Path:
    dur = df_trips["duracion_minutos"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dur, bins=40, color="#ff7f0e", edgecolor="black", alpha=0.8)
    ax.axvline(dur.median(), color="blue", linestyle="--", label=f"Mediana: {dur.median():.1f} min")
    ax.axvline(dur.mean(), color="darkred", label=f"Media: {dur.mean():.1f} min")
    ax.set(title="Duración de los viajes", xlabel="Minutos", ylabel="Frecuencia", xlim=(0, 50))
    ax.legend()
    return _save(fig, path)


def plot_demographics(df_trips: pd.DataFrame, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    gender = df_trips["Genero"].value_counts()
    ax1.pie(gender.values, labels=gender.index, autopct="%1.1f%%", startangle=140)
    ax1.set_title("Proporción por género")
    ax2.hist(df_trips["Edad"], bins=25, color="#16a085", edgecolor="black")
    ax2.set(title="Edad de los usuarios", xlabel="Años", ylabel="Frecuencia")
    return _save(fig, path)


def generate_eda_figures(df_trips: pd.DataFrame, df_stations: pd.DataFrame, figures_dir: Path) -> List[Path]:
    """Genera las 5 figuras del EDA de la Fase 1 a partir de los viajes limpios."""
    _style()
    df_trips = df_trips.copy()
    df_trips["Inicio_del_viaje"] = pd.to_datetime(df_trips["Inicio_del_viaje"])
    figures_dir = Path(figures_dir)
    return [
        plot_hourly_distribution(df_trips, figures_dir / "01_distribucion_horaria.png"),
        plot_weekday_volume(df_trips, figures_dir / "02_viajes_dia_semana.png"),
        plot_top_stations(df_trips, df_stations, figures_dir / "03_top_estaciones_demanda.png"),
        plot_duration_distribution(df_trips, figures_dir / "04_distribucion_duracion.png"),
        plot_demographics(df_trips, figures_dir / "05_demografia_usuarios.png"),
    ]


# --------------------------------------------------------------------------- modelos
def plot_hourly_profile(fecha_hora: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, path: Path,
                        title: str) -> Path:
    """Demanda total real vs. predicha por hora del día (suma de todas las estaciones)."""
    _style()
    df = pd.DataFrame({"hour": pd.to_datetime(fecha_hora).dt.hour.values, "real": y_true, "predicha": y_pred})
    profile = df.groupby("hour")[["real", "predicha"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(profile.index, profile["real"], marker="o", label="Real")
    ax.plot(profile.index, profile["predicha"], marker="s", linestyle="--", label="Predicha")
    ax.set(title=title, xlabel="Hora del día", ylabel="Demanda media por estación")
    ax.set_xticks(range(24))
    ax.legend()
    return _save(fig, path)


def plot_model_comparison(results: pd.DataFrame, path: Path, metric: str = "test_mae") -> Path:
    """Barras con una métrica de prueba por modelo (menor es mejor)."""
    _style()
    data = results.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(data["model"], data[metric], color="#1f77b4", edgecolor="black")
    ax.bar_label(bars, fmt="%.3f")
    ax.set(title=f"Comparación de modelos ({metric})", ylabel=metric)
    return _save(fig, path)


def main() -> None:
    args = base_arg_parser("Figuras del análisis exploratorio").parse_args()
    config = load_config(args.config)
    df_trips = pd.read_csv(config.file("interim_dir", "interim_file"))
    df_stations = pd.read_csv(config.file("raw_dir", "stations_file"))
    paths = generate_eda_figures(df_trips, df_stations, config.path("figures_dir"))
    print(f"{len(paths)} figuras guardadas en {config.path('figures_dir')}")


if __name__ == "__main__":
    main()
