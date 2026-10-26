"""Módulo de Exploración de Datos (EDA) e Ingeniería de Características (Feature Engineering)."""

import os
from typing import Tuple
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica para renderizado seguro
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.mlflow_logger import MLflowDataVersioner


class MiBiciFeatureEngineer:
    """Genera visualizaciones del EDA y construye el conjunto de características para modelado."""

    def __init__(
        self,
        cleaned_trips_path: str = "data/interim/viajes_cleaned.csv",
        stations_path: str = "data/raw/nomenclatura_estaciones.csv",
        output_dir: str = "data/processed",
        figures_dir: str = "reports/figures",
    ):
        self.cleaned_trips_path = cleaned_trips_path
        self.stations_path = stations_path
        self.output_dir = output_dir
        self.figures_dir = figures_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def generate_eda_figures(self, df_trips: pd.DataFrame, df_stations: pd.DataFrame) -> None:
        """Genera y guarda en disco las gráficas clave del análisis exploratorio."""
        print("--- Generando gráficas del Análisis Exploratorio (Data Scientist) ---")
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Distribución Horaria de Viajes
        hours = df_trips["Inicio_del_viaje"].dt.hour.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(hours.index, hours.values, color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.set_title("Distribución de Viajes por Hora del Día (MiBici GDL)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Hora del Día (0 - 23)", fontsize=12)
        ax.set_ylabel("Cantidad de Viajes", fontsize=12)
        ax.set_xticks(range(0, 24))
        ax.axvspan(7.5, 9.5, color="orange", alpha=0.25, label="Pico Matutino (8:00 - 10:00)")
        ax.axvspan(17.5, 19.5, color="red", alpha=0.20, label="Pico Vespertino (18:00 - 20:00)")
        ax.legend(frameon=True)
        plt.tight_layout()
        fig_path_1 = os.path.join(self.figures_dir, "01_distribucion_horaria.png")
        fig.savefig(fig_path_1, dpi=300)
        plt.close(fig)

        # 2. Distribución por Día de la Semana
        days_map = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
        day_counts = df_trips["Inicio_del_viaje"].dt.dayofweek.map(days_map).value_counts()
        ordered_days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        day_values = [day_counts.get(d, 0) for d in ordered_days]
        colors = ["#2ca02c" if i < 5 else "#d62728" for i in range(7)]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(ordered_days, day_values, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title("Volumen de Viajes: Días Laborables vs. Fin de Semana", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Cantidad de Viajes", fontsize=12)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 20, f"{int(yval)}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        fig_path_2 = os.path.join(self.figures_dir, "02_viajes_dia_semana.png")
        fig.savefig(fig_path_2, dpi=300)
        plt.close(fig)

        # 3. Top 10 Estaciones con Mayor Demanda de Origen
        top_stations = df_trips["Origen_Id"].value_counts().head(10).reset_index()
        top_stations.columns = ["id", "viajes"]
        top_merged = top_stations.merge(df_stations[["id", "name"]], on="id", how="left")
        top_merged["short_name"] = top_merged["name"].str.replace(r"^GDL-\d+\s*", "", regex=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(top_merged))
        ax.barh(y_pos, top_merged["viajes"], color="#9467bd", edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_merged["short_name"], fontsize=11)
        ax.invert_yaxis()
        ax.set_title("Top 10 Estaciones con Mayor Demanda (Origen)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Total de Extracciones / Viajes", fontsize=12)
        plt.tight_layout()
        fig_path_3 = os.path.join(self.figures_dir, "03_top_estaciones_demanda.png")
        fig.savefig(fig_path_3, dpi=300)
        plt.close(fig)

        # 4. Distribución de Duración de Viajes (Histograma)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(df_trips["duracion_minutos"], bins=40, color="#ff7f0e", edgecolor="black", alpha=0.8)
        median_dur = df_trips["duracion_minutos"].median()
        mean_dur = df_trips["duracion_minutos"].mean()
        ax.axvline(median_dur, color="blue", linestyle="--", linewidth=2, label=f"Mediana: {median_dur:.1f} min")
        ax.axvline(mean_dur, color="darkred", linestyle="-", linewidth=2, label=f"Media: {mean_dur:.1f} min")
        ax.set_title("Distribución de la Duración de Viajes (Minutos)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Duración (Minutos)", fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)
        ax.set_xlim(0, 50)
        ax.legend(frameon=True)
        plt.tight_layout()
        fig_path_4 = os.path.join(self.figures_dir, "04_distribucion_duracion.png")
        fig.savefig(fig_path_4, dpi=300)
        plt.close(fig)

        # 5. Demografía de Usuarios (Edad y Género)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Género
        gen_counts = df_trips["Genero"].value_counts()
        ax1.pie(gen_counts.values, labels=gen_counts.index, autopct="%1.1f%%", colors=["#3498db", "#e74c3c", "#95a5a6"], startangle=140)
        ax1.set_title("Proporción por Género", fontsize=13, fontweight="bold")
        # Edad
        ax2.hist(df_trips["Edad"], bins=25, color="#16a085", edgecolor="black", alpha=0.8)
        ax2.set_title("Distribución de Edad de los Usuarios", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Edad (Años)", fontsize=11)
        ax2.set_ylabel("Frecuencia", fontsize=11)
        plt.tight_layout()
        fig_path_5 = os.path.join(self.figures_dir, "05_demografia_usuarios.png")
        fig.savefig(fig_path_5, dpi=300)
        plt.close(fig)

        print(f"5 Gráficas del EDA exportadas exitosamente a: {self.figures_dir}")

    def build_features(self) -> Tuple[pd.DataFrame, str]:
        """Construye el dataset de demanda horaria con variables cíclicas, espaciales y de rezago."""
        print("=== Iniciando Ingeniería de Características (Data Scientist / ML Engineer) ===")
        df_trips = pd.read_csv(self.cleaned_trips_path)
        df_stations = pd.read_csv(self.stations_path)
        df_trips["Inicio_del_viaje"] = pd.to_datetime(df_trips["Inicio_del_viaje"])

        # Generar figuras del EDA
        self.generate_eda_figures(df_trips, df_stations)

        # 1. Agrupación temporal por estación y bloque horario
        df_trips["fecha_hora"] = df_trips["Inicio_del_viaje"].dt.floor("h")
        demand_series = (
            df_trips.groupby(["Origen_Id", "fecha_hora"])
            .size()
            .reset_index(name="demand_count")
        )

        # 2. Construcción de una cuadrícula completa (Estaciones x Horas) para modelar demanda cero
        # Solo estaciones activas en el periodo; el catálogo incluye estaciones abiertas después.
        all_stations = demand_series["Origen_Id"].unique()
        min_time = demand_series["fecha_hora"].min()
        max_time = demand_series["fecha_hora"].max()
        all_hours = pd.date_range(start=min_time, end=max_time, freq="h")

        full_index = pd.MultiIndex.from_product(
            [all_stations, all_hours], names=["Origen_Id", "fecha_hora"]
        )
        df_grid = demand_series.set_index(["Origen_Id", "fecha_hora"]).reindex(full_index, fill_value=0).reset_index()

        # 3. Características temporales y de calendario
        df_grid["hour"] = df_grid["fecha_hora"].dt.hour
        df_grid["day_of_week"] = df_grid["fecha_hora"].dt.dayofweek
        df_grid["is_weekend"] = (df_grid["day_of_week"] >= 5).astype(int)
        df_grid["month"] = df_grid["fecha_hora"].dt.month

        # 4. Transformaciones Cíclicas Trigonométricas
        df_grid["sin_hour"] = np.sin(2 * np.pi * df_grid["hour"] / 24.0)
        df_grid["cos_hour"] = np.cos(2 * np.pi * df_grid["hour"] / 24.0)
        df_grid["sin_dayofweek"] = np.sin(2 * np.pi * df_grid["day_of_week"] / 7.0)
        df_grid["cos_dayofweek"] = np.cos(2 * np.pi * df_grid["day_of_week"] / 7.0)

        # 5. Fusión de metadatos de estación
        station_cols = ["id", "latitude", "longitude", "capacity"]
        df_grid = df_grid.merge(df_stations[station_cols], left_on="Origen_Id", right_on="id", how="left")
        df_grid = df_grid.drop(columns=["id"])

        # 6. Variables históricas y de rezago (Lag Features)
        df_grid = df_grid.sort_values(by=["Origen_Id", "fecha_hora"]).reset_index(drop=True)
        df_grid["demand_lag_1h"] = df_grid.groupby("Origen_Id")["demand_count"].shift(1).fillna(0)
        df_grid["demand_lag_24h"] = df_grid.groupby("Origen_Id")["demand_count"].shift(24).fillna(0)

        output_path = os.path.join(self.output_dir, "mibici_hourly_demand.csv")
        df_grid.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Dataset con características generado en: {output_path}")
        print(f"Dimensiones finales: {df_grid.shape[0]} filas, {df_grid.shape[1]} columnas")
        return df_grid, output_path

    def run(self) -> None:
        """Ejecuta el pipeline de features y registra la versión procesada en MLflow."""
        df_features, output_path = self.build_features()

        versioner = MLflowDataVersioner()
        run_id = versioner.log_dataset_version(
            run_name="03_Processed_Feature_Engineering",
            df=df_features,
            file_path=output_path,
            stage="processed",
            extra_params={
                "target_variable": "demand_count",
                "time_aggregation": "hourly",
                "cyclical_transforms": "sin/cos hour & dayofweek",
                "lag_features": "1h, 24h",
            },
            extra_metrics={
                "total_samples": float(len(df_features)),
                "num_features": float(len(df_features.columns) - 1),
                "mean_hourly_demand": round(float(df_features["demand_count"].mean()), 3),
                "max_hourly_demand": float(df_features["demand_count"].max()),
                "zero_demand_pct": round(float((df_features["demand_count"] == 0).mean() * 100), 2),
            },
            tags={"role": "Data Scientist / ML Engineer", "version": "v1.2.0-processed"},
        )
        print(f"Versión PROCESSED registrada en MLflow con Run ID: {run_id}")


if __name__ == "__main__":
    engineer = MiBiciFeatureEngineer()
    engineer.run()
