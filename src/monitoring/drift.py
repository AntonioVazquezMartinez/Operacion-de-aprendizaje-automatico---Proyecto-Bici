"""Simulación de data drift y detección de pérdida de desempeño del modelo desplegado.

Línea base: el champion se reentrenó con train + val, así que la línea base de desempeño es el
periodo de prueba (8 nov - 31 dic 2025), que el modelo nunca vio. La distribución de referencia
de las variables es la de los datos de entrenamiento (train + val).

Conjuntos de monitoreo (todos derivados del periodo de prueba, con semilla fija):
    base_prueba                 sin cambios (línea base)
    estacional_diciembre        drift natural: 15-31 de diciembre (vacaciones, festivos)
    rezagos_inflados_x2         desplazamiento de medias: los rezagos llegan al doble
                                (p. ej., viajes duplicados en el feed que alimenta el feature store)
    rezagos_faltantes_40pct     features faltantes: el feature store no entrega rezagos en 40 % de filas
    caida_demanda_40pct         cambio real de la demanda (lluvias, obras, cierre de ciclovías):
                                la demanda observada cae 40 % y las variables de entrada no lo anticipan

Para cada conjunto se calculan MAE/RMSE/R² y el sesgo de la predicción contra la línea base, el PSI y el estadístico KS de
cada variable contra la referencia, el porcentaje de valores faltantes y un nivel de alerta con
la acción propuesta. Además se monitorea el periodo de prueba semana por semana.

Uso:
    python -m src.monitoring.drift
    python -m src.monitoring.drift --no-mlflow --evidently
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.config import PROJ_ROOT, Config, base_arg_parser, load_config  # noqa: E402
from src.features import lag_columns, load_features, temporal_split  # noqa: E402
from src.modeling.pipeline import evaluate  # noqa: E402
from src.serving.bundle import serving_settings  # noqa: E402
from src.serving.service import ModelService  # noqa: E402

DEFAULTS = {
    "experiment": "MiBici_Monitoreo",
    "output_dir": "reports/drift",
    "reference_sample_rows": 200000,
    "psi_bins": 10,
    "psi_warning": 0.10,          # cambio moderado
    "psi_alert": 0.25,            # cambio significativo
    "drift_share_alert": 0.30,    # fracción de variables con PSI >= psi_alert
    "mae_warning_pct": 10.0,      # aumento relativo del MAE contra la línea base
    "mae_alert_pct": 25.0,
    "bias_warning_pct": 10.0,     # sesgo: (predicción media - demanda media) / demanda media
    "bias_alert_pct": 20.0,
    "missing_alert_pct": 5.0,     # % de filas con alguna variable faltante
    "lag_shift_factor": 2.0,
    "missing_fraction": 0.40,
    "demand_drop_fraction": 0.40,
    "seasonal_start": "2025-12-15",
}

LEVELS = ["OK", "ADVERTENCIA", "ALERTA"]


def monitoring_settings(config: Config) -> Dict[str, Any]:
    return {**DEFAULTS, **(config.get("monitoring") or {})}


def monitored_features(config: Config) -> List[str]:
    """Variables de entrada con variación en el tiempo (ubicación y capacidad son fijas por estación)."""
    feats = config["features"]
    base = [c for c in feats["numeric_base"] if c not in ("latitude", "longitude", "capacity")]
    return base + lag_columns(config) + list(feats["binary"])


# --------------------------------------------------------------------------- estadísticos
def psi(reference: pd.Series, current: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index. Variables con pocos valores distintos se tratan como categorías.

    Regla práctica: < 0.10 estable, 0.10-0.25 cambio moderado, >= 0.25 cambio significativo.
    Los valores faltantes se excluyen (se miden aparte con el porcentaje de faltantes).
    """
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    uniques = np.unique(ref)
    if len(uniques) <= bins:
        cats = np.union1d(uniques, np.unique(cur))
        ref_pct = np.array([(ref == c).mean() for c in cats])
        cur_pct = np.array([(cur == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_pct = np.histogram(ref, edges)[0] / len(ref)
        cur_pct = np.histogram(cur, edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, columns: List[str], bins: int,
                seed: int, ks_rows: int = 50000) -> pd.DataFrame:
    rows = []
    for col in columns:
        ref, cur = reference[col], current[col]
        ks = ks_2samp(ref.dropna().sample(min(ks_rows, ref.notna().sum()), random_state=seed),
                      cur.dropna().sample(min(ks_rows, cur.notna().sum()), random_state=seed))
        rows.append({"feature": col, "psi": psi(ref, cur, bins), "ks_statistic": float(ks.statistic),
                     "ref_mean": float(ref.mean()), "cur_mean": float(cur.mean()),
                     "missing_pct": float(cur.isna().mean() * 100)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- escenarios
def build_scenarios(test: pd.DataFrame, config: Config, settings: Dict[str, Any]
                    ) -> Dict[str, Tuple[pd.DataFrame, str, str]]:
    """Regresa {nombre: (datos, tipo de drift, descripción)}."""
    rng = np.random.default_rng(config["seed"])
    target = config["features"]["target"]
    lags = lag_columns(config)
    scenarios: Dict[str, Tuple[pd.DataFrame, str, str]] = {
        "base_prueba": (test.copy(), "ninguno", "Periodo de prueba sin cambios (línea base)"),
    }

    start = pd.Timestamp(settings["seasonal_start"])
    seasonal = test[test["fecha_hora"] >= start].copy()
    if seasonal.empty:  # otro periodo de datos (p. ej. en pruebas): último cuarto del periodo de prueba
        start = test["fecha_hora"].quantile(0.75)
        seasonal = test[test["fecha_hora"] >= start].copy()
    scenarios["estacional_diciembre"] = (
        seasonal, "estacional (natural)",
        f"Datos reales desde {start.date()}: vacaciones decembrinas, Navidad y fin de año")

    factor = float(settings["lag_shift_factor"])
    shifted = test.copy()
    shifted[lags] = shifted[lags] * factor
    scenarios[f"rezagos_inflados_x{factor:g}"] = (
        shifted, "covariables: desplazamiento de medias",
        f"Los rezagos llegan multiplicados por {factor:g} (p. ej., viajes duplicados en el feed); "
        "la demanda real no cambia")

    frac = float(settings["missing_fraction"])
    missing = test.copy()
    mask = rng.random(len(missing)) < frac
    missing.loc[mask, lags] = np.nan
    scenarios[f"rezagos_faltantes_{int(frac * 100)}pct"] = (
        missing, "calidad de datos: features faltantes",
        f"El feature store no entrega los rezagos en {frac:.0%} de las filas; el pipeline los imputa con la mediana")

    drop = float(settings["demand_drop_fraction"])
    concept = test.copy()
    concept[target] = rng.binomial(concept[target].to_numpy(dtype=int), 1 - drop)
    scenarios[f"caida_demanda_{int(drop * 100)}pct"] = (
        concept, "concepto (relación X -> y)",
        f"La demanda real cae {drop:.0%} (lluvias, obras o cierres) y las variables de entrada no lo anticipan")
    return scenarios


# --------------------------------------------------------------------------- reglas de alerta
def assess(row: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[str, str]:
    """Nivel de alerta y acción propuesta a partir de desempeño, drift y faltantes."""
    mae_pct, missing = row["mae_change_pct"], row["missing_pct"]
    share, max_psi, target_psi = row["drift_share"], row["max_feature_psi"], row["target_psi"]
    bias = abs(row["bias_pct"])

    perf_alert = mae_pct >= settings["mae_alert_pct"] or bias >= settings["bias_alert_pct"]
    perf_warn = mae_pct >= settings["mae_warning_pct"] or bias >= settings["bias_warning_pct"]
    data_alert = share >= settings["drift_share_alert"] or max_psi >= settings["psi_alert"]
    data_warn = max_psi >= settings["psi_warning"]

    if missing >= settings["missing_alert_pct"]:
        return "ALERTA", ("Calidad de datos: revisar el feature pipeline y el feature store (rezagos faltantes). "
                          "No reentrenar con datos incompletos; corregir la fuente y reprocesar.")
    if data_alert and target_psi < settings["psi_warning"]:
        level = "ALERTA" if perf_alert or perf_warn else "ADVERTENCIA"
        return level, ("Drift en las entradas sin cambio en la demanda real: probable error en el cálculo de "
                       "variables. Revisar el feature pipeline antes de considerar reentrenar.")
    if perf_alert:
        return "ALERTA", ("Pérdida de desempeño: reentrenar con los datos más recientes (incluyendo el periodo "
                          "con drift), comparar contra el challenger y promover por alias si mejora el MAE.")
    if perf_warn or data_alert or data_warn or target_psi >= settings["psi_warning"]:
        return "ADVERTENCIA", ("Cambio moderado: aumentar la frecuencia de monitoreo y programar reentrenamiento "
                               "si persiste dos semanas; considerar variables de temporada (vacaciones, clima).")
    return "OK", "Sin acción: el modelo opera dentro de los umbrales."


# --------------------------------------------------------------------------- ejecución
class DriftMonitor:
    def __init__(self, config: Config, bundle_dir: str | Path | None = None):
        self.config = config
        self.settings = monitoring_settings(config)
        bundle = Path(bundle_dir or serving_settings(config)["bundle_dir"])
        self.service = ModelService(bundle if bundle.is_absolute() else PROJ_ROOT / bundle)
        self.target = config["features"]["target"]
        self.features = monitored_features(config)
        out = Path(self.settings["output_dir"])
        self.output_dir = out if out.is_absolute() else PROJ_ROOT / out
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = config.path("figures_dir")

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = load_features(self.config)
        split = temporal_split(df, self.config)
        reference = df[split != "test"]
        n = int(self.settings["reference_sample_rows"])
        if len(reference) > n:
            reference = reference.sample(n=n, random_state=self.config["seed"])
        reference = reference.copy()
        reference["prediction"] = self.service.predict_frame(reference)
        return reference, df[split == "test"].reset_index(drop=True)

    def evaluate_set(self, reference: pd.DataFrame, data: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        data = data.copy()
        data["prediction"] = self.service.predict_frame(data)
        s = self.settings
        table = drift_table(reference, data, self.features, int(s["psi_bins"]), self.config["seed"])
        target_psi = psi(reference[self.target], data[self.target], int(s["psi_bins"]))
        pred_psi = psi(reference["prediction"], data["prediction"], int(s["psi_bins"]))
        metrics = evaluate(data[self.target], data["prediction"], "m")
        summary = {
            "rows": int(len(data)),
            "mean_demand": float(data[self.target].mean()),
            "mean_prediction": float(data["prediction"].mean()),
            "mae": metrics["m_mae"], "rmse": metrics["m_rmse"], "r2": metrics["m_r2"],
            "mae_relative_pct": float(metrics["m_mae"] / max(data[self.target].mean(), 1e-9) * 100),
            "bias_pct": float((data["prediction"].mean() - data[self.target].mean())
                              / max(data[self.target].mean(), 1e-9) * 100),
            "max_feature_psi": float(table["psi"].max()),
            "drift_share": float((table["psi"] >= s["psi_alert"]).mean()),
            "drifted_features": ",".join(table.loc[table["psi"] >= s["psi_alert"], "feature"]),
            "target_psi": target_psi,
            "prediction_psi": pred_psi,
            "missing_pct": float(data[self.features].isna().any(axis=1).mean() * 100),
        }
        return summary, table

    def run(self, use_mlflow: bool = True, use_evidently: bool = False) -> pd.DataFrame:
        print("=== Monitoreo de data drift ===")
        reference, test = self.load()
        scenarios = build_scenarios(test, self.config, self.settings)

        rows, psi_rows = [], []
        baseline_mae = None
        for name, (data, kind, description) in scenarios.items():
            summary, table = self.evaluate_set(reference, data)
            if baseline_mae is None:
                baseline_mae = summary["mae"]
            summary["mae_change_pct"] = (summary["mae"] - baseline_mae) / baseline_mae * 100
            summary["alert_level"], summary["action"] = assess(summary, self.settings)
            rows.append({"scenario": name, "drift_type": kind, "description": description, **summary})
            psi_rows.append(table.assign(scenario=name))
            print(f"  {name:28s} MAE {summary['mae']:.3f} ({summary['mae_change_pct']:+.1f} %) | "
                  f"sesgo {summary['bias_pct']:+.1f} % | "
                  f"PSI máx {summary['max_feature_psi']:.3f} | PSI demanda {summary['target_psi']:.3f} | "
                  f"faltantes {summary['missing_pct']:.1f} % -> {summary['alert_level']}")

        summary_df = pd.DataFrame(rows)
        psi_df = pd.concat(psi_rows, ignore_index=True)
        weekly = self.weekly_monitoring(reference, test, baseline_mae)

        files = self.save(summary_df, psi_df, weekly)
        if use_evidently:
            files += self.evidently_reports(reference, scenarios)
        if use_mlflow:
            self.log_mlflow(summary_df, files)
        return summary_df

    def weekly_monitoring(self, reference: pd.DataFrame, test: pd.DataFrame, baseline_mae: float) -> pd.DataFrame:
        """Monitoreo semana a semana del periodo de prueba (como se haría en producción)."""
        test = test.copy()
        test["prediction"] = self.service.predict_frame(test)
        test["week"] = test["fecha_hora"].dt.to_period("W-SUN").dt.start_time
        # En una ventana semanal la distribución de las variables de calendario es fija por construcción
        # (o incompleta en semanas parciales), así que el drift semanal se mide en los rezagos.
        lags = lag_columns(self.config)
        rows = []
        for week, data in test.groupby("week"):
            m = evaluate(data[self.target], data["prediction"], "w")
            row = {"week_start": week.date().isoformat(), "rows": len(data),
                   "mean_demand": float(data[self.target].mean()), "mae": m["w_mae"],
                   "mae_change_pct": (m["w_mae"] - baseline_mae) / baseline_mae * 100,
                   "bias_pct": float((data["prediction"].mean() - data[self.target].mean())
                                     / data[self.target].mean() * 100),
                   "target_psi": psi(reference[self.target], data[self.target]),
                   "lag24_psi": psi(reference["demand_lag_24h"], data["demand_lag_24h"]),
                   "max_feature_psi": float(max(psi(reference[c], data[c]) for c in lags)),
                   "missing_pct": 0.0}
            row["drift_share"] = float(np.mean([psi(reference[c], data[c]) >= self.settings["psi_alert"]
                                                for c in lags]))
            row["alert_level"], row["action"] = assess(row, self.settings)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ salidas
    def save(self, summary: pd.DataFrame, psi_df: pd.DataFrame, weekly: pd.DataFrame) -> List[Path]:
        out, fig_dir = self.output_dir, self.figures_dir
        files = [out / "drift_summary.csv", out / "psi_por_variable.csv", out / "monitoreo_semanal.csv",
                 out / "alertas.json"]
        summary.to_csv(files[0], index=False, float_format="%.4f")
        psi_df.to_csv(files[1], index=False, float_format="%.4f")
        weekly.to_csv(files[2], index=False, float_format="%.4f")
        thresholds = {k: self.settings[k] for k in ("psi_warning", "psi_alert", "drift_share_alert",
                                                    "mae_warning_pct", "mae_alert_pct", "bias_warning_pct",
                                                    "bias_alert_pct", "missing_alert_pct")}
        alerts = {"model_uri": self.service.card.get("model_uri"), "thresholds": thresholds,
                  "scenarios": summary[["scenario", "alert_level", "mae", "mae_change_pct", "bias_pct", "max_feature_psi",
                                        "target_psi", "missing_pct", "action"]].to_dict("records"),
                  "weekly": weekly[["week_start", "alert_level", "mae", "mae_change_pct", "bias_pct"]].to_dict("records")}
        files[3].write_text(json.dumps(alerts, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        files += self.plots(summary, psi_df, weekly, fig_dir)
        print(f"Resultados en {out} y figuras en {fig_dir}")
        return files

    def plots(self, summary: pd.DataFrame, psi_df: pd.DataFrame, weekly: pd.DataFrame, fig_dir: Path) -> List[Path]:
        colors = {"OK": "#2e7d32", "ADVERTENCIA": "#f9a825", "ALERTA": "#c62828"}
        s = self.settings
        paths = []

        base = summary["mae"].iloc[0]
        fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 4.8))
        bar_colors = [colors[x] for x in summary["alert_level"]]
        bars = ax.bar(summary["scenario"], summary["mae"], color=bar_colors)
        for k, style in (("mae_warning_pct", ":"), ("mae_alert_pct", "--")):
            ax.axhline(base * (1 + s[k] / 100), color="gray", ls=style, lw=1,
                       label=f"+{s[k]:g} % ({'advertencia' if 'warning' in k else 'alerta'})")
        for bar, pct in zip(bars, summary["mae_change_pct"]):
            ax.annotate(f"{pct:+.1f} %", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("MAE (bicicletas por hora)")
        ax.set_title("MAE contra la línea base")
        ax.legend(fontsize=8)
        bars = axb.bar(summary["scenario"], summary["bias_pct"], color=bar_colors)
        for sign in (1, -1):
            axb.axhline(sign * s["bias_warning_pct"], color="gray", ls=":", lw=1)
            axb.axhline(sign * s["bias_alert_pct"], color="gray", ls="--", lw=1)
        axb.axhline(0, color="black", lw=0.8)
        for bar, pct in zip(bars, summary["bias_pct"]):
            axb.annotate(f"{pct:+.1f} %", (bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0)),
                         ha="center", va="bottom", fontsize=9)
        axb.set_ylabel("Sesgo de la predicción (%)")
        axb.set_title("Sesgo: (predicción media - demanda media) / demanda media")
        for a in (ax, axb):
            plt.setp(a.get_xticklabels(), rotation=20, ha="right")
        fig.suptitle("Desempeño del champion por escenario de drift (color = nivel de alerta)")
        paths.append(self._save(fig, fig_dir / "07_drift_mae_escenarios.png"))

        pivot = psi_df.pivot(index="feature", columns="scenario", values="psi")[summary["scenario"]]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        im = ax.imshow(pivot.to_numpy(), cmap="Reds", vmin=0, vmax=max(0.5, float(np.nanmax(pivot.to_numpy()))),
                       aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iat[i, j]
                ax.text(j, i, "n/d" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.3 else "black")
        fig.colorbar(im, ax=ax, label="PSI")
        ax.set_title(f"PSI por variable contra la referencia (alerta >= {s['psi_alert']})")
        paths.append(self._save(fig, fig_dir / "08_drift_psi_variables.png"))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        x = pd.to_datetime(weekly["week_start"])
        week_colors = [colors[lvl] for lvl in weekly["alert_level"]]
        ax1.plot(x, weekly["mae"], color="#1565c0", label="MAE semanal")
        ax1.scatter(x, weekly["mae"], color=week_colors, zorder=3, s=60)
        ax1.axhline(base * (1 + s["mae_warning_pct"] / 100), color="#f9a825", ls=":", lw=1, label="advertencia")
        ax1.axhline(base * (1 + s["mae_alert_pct"] / 100), color="#c62828", ls="--", lw=1, label="alerta")
        ax1.set_ylabel("MAE")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.set_title("Monitoreo semanal del periodo de prueba (nov-dic 2025); color = nivel de alerta")
        ax2.bar(x, weekly["bias_pct"], width=5, color=week_colors)
        for sign in (1, -1):
            ax2.axhline(sign * s["bias_warning_pct"], color="#f9a825", ls=":", lw=1)
            ax2.axhline(sign * s["bias_alert_pct"], color="#c62828", ls="--", lw=1)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_ylabel("Sesgo de la predicción (%)")
        for xi, d in zip(x, weekly["mean_demand"]):
            ax2.annotate(f"dem. {d:.2f}", (xi, 0), xytext=(0, -14), textcoords="offset points", ha="center",
                         fontsize=7, color="dimgray")
        paths.append(self._save(fig, fig_dir / "09_monitoreo_semanal.png"))

        return paths

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def evidently_reports(self, reference: pd.DataFrame, scenarios: Dict[str, Tuple[pd.DataFrame, str, str]]
                          ) -> List[Path]:
        """Reportes HTML de Evidently (opcional; requiere `pip install evidently`)."""
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
        except ImportError:
            print("Evidently no está instalado o su versión no es compatible (se requiere >= 0.7); se omite.")
            return []
        cols = self.features + [self.target]
        paths = []
        for name, (data, _, _) in scenarios.items():
            if name == "base_prueba":
                continue
            snapshot = Report([DataDriftPreset()]).run(data[cols].sample(min(50000, len(data)), random_state=0),
                                                       reference[cols])
            path = self.output_dir / f"evidently_{name}.html"
            snapshot.save_html(str(path))
            paths.append(path)
        print(f"Reportes de Evidently: {len(paths)}")
        return paths

    def log_mlflow(self, summary: pd.DataFrame, files: List[Path]) -> None:
        import mlflow

        from src.tracking import setup_mlflow

        setup_mlflow(self.config, self.settings["experiment"])
        with mlflow.start_run(run_name="06_Simulacion_Data_Drift"):
            mlflow.set_tags({"phase": "Entrega Final", "stage": "monitoring", "role": "Site Reliability Engineer",
                             "model_uri": self.service.card.get("model_uri")})
            mlflow.log_params({k: self.settings[k] for k in ("psi_alert", "psi_warning", "mae_alert_pct",
                                                             "mae_warning_pct", "bias_alert_pct", "bias_warning_pct",
                                                             "missing_alert_pct", "drift_share_alert")})
            for row in summary.to_dict("records"):
                key = row["scenario"].replace(".", "_")
                mlflow.log_metrics({f"{key}_mae": row["mae"], f"{key}_mae_change_pct": row["mae_change_pct"],
                                    f"{key}_bias_pct": row["bias_pct"],
                                    f"{key}_max_psi": row["max_feature_psi"], f"{key}_target_psi": row["target_psi"],
                                    f"{key}_alert": float(LEVELS.index(row["alert_level"]))})
            for f in files:
                mlflow.log_artifact(str(f), artifact_path="drift")
        print("Resultados registrados en MLflow (experimento de monitoreo).")


def main() -> None:
    parser = base_arg_parser("Simulación de data drift y detección de pérdida de desempeño")
    parser.add_argument("--bundle", help="Carpeta del paquete de despliegue (por defecto models/champion)")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--evidently", action="store_true", help="Generar además reportes HTML con Evidently")
    args = parser.parse_args()
    DriftMonitor(load_config(args.config), args.bundle).run(not args.no_mlflow, args.evidently)


if __name__ == "__main__":
    main()
