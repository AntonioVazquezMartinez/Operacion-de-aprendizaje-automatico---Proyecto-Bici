"""Simulación de data drift y detección de pérdida de desempeño del modelo champion.

Flujo:
    1. Referencia de distribución = conjunto de validación; línea base de desempeño = conjunto de prueba
       (sección `monitoring` de config.yaml; el modelo final vio train + val).
    2. Para cada escenario de `monitoring.scenarios` se genera un conjunto de monitoreo, se predice con el
       champion y se calculan drift por variable (PSI, KS, Wasserstein), calidad de datos (faltantes,
       estaciones no vistas, perfil horario) y desempeño (MAE, RMSE, R2, devianza de Poisson, WAPE, sesgo).
    3. Las reglas de src/monitoring/rules.py dan el nivel OK / AVISO / ALERTA y la acción propuesta, con y
       sin valor real (en producción las etiquetas llegan con retraso).
    4. Vista temporal: el conjunto de prueba en ventanas semanales (drift natural, incluye diciembre).
    5. Salidas: reports/drift/drift_report.{json,md}, drift_summary.csv, figuras 07-10 y corridas en MLflow.

Uso:
    python -m src.monitoring.drift
    python -m src.monitoring.drift --scenarios control,reloj_desfasado --no-mlflow
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import PROJ_ROOT, Config, base_arg_parser, load_config, set_global_seeds
from src.features import (MiBiciFeatureEngineer, feature_columns, lag_columns, load_features, model_input_columns,
                          temporal_split)
from src.modeling.pipeline import evaluate
from src.monitoring import metrics as M
from src.monitoring import rules as R
from src.monitoring import scenarios as S

PRED_COL = "prediccion"
FAMILY = {
    "control": "control",
    "missing_lags": "corrupcion_entradas",
    "scale_lags": "corrupcion_entradas",
    "clock_shift": "corrupcion_entradas",
    "demand_change": "cambio_real",
    "time_window": "cambio_real",
    "new_stations": "cambio_real",
}
TRIP_SCENARIOS = {"demand_change", "new_stations"}
# Variables cuya muestra se guarda para la figura de distribuciones (07).
PLOT_VARIABLES = ("demand_lag_24h", "demand_same_hour_mean_7d", PRED_COL)


def _resolve(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJ_ROOT / path


# --------------------------------------------------------------------------- carga
def load_champion(config: Config) -> Any:
    import mlflow.sklearn

    return mlflow.sklearn.load_model(str(_resolve(config["monitoring"]["model_dir"])))


def known_stations(model: Any) -> Optional[np.ndarray]:
    """Categorías de estación que aprendió el TargetEncoder del pipeline (None si no se encuentran)."""
    try:
        encoder = model.named_steps["preprocessor"].named_transformers_["station"].named_steps["encoder"]
        return np.asarray(encoder.categories_[0])
    except (AttributeError, KeyError, IndexError):
        return None


def load_trips(config: Config) -> pd.DataFrame:
    path = config.file("interim_dir", "interim_file")
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: python -m src.dataset")
    return pd.read_csv(path, usecols=["Inicio_del_viaje", "Origen_Id"], parse_dates=["Inicio_del_viaje"])


# --------------------------------------------------------------------------- métricas
def performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Métricas de src.modeling.pipeline.evaluate + WAPE (MAE / demanda media) y sesgo relativo."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    out = {k.replace("m_", "", 1): v for k, v in evaluate(y_true, y_pred, "m").items()}
    total = float(y_true.sum())
    out["wape"] = float(np.abs(y_true - y_pred).sum() / total) if total > 0 else float("nan")
    out["bias"] = float(y_pred.sum() / total - 1) if total > 0 else float("nan")
    out["mean_demand"] = float(y_true.mean())
    out["mean_prediction"] = float(y_pred.mean())
    return out


class Monitor:
    """Contexto común: referencia, línea base, umbrales y cálculo de drift para cualquier conjunto."""

    def __init__(self, config: Config, model: Any, reference: pd.DataFrame, baseline: pd.DataFrame):
        self.config = config
        self.mcfg = config["monitoring"]
        self.seed = int(config["seed"])
        self.thresholds = self.mcfg["thresholds"]
        self.model = model
        self.target = config["features"]["target"]
        self.inputs = model_input_columns(config)
        groups = feature_columns(config)
        self.drift_vars = list(groups["numeric"]) + list(groups["binary"])
        # Variables de calendario: se reportan, pero no deciden el nivel de drift (ver config.yaml).
        self.informative = set(self.mcfg.get("informative_variables") or [])
        self.lags = lag_columns(config)
        # Rezagos que son conteos de viajes (enteros >= 0); el promedio de 7 días no lo es.
        self.count_lags = [f"demand_lag_{h}h" for h in config["features"]["lag_hours"]]
        self.known = known_stations(model)
        if self.known is None:  # respaldo: estaciones de los datos con que se entrenó el modelo final
            self.known = np.unique(reference["Origen_Id"])

        self.reference = reference.copy()
        self.reference[PRED_COL] = self.predict(self.reference)
        self.baseline = baseline.copy()
        self.baseline[PRED_COL] = self.predict(self.baseline)
        self.baseline_perf = performance_metrics(self.baseline[self.target], self.baseline[PRED_COL])
        self.reference_perf = performance_metrics(self.reference[self.target], self.reference[PRED_COL])
        self.reference_profile = self.profile(self.reference)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self.model.predict(df[self.inputs]), dtype=float), 0, None)

    def profile(self, df: pd.DataFrame) -> np.ndarray:
        """Perfil horario de la media del rezago de 24 h según la hora CODIFICADA en las variables."""
        hours = M.hour_from_cyclic(df["sin_hour"], df["cos_hour"])
        return M.hourly_profile(hours, df[self.lags[0]])

    def assess(self, data: pd.DataFrame, with_labels: bool = True) -> Dict[str, Any]:
        """Drift, calidad, desempeño, niveles y decisión de un conjunto de monitoreo (ya con predicción)."""
        mc = self.mcfg
        variables: Dict[str, Dict[str, Any]] = {}
        for var in self.drift_vars + [PRED_COL]:
            vm = M.variable_drift(self.reference[var], data[var], bins=int(mc["psi_bins"]),
                                  eps=float(mc["psi_epsilon"]), max_unique=int(mc["discrete_max_unique"]),
                                  ks_sample_rows=mc.get("ks_sample_rows"), seed=self.seed)
            vm["missing_increase"] = vm["missing_rate"] - vm["missing_rate_reference"]
            vm["level"] = R.variable_level(vm, self.thresholds)
            vm["informative"] = var in self.informative
            variables[var] = vm

        unseen = M.unseen_category_rate(data["Origen_Id"].to_numpy(), self.known)
        shift = M.hour_profile_shift(self.reference_profile, self.profile(data))
        max_missing = max(v["missing_increase"] for v in variables.values())
        domain = M.count_domain_violation_rate(data[self.count_lags].to_numpy(float)) if self.count_lags else None
        quality = R.quality_level(max_missing, unseen, shift["shift_hours"], self.thresholds, domain)
        quality.update({"max_missing_increase": max_missing, "unseen_category_rate": unseen,
                        "count_domain_violation_rate": domain,
                        "unseen_stations": int(len(set(np.unique(data["Origen_Id"])) - set(self.known))),
                        "hour_profile": shift})
        deciding = {v: m for v, m in variables.items() if not m["informative"]}
        drift_level = R.max_level(*(m["level"] for m in deciding.values()))

        perf = degradation = None
        perf_level = None
        if with_labels:
            perf = performance_metrics(data[self.target], data[PRED_COL])
            base = self.baseline_perf
            degradation = {"mae": perf["mae"] / base["mae"] - 1, "wape": perf["wape"] / base["wape"] - 1}
            perf_level = R.performance_level(degradation["mae"], degradation["wape"], self.thresholds)

        return {
            "rows": int(len(data)),
            "period": {"start": str(data["fecha_hora"].min()), "end": str(data["fecha_hora"].max())},
            "variables": variables,
            "drifted_variables": [v for v, m in deciding.items() if m["level"] != "OK"],
            "drifted_informative_variables": [v for v, m in variables.items()
                                              if m["informative"] and m["level"] != "OK"],
            "quality": quality,
            "performance": perf,
            "degradation": degradation,
            "levels": {"quality": quality["level"], "drift": drift_level, "performance": perf_level},
            "decision": R.decide(quality, drift_level, perf_level),
            "decision_without_labels": R.decide(quality, drift_level, None),
        }


# --------------------------------------------------------------------------- escenarios
def generate_scenario(monitor: Monitor, spec: Dict[str, Any], trips: Optional[pd.DataFrame],
                      stations: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Conjunto de monitoreo de un escenario a partir de la línea base (o de los viajes)."""
    kind, seed, base = spec["type"], monitor.seed, monitor.baseline.drop(columns=[PRED_COL])
    if kind == "control":
        return base.copy()
    if kind == "missing_lags":
        return S.missing_lags(base, monitor.lags, spec["fraction"], seed)
    if kind == "scale_lags":
        return S.scale_lags(base, monitor.lags, spec["factor"])
    if kind == "clock_shift":
        return S.clock_shift(base, spec["hours"])
    if kind == "time_window":
        return S.time_window(base, spec["start"], spec["end"])
    if kind in TRIP_SCENARIOS:
        if trips is None or stations is None:
            raise ValueError(f"El escenario {kind} necesita los viajes limpios y el catálogo de estaciones")
        engineer = MiBiciFeatureEngineer(monitor.config)
        start, end = base["fecha_hora"].min(), base["fecha_hora"].max()
        station_ids = sorted(base["Origen_Id"].unique())
        if kind == "demand_change":
            altered = S.change_demand(trips, spec["factor"], seed)
            return S.rebuild_features(engineer, altered, stations, station_ids, start, end)
        altered, catalog, grid_ids, _ = S.new_stations(trips, stations, station_ids, spec["fraction"], start,
                                                       int(spec["id_offset"]), seed)
        return S.rebuild_features(engineer, altered, catalog, grid_ids, start, end)
    raise ValueError(f"Tipo de escenario desconocido: {kind}")


def reconstruction_check(monitor: Monitor, trips: pd.DataFrame, stations: pd.DataFrame) -> Dict[str, Any]:
    """Reconstruye la ventana de prueba desde los viajes SIN alterar y la compara con la matriz procesada."""
    base = monitor.baseline.drop(columns=[PRED_COL])
    rebuilt = S.rebuild_features(MiBiciFeatureEngineer(monitor.config), trips, stations,
                                 sorted(base["Origen_Id"].unique()), base["fecha_hora"].min(),
                                 base["fecha_hora"].max())
    cols = [monitor.target] + monitor.lags + ["sin_hour", "cos_hour", "is_weekend", "is_holiday", "capacity"]
    a = base.sort_values(["fecha_hora", "Origen_Id"]).reset_index(drop=True)
    same_shape = len(a) == len(rebuilt)
    max_diff = (float(np.nanmax(np.abs(a[cols].to_numpy(float) - rebuilt[cols].to_numpy(float))))
                if same_shape else float("nan"))
    return {"rows_baseline": int(len(a)), "rows_rebuilt": int(len(rebuilt)), "columns": cols,
            "max_abs_diff": max_diff, "ok": bool(same_shape and max_diff < 1e-9)}


def trips_window(monitor: Monitor, trips: pd.DataFrame) -> pd.DataFrame:
    """Viajes necesarios para reconstruir la línea base (ventana + historial de los rezagos)."""
    start = monitor.baseline["fecha_hora"].min() - pd.Timedelta(hours=S.history_hours(monitor.config))
    end = monitor.baseline["fecha_hora"].max() + pd.Timedelta(hours=1)
    ts = trips["Inicio_del_viaje"]
    return trips.loc[(ts >= start) & (ts < end)].reset_index(drop=True)


def time_windows(monitor: Monitor) -> List[Dict[str, Any]]:
    """Conjunto de prueba en ventanas de `window_days` días con drift y desempeño por ventana."""
    base = monitor.baseline
    days = int(monitor.mcfg.get("window_days", 7))
    offset = ((base["fecha_hora"] - base["fecha_hora"].min()) / pd.Timedelta(days=days)).astype(int)
    out = []
    for k, part in base.groupby(offset, sort=True):
        res = monitor.assess(part)
        out.append({
            "window": int(k), "start": res["period"]["start"], "end": res["period"]["end"], "rows": res["rows"],
            "performance": res["performance"], "degradation": res["degradation"], "levels": res["levels"],
            "decision": res["decision"], "drifted_variables": res["drifted_variables"],
            "psi": {v: m["psi"] for v, m in res["variables"].items()},
            "ks_statistic": {v: m["ks_statistic"] for v, m in res["variables"].items()},
            "wasserstein_norm": {v: m["wasserstein_norm"] for v, m in res["variables"].items()},
            "hour_shift": res["quality"]["hour_profile"]["shift_hours"],
        })
    return out


def _plot_sample(monitor: Monitor, data: pd.DataFrame, max_rows: int = 200_000) -> Dict[str, Any]:
    """Muestra fija de las variables de la figura 07 y perfil horario (para no guardar cada escenario completo)."""
    sample = data.sample(n=min(max_rows, len(data)), random_state=monitor.seed)
    return {"values": {v: sample[v].to_numpy(float) for v in PLOT_VARIABLES},
            "profile": monitor.profile(data)}


# --------------------------------------------------------------------------- ejecución
def run_monitoring(config: Config, use_mlflow: bool = True, scenario_names: Optional[Sequence[str]] = None,
                   features: Optional[pd.DataFrame] = None, trips: Optional[pd.DataFrame] = None,
                   stations: Optional[pd.DataFrame] = None, model: Any = None,
                   output_dir: Optional[Path] = None, figures_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Ejecuta todos los escenarios y escribe reporte, figuras y (opcional) corridas de MLflow.

    Los argumentos `features`, `trips`, `stations` y `model` permiten inyectar datos (pruebas); si se
    omiten se leen de las rutas de config.yaml.
    """
    from src.monitoring import figures as F
    from src.monitoring import report as REP

    t0 = time.perf_counter()
    mcfg = config["monitoring"]
    specs = dict(mcfg["scenarios"])
    if scenario_names:
        unknown = set(scenario_names) - set(specs)
        if unknown:
            raise ValueError(f"Escenarios no definidos en config.yaml: {sorted(unknown)}")
        specs = {name: specs[name] for name in scenario_names}

    print("=== Monitoreo de drift ===")
    if features is None:
        features = load_features(config)
    split = temporal_split(features, config)
    reference = features[split == mcfg["reference_split"]].reset_index(drop=True)
    baseline = features[split == mcfg["baseline_split"]].sort_values(["fecha_hora", "Origen_Id"])
    baseline = baseline.reset_index(drop=True)
    model = model if model is not None else load_champion(config)
    monitor = Monitor(config, model, reference, baseline)
    print(f"Referencia ({mcfg['reference_split']}): {len(reference)} filas "
          f"[{reference['fecha_hora'].min()} - {reference['fecha_hora'].max()}]")
    print(f"Línea base ({mcfg['baseline_split']}): {len(baseline)} filas "
          f"[{baseline['fecha_hora'].min()} - {baseline['fecha_hora'].max()}] "
          f"MAE {monitor.baseline_perf['mae']:.4f} | WAPE {monitor.baseline_perf['wape']:.4f}")

    needs_trips = any(spec["type"] in TRIP_SCENARIOS for spec in specs.values())
    check = None
    if needs_trips:
        if trips is None:
            trips = load_trips(config)
        if stations is None:
            stations = pd.read_csv(config.file("raw_dir", "stations_file"))
        trips = trips_window(monitor, trips)
        check = reconstruction_check(monitor, trips, stations)
        print(f"Reconstrucción desde viajes sin alterar: {check['rows_rebuilt']} filas, "
              f"diferencia máxima {check['max_abs_diff']:.3g} -> {'OK' if check['ok'] else 'DIFERENTE'}")

    results, samples = [], {"referencia": _plot_sample(monitor, monitor.reference)}
    for name, spec in specs.items():
        ts = time.perf_counter()
        data = generate_scenario(monitor, spec, trips, stations)
        if data.empty:
            print(f"  {name}: sin filas en la ventana; se omite")
            continue
        data = data.copy()
        data[PRED_COL] = monitor.predict(data)
        res = monitor.assess(data)
        params = {k: v for k, v in spec.items() if k not in ("type", "description")}
        res = {"name": name, "type": spec["type"], "family": FAMILY[spec["type"]],
               "description": spec.get("description", ""), "params": params, **res}
        results.append(res)
        samples[name] = _plot_sample(monitor, data)
        perf, deg = res["performance"], res["degradation"]
        print(f"  {name:<22} {res['decision']['level']:<7} MAE {perf['mae']:.4f} ({deg['mae']:+.1%}) "
              f"WAPE {perf['wape']:.4f} ({deg['wape']:+.1%}) drift={res['levels']['drift']} "
              f"calidad={res['levels']['quality']} -> {res['decision']['rule']} "
              f"[{time.perf_counter() - ts:.1f} s]")

    windows = time_windows(monitor)
    for w in windows:
        print(f"  ventana {w['window']} {w['start'][:10]}: {w['decision']['level']:<7} "
              f"MAE {w['performance']['mae']:.4f} ({w['degradation']['mae']:+.1%}) "
              f"WAPE ({w['degradation']['wape']:+.1%}) -> {w['decision']['rule']}")

    output_dir = Path(output_dir) if output_dir else _resolve(mcfg["output_dir"])
    figures_dir = Path(figures_dir) if figures_dir else config.path("figures_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_dir": str(mcfg["model_dir"]),
        "seed": monitor.seed,
        "reference": {"split": mcfg["reference_split"], "rows": int(len(reference)),
                      "start": str(reference["fecha_hora"].min()), "end": str(reference["fecha_hora"].max()),
                      "performance_final_model": monitor.reference_perf},
        "baseline": {"split": mcfg["baseline_split"], "rows": int(len(baseline)),
                     "start": str(baseline["fecha_hora"].min()), "end": str(baseline["fecha_hora"].max()),
                     "performance": monitor.baseline_perf},
        "settings": {k: mcfg.get(k) for k in ("psi_bins", "psi_epsilon", "discrete_max_unique", "ks_sample_rows",
                                              "window_days", "informative_variables")},
        "thresholds": mcfg["thresholds"],
        "rules": R.ACTIONS,
        "reconstruction_check": check,
        "scenarios": results,
        "windows": windows,
    }
    figure_paths = F.generate_figures(report, samples, figures_dir)
    report["figures"] = [p.name for p in figure_paths]
    report["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    report_paths = REP.write_reports(report, output_dir)
    for path in report_paths + figure_paths:
        print(f"Artefacto: {path}")

    if use_mlflow:
        log_to_mlflow(config, report, report_paths + figure_paths)
    print(f"Tiempo total: {time.perf_counter() - t0:.1f} s")
    return report


def log_to_mlflow(config: Config, report: Dict[str, Any], artifacts: List[Path]) -> None:
    """Una corrida por escenario (más una de ventanas) en el experimento de monitoreo."""
    import mlflow

    from src.tracking import setup_mlflow

    setup_mlflow(config, config["monitoring"]["experiment_name"])
    common = {"reference_split": report["reference"]["split"], "baseline_split": report["baseline"]["split"],
              "model_dir": report["model_dir"], "seed": report["seed"]}
    for res in report["scenarios"]:
        with mlflow.start_run(run_name=f"drift_{res['name']}"):
            mlflow.log_params({**common, "scenario": res["name"], "scenario_type": res["type"],
                               "family": res["family"], "rows": res["rows"],
                               **{f"param_{k}": v for k, v in res["params"].items()}})
            metrics = {f"{k}": v for k, v in res["performance"].items()}
            metrics.update({f"degradation_{k}": v for k, v in res["degradation"].items()})
            metrics.update({f"psi_{v}": m["psi"] for v, m in res["variables"].items()})
            metrics.update({f"ks_{v}": m["ks_statistic"] for v, m in res["variables"].items()})
            metrics.update({f"wasserstein_{v}": m["wasserstein_norm"] for v, m in res["variables"].items()})
            q = res["quality"]
            metrics.update({"unseen_category_rate": q["unseen_category_rate"],
                            "max_missing_increase": q["max_missing_increase"],
                            "count_domain_violation_rate": q["count_domain_violation_rate"],
                            "hour_profile_shift": q["hour_profile"]["shift_hours"],
                            "n_drifted_variables": float(len(res["drifted_variables"]))})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})
            mlflow.set_tags({"nivel_alerta": res["decision"]["level"], "regla": res["decision"]["rule"],
                             "nivel_sin_etiquetas": res["decision_without_labels"]["level"],
                             "familia": res["family"], "accion": res["decision"]["action"][:250],
                             "phase": "Fase final", "role": "ML Engineer / Monitoreo"})
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"escenario_{res['name']}.json"
                path.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
                mlflow.log_artifact(str(path), artifact_path="escenario")
            for art in artifacts:
                mlflow.log_artifact(str(art), artifact_path="figuras" if art.suffix == ".png" else "reporte")
    informative = set(report["settings"].get("informative_variables") or [])
    with mlflow.start_run(run_name="drift_ventanas_semanales"):
        mlflow.log_params(common)
        for w in report["windows"]:
            step = w["window"]
            mlflow.log_metrics({"mae": w["performance"]["mae"], "wape": w["performance"]["wape"],
                                "degradation_mae": w["degradation"]["mae"],
                                "degradation_wape": w["degradation"]["wape"],
                                "max_psi": max(v for k, v in w["psi"].items() if k not in informative)},
                               step=step)
        mlflow.set_tags({"phase": "Fase final", "role": "ML Engineer / Monitoreo"})
        for art in artifacts:
            mlflow.log_artifact(str(art), artifact_path="figuras" if art.suffix == ".png" else "reporte")
    print(f"Corridas registradas en MLflow (experimento {config['monitoring']['experiment_name']}).")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    parser = base_arg_parser("Simulación de data drift y detección de pérdida de desempeño")
    parser.add_argument("--scenarios", help="Lista separada por comas (por defecto: todos los de config.yaml)")
    parser.add_argument("--no-mlflow", action="store_true", help="No registrar corridas en MLflow")
    args = parser.parse_args()
    config = load_config(args.config)
    set_global_seeds(config["seed"])
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()] if args.scenarios else None
    run_monitoring(config, use_mlflow=not args.no_mlflow, scenario_names=names)


if __name__ == "__main__":
    main()
