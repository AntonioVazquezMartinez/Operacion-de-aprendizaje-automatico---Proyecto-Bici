"""Escritura del reporte de drift: drift_report.json (completo), drift_summary.csv y drift_report.md.

Todo número del Markdown sale del diccionario de resultados; los textos fijos solo explican qué prueba
cada escenario y cómo leer las tablas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.monitoring.figures import VAR_LABELS
from src.monitoring.rules import ACTIONS

FAMILY_LABELS = {"control": "Control", "corrupcion_entradas": "Corrupción de entradas",
                 "cambio_real": "Cambio real"}
WHAT_IT_TESTS = {
    "control": "Sin alteración: mide si los umbrales dan falsas alarmas con datos normales.",
    "missing_lags": "Falla del pipeline: rezagos vacíos que el imputador rellena con la mediana de entrenamiento.",
    "scale_lags": "Falla del pipeline: rezagos en otra escala (unidades o agregación equivocadas).",
    "clock_shift": ("Falla del pipeline: calendario calculado con otra zona horaria. No cambia ninguna "
                    "distribución marginal (cada hora sigue apareciendo igual de seguido), solo su relación "
                    "con la demanda; por eso se vigila el perfil horario."),
    "demand_change": ("Cambio real: más o menos viajes de forma sostenida; se recalculan la cuadrícula, los "
                      "rezagos y el objetivo desde los viajes con la lógica de MiBiciFeatureEngineer."),
    "time_window": "Cambio real: periodo observado con un patrón estacional distinto (sin datos sintéticos).",
    "new_stations": ("Cambio real: estaciones con Id nunca visto que abren al inicio de la ventana: sin "
                     "historial la primera semana (rezagos en 0) y sin codificación en el TargetEncoder."),
}
THRESHOLD_LABELS = {
    "psi": "PSI por variable",
    "ks_statistic": "Estadístico D de Kolmogorov-Smirnov",
    "wasserstein_norm": "Wasserstein / desv. est. de la referencia",
    "missing_rate": "Aumento de la tasa de faltantes",
    "count_domain_violation_rate": "Filas con rezagos de conteo no enteros o negativos",
    "unseen_category_rate": "Filas con estación no vista",
    "hour_profile_shift_hours": "Desfase del perfil horario (horas)",
    "mae_degradation": "Degradación relativa del MAE",
    "wape_degradation": "Degradación relativa del WAPE",
}
THRESHOLD_WHY = {
    "psi": "Convención usual de la industria (riesgo crediticio); regla práctica, no prueba estadística.",
    "ks_statistic": "Tamaño del efecto: el p-valor es ~0 con cientos de miles de filas. Criterio del equipo.",
    "wasserstein_norm": "Desplazamiento de 0.10 / 0.25 desviaciones estándar. Criterio del equipo.",
    "missing_rate": "La referencia no tiene faltantes; 1 % ya indica una falla del pipeline.",
    "count_domain_violation_rate": "Los rezagos de 24 h y 168 h son conteos; mismo criterio que los faltantes.",
    "unseen_category_rate": "1 % de filas equivale a ~4 de 370 estaciones.",
    "hour_profile_shift_hours": "Cualquier desfase indica reloj/zona horaria equivocados.",
    "mae_degradation": "Lo prometido en el ML Canvas (sección 9): degradación del MAE.",
    "wape_degradation": "El MAE baja solo cuando baja la demanda; el WAPE corrige por el nivel.",
}


def _clean(obj: Any) -> Any:
    """Convierte tipos de NumPy a nativos y NaN/inf a None para un JSON válido."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    return obj


def _pct(x: float | None) -> str:
    return "n/d" if x is None else f"{x:+.1%}".replace("%", " %")


def _var(name: str) -> str:
    return VAR_LABELS.get(name, name).replace("\n", " ")


def _deciding(variables: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Variables que deciden el nivel de drift (sin las informativas de calendario)."""
    return {v: m for v, m in variables.items() if not m.get("informative")}


def _w_max(variables: Dict[str, Dict[str, Any]]) -> float:
    """Wasserstein normalizada máxima entre las variables continuas que deciden."""
    vals = [m["wasserstein_norm"] for m in _deciding(variables).values() if not m.get("discrete")]
    return max(vals) if vals else float("nan")


def summary_table(report: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for s in report["scenarios"]:
        perf, deg, q = s["performance"], s["degradation"], s["quality"]
        variables = _deciding(s["variables"])
        top = max(variables, key=lambda v: variables[v]["psi"])
        rows.append({
            "escenario": s["name"], "familia": s["family"], "nivel": s["decision"]["level"],
            "regla": s["decision"]["rule"], "nivel_sin_etiquetas": s["decision_without_labels"]["level"],
            "regla_sin_etiquetas": s["decision_without_labels"]["rule"],
            "nivel_calidad": s["levels"]["quality"], "nivel_drift": s["levels"]["drift"],
            "nivel_desempeno": s["levels"]["performance"],
            "filas": s["rows"], "mae": perf["mae"], "delta_mae": deg["mae"], "rmse": perf["rmse"], "r2": perf["r2"],
            "poisson_deviance": perf["poisson_deviance"], "wape": perf["wape"], "delta_wape": deg["wape"],
            "sesgo": perf["bias"], "demanda_media": perf["mean_demand"], "prediccion_media": perf["mean_prediction"],
            "psi_max": variables[top]["psi"], "variable_psi_max": top,
            "ks_max": max(m["ks_statistic"] for m in variables.values()),
            "wasserstein_max": _w_max(variables),
            "faltantes_max": q["max_missing_increase"], "estaciones_no_vistas": q["unseen_category_rate"],
            "desfase_perfil_h": q["hour_profile"]["shift_hours"],
            "rezagos_fuera_de_dominio": q.get("count_domain_violation_rate"),
            "variables_con_drift": ";".join(s["drifted_variables"]),
            "calendario_con_cambio": ";".join(s.get("drifted_informative_variables", [])),
            "accion": s["decision"]["action"],
        })
    return pd.DataFrame(rows)


def _scenario_text(s: Dict[str, Any], base: Dict[str, float]) -> List[str]:
    perf, deg, q, variables = s["performance"], s["degradation"], s["quality"], _deciding(s["variables"])
    top = sorted(variables, key=lambda v: variables[v]["psi"], reverse=True)[:3]
    lines = [f"### {s['name']} — {s['decision']['level']}", "",
             f"- Qué prueba: {WHAT_IT_TESTS.get(s['type'], s.get('description', ''))}",
             f"- Configuración: {s.get('description', '')} Parámetros: `{json.dumps(s['params'], ensure_ascii=False)}`;"
             f" {s['rows']:,} filas ({s['period']['start']} a {s['period']['end']}).",
             f"- Desempeño: MAE {perf['mae']:.4f} ({_pct(deg['mae'])} vs. {base['mae']:.4f}), WAPE "
             f"{perf['wape']:.4f} ({_pct(deg['wape'])}), R² {perf['r2']:.4f}, sesgo {_pct(perf['bias'])} "
             f"(demanda media {perf['mean_demand']:.3f} vs. predicción media {perf['mean_prediction']:.3f}).",
             "- Mayor PSI: " + ", ".join(f"{_var(v)} {variables[v]['psi']:.3f}" for v in top)
             + f"; KS máx. {max(m['ks_statistic'] for m in variables.values()):.3f}; Wasserstein norm. máx. "
             f"{_w_max(variables):.3f} (variables que deciden; calendario en el JSON).",
             f"- Calidad: faltantes +{q['max_missing_increase']:.1%}, rezagos fuera de dominio "
             f"{(q.get('count_domain_violation_rate') or 0):.1%}, filas con estación no vista "
             f"{q['unseen_category_rate']:.1%} ({q['unseen_stations']} estaciones), desfase del perfil horario "
             f"{q['hour_profile']['shift_hours']:+.0f} h (correlación sin desfase "
             f"{q['hour_profile']['corr_at_zero']:.3f})."
             .replace("%", " %"),
             f"- Niveles: calidad {s['levels']['quality']}, drift {s['levels']['drift']}, desempeño "
             f"{s['levels']['performance']} → regla `{s['decision']['rule']}`. Sin etiquetas: "
             f"{s['decision_without_labels']['level']} (`{s['decision_without_labels']['rule']}`).",
             f"- Acción propuesta: {s['decision']['action']}"]
    if deg["mae"] < 0 < deg["wape"]:
        lines.append(f"- Nota: el MAE baja ({_pct(deg['mae'])}) pero el WAPE sube ({_pct(deg['wape'])}): la demanda "
                     "bajó de nivel y el MAE absoluto, por sí solo, ocultaría que el error relativo empeoró.")
    lines.append("")
    return lines


def to_markdown(report: Dict[str, Any]) -> str:
    base, ref = report["baseline"], report["reference"]
    bp, rp = base["performance"], ref["performance_final_model"]
    table = summary_table(report)
    out = [
        "# Reporte de data drift y pérdida de desempeño — MiBici (Equipo 19)", "",
        f"Generado automáticamente por `python -m src.monitoring.drift` el {report['generated_at']} "
        f"(semilla {report['seed']}, modelo `{report['model_dir']}`, {report['runtime_seconds']} s). "
        "Todos los números salen de `drift_report.json`.", "",
        "## Referencia y línea base", "",
        f"- **Referencia de distribución:** conjunto de `{ref['split']}` ({ref['rows']:,} filas, {ref['start']} a "
        f"{ref['end']}). Es lo que pide la tarea y el periodo más reciente que conoce el modelo; contra él se "
        "calculan PSI, KS y Wasserstein.",
        f"- **Línea base de desempeño:** conjunto de `{base['split']}` ({base['rows']:,} filas, {base['start']} a "
        f"{base['end']}): MAE {bp['mae']:.4f}, RMSE {bp['rmse']:.4f}, R² {bp['r2']:.4f}, devianza de Poisson "
        f"{bp['poisson_deviance']:.4f}, WAPE {bp['wape']:.4f}, sesgo {_pct(bp['bias'])}.",
        f"- El modelo final se entrenó con train + val, así que su MAE sobre validación ({rp['mae']:.4f}) es "
        "optimista (vio esos datos); por eso el desempeño se compara contra prueba, que nunca vio.",
        "- Los escenarios se generan a partir del conjunto de prueba, de modo que la diferencia contra la línea "
        "base se debe solo a la alteración.", "",
    ]
    check = report.get("reconstruction_check")
    if check:
        out += [f"**Verificación de la reconstrucción desde viajes:** reconstruir la ventana de prueba desde los "
                f"viajes limpios sin alterar da {check['rows_rebuilt']:,} filas (matriz procesada: "
                f"{check['rows_baseline']:,}) con diferencia máxima {check['max_abs_diff']:.3g} en "
                f"{', '.join(check['columns'])} → {'idéntica' if check['ok'] else 'DIFERENTE'}. Así los escenarios "
                "de cambio real parten exactamente de la misma línea base.", ""]

    out += ["## Resumen por escenario", "",
            "| Escenario | Familia | Nivel | MAE | ΔMAE | WAPE | ΔWAPE | Variables con drift | Nivel sin etiquetas "
            "| Acción (regla) |",
            "|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for _, r in table.iterrows():
        drifted = ", ".join(_var(v) for v in r["variables_con_drift"].split(";") if v) or "ninguna"
        out.append(f"| {r['escenario']} | {FAMILY_LABELS.get(r['familia'], r['familia'])} | **{r['nivel']}** | "
                   f"{r['mae']:.4f} | {_pct(r['delta_mae'])} | {r['wape']:.4f} | {_pct(r['delta_wape'])} | "
                   f"{drifted} | {r['nivel_sin_etiquetas']} | `{r['regla']}` |")
    out += ["", "Nivel = el más severo entre calidad de datos, drift de entradas y desempeño. "
            "\"Nivel sin etiquetas\" = lo que diría el monitoreo en producción antes de conocer la demanda real "
            "(solo calidad y drift de entradas).", ""]

    out += ["## Dos familias de escenarios", "",
            "- **Corrupción de entradas:** la demanda real no cambia; falla el pipeline que calcula las variables. "
            "Si el desempeño cae, la solución es arreglar el pipeline, no reentrenar (reentrenar con datos corruptos "
            "aprendería el error).",
            "- **Cambio real de distribución:** cambia el comportamiento de los usuarios o del sistema; las variables "
            "y el objetivo se recalculan de forma coherente. Si el desempeño cae, el modelo quedó desactualizado y la "
            "acción es reentrenar y promover un challenger.", ""]

    out += ["## Umbrales (config.yaml → monitoring.thresholds)", "",
            "| Señal | Aviso (≥) | Alerta (≥) | Justificación |", "|---|---:|---:|---|"]
    for key, t in report["thresholds"].items():
        out.append(f"| {THRESHOLD_LABELS.get(key, key)} | {t['warning']} | {t['alert']} | "
                   f"{THRESHOLD_WHY.get(key, '')} |")
    st = report["settings"]
    out += ["", f"PSI con {st['psi_bins']} intervalos por cuantiles de la referencia (variables con ≤ "
            f"{st['discrete_max_unique']} valores distintos: un intervalo por valor), faltantes como categoría propia "
            f"y épsilon {st['psi_epsilon']} en intervalos vacíos. KS sobre una submuestra fija de "
            f"{st['ks_sample_rows']:,} filas por lado: con tantas filas el p-valor es prácticamente 0 ante cualquier "
            "diferencia, así que se decide con el estadístico D (tamaño del efecto) y el p-valor solo se reporta en "
            "el JSON. Los umbrales se fijaron antes de ver los resultados y no se recalibraron.", "",
            "Tres ajustes de diseño (ningún umbral se movió) se hicieron después de la primera ejecución; los dos "
            "primeros porque el escenario de control salía en AVISO solo por `is_holiday`:",
            "1. La Wasserstein normalizada no decide en variables discretas: en una binaria rara la desviación "
            "estándar es pequeña y una diferencia mínima de proporciones parece un desplazamiento grande; ahí "
            "deciden PSI y KS (que en una binaria es |p − q|).",
            "2. Las variables de calendario ("
            + ", ".join(f"`{v}`" for v in (st.get("informative_variables") or []))
            + ") se reportan como informativas y no deciden el nivel de drift: son funciones deterministas de la "
            "fecha y su cambio solo refleja qué días caen en la ventana. Su falla real (reloj o zona horaria "
            "equivocados) la detecta el desfase del perfil horario.",
            "3. Se agregó la validación de dominio de los rezagos de conteo (enteros no negativos) porque, sin ella, "
            "multiplicar los rezagos por una constante (error de unidades) se confundía con un aumento real de "
            "demanda y la regla proponía reentrenar. Limitación: un factor entero (p. ej. ×2) no rompe el dominio "
            "y seguiría pareciendo un cambio real hasta comparar los rezagos contra los viajes registrados.", ""]

    out += ["## Reglas de decisión (en orden; gana la primera que aplica)", ""]
    for rule, action in ACTIONS.items():
        out.append(f"- `{rule}`: {action}")
    out += ["", "Las reglas funcionan sin valor real (R1, R2, R6, R7): en producción la demanda de una hora se "
            "conoce después, así que la primera alarma sale de calidad y drift de entradas; el desempeño confirma "
            "cuando llegan las etiquetas.", ""]

    out += ["## Interpretación por escenario", ""]
    for s in report["scenarios"]:
        out += _scenario_text(s, bp)

    windows = report.get("windows") or []
    if windows:
        out += ["## Vista temporal: conjunto de prueba en ventanas semanales (sin alterar)", "",
                "| Ventana | Inicio | Filas | MAE | ΔMAE | WAPE | ΔWAPE | Sesgo | PSI máx. (variable) | Nivel "
                "| Regla |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
        informative = set(report["settings"].get("informative_variables") or [])
        for w in windows:
            top = max((v for v in w["psi"] if v not in informative), key=w["psi"].get)
            p, d = w["performance"], w["degradation"]
            out.append(f"| {w['window']} | {w['start'][:10]} | {w['rows']:,} | {p['mae']:.4f} | {_pct(d['mae'])} | "
                       f"{p['wape']:.4f} | {_pct(d['wape'])} | {_pct(p['bias'])} | {w['psi'][top]:.3f} ({_var(top)}) "
                       f"| {w['decision']['level']} | `{w['decision']['rule']}` |")
        out += ["", "ΔMAE y ΔWAPE contra el conjunto de prueba completo; la última ventana puede estar incompleta. "
                "PSI máx. sin variables de calendario (informativas).", ""]

    out += ["## Figuras", ""] + [f"- `reports/figures/{name}`" for name in report.get("figures", [])] + [""]
    return "\n".join(out)


def write_reports(report: Dict[str, Any], output_dir: Path) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = _clean(report)
    json_path = output_dir / "drift_report.json"
    json_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output_dir / "drift_summary.csv"
    summary_table(report).to_csv(csv_path, index=False, encoding="utf-8")
    md_path = output_dir / "drift_report.md"
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return [json_path, md_path, csv_path]
