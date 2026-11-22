"""Umbrales, niveles de alerta (OK / AVISO / ALERTA) y acción propuesta mediante reglas explícitas.

Se evalúan tres componentes por separado:
    - calidad de datos: faltantes, categorías no vistas y desfase del perfil horario (fallas del pipeline);
    - drift de entradas: PSI, estadístico KS y Wasserstein normalizada por variable (y de la predicción);
    - desempeño: degradación relativa del MAE y del WAPE contra la línea base (requiere el valor real).

Las reglas funcionan también SIN valor real (`performance=None`): en producción la demanda observada
de una hora solo se conoce después, así que la primera alarma siempre sale de calidad y drift de
entradas; la de desempeño llega con retraso.

Un valor igual al umbral ya dispara el nivel (comparación >=).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

LEVELS = ("OK", "AVISO", "ALERTA")
_RANK = {name: i for i, name in enumerate(LEVELS)}

# Métricas de drift por variable que deciden el nivel de la variable (el p-valor de KS solo se reporta).
DRIFT_METRICS = ("psi", "ks_statistic", "wasserstein_norm")

ACTIONS = {
    "R1_pipeline": ("Revisar el pipeline de variables y la calidad de datos (faltantes, unidades, zona horaria, "
                    "esquema) antes de reentrenar: reentrenar con entradas corruptas solo aprendería el error."),
    "R2_catalogo": ("Actualizar el catálogo de estaciones y verificar el alta de las estaciones nuevas; "
                    "reentrenar cuando acumulen historial (>= 7 días para los rezagos) y vigilar su error aparte."),
    "R3_reentrenar": ("Reentrenar con datos recientes, evaluar el challenger contra el champion en la misma "
                      "ventana y, si mejora, promoverlo moviendo el alias challenger -> champion en el Registry."),
    "R4_programar": ("Programar reentrenamiento con datos recientes y aumentar la frecuencia de monitoreo; "
                     "promover el challenger solo si supera al champion."),
    "R5_concepto": ("Posible concept drift (cambia la relación entradas-demanda sin drift claro en las entradas): "
                    "revisar las etiquetas y el calendario (vacaciones, eventos) y reentrenar incorporándolos."),
    "R6_vigilar": ("Drift en entradas sin degradación del desempeño: vigilar, aumentar la frecuencia de monitoreo "
                   "y confirmar con las etiquetas en cuanto lleguen."),
    "R7_sin_etiquetas": ("Drift fuerte en entradas y aún sin valor real: preparar un reentrenamiento (challenger) y "
                         "evaluarlo con las primeras etiquetas disponibles; no promover sin evidencia de desempeño."),
    "R0_ok": "Sin acción: continuar con el monitoreo programado.",
}


def max_level(*levels: Optional[str]) -> str:
    """Nivel más severo (ignora None)."""
    present = [lv for lv in levels if lv is not None]
    return max(present, key=lambda lv: _RANK[lv]) if present else "OK"


def level_from_value(value: Optional[float], warning: float, alert: float) -> Optional[str]:
    """OK / AVISO / ALERTA según dos umbrales crecientes (>= dispara). None o NaN -> None (sin dato)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if value >= alert:
        return "ALERTA"
    if value >= warning:
        return "AVISO"
    return "OK"


def _threshold(thresholds: Mapping[str, Any], name: str) -> Dict[str, float]:
    spec = thresholds[name]
    return {"warning": float(spec["warning"]), "alert": float(spec["alert"])}


def variable_level(metrics: Mapping[str, float], thresholds: Mapping[str, Any]) -> str:
    """Nivel de drift de una variable: el más severo entre PSI, estadístico KS y Wasserstein normalizada.

    En variables discretas (`metrics["discrete"]`) la Wasserstein normalizada no decide: en una binaria rara
    la desviación estándar es pequeña (p. ej. 0.14 con 2 % de festivos) y una diferencia de 2 puntos
    porcentuales ya parecería un desplazamiento de 0.14 desviaciones. Ahí bastan PSI y KS (= |p - q|).
    """
    levels = []
    for name in DRIFT_METRICS:
        if name == "wasserstein_norm" and metrics.get("discrete", False):
            continue
        if name in metrics:
            t = _threshold(thresholds, name)
            levels.append(level_from_value(metrics[name], t["warning"], t["alert"]))
    return max_level(*levels)


def quality_level(max_missing_increase: float, unseen_rate: float, hour_shift: float,
                  thresholds: Mapping[str, Any], domain_violation_rate: Optional[float] = None
                  ) -> Dict[str, Optional[str]]:
    """Niveles de calidad de datos por señal y el nivel agregado.

    `domain_violation_rate`: filas cuyos rezagos de conteo no son enteros no negativos (unidades o escala
    equivocadas); None si no se evalúa.
    """
    t_miss = _threshold(thresholds, "missing_rate")
    t_unseen = _threshold(thresholds, "unseen_category_rate")
    t_shift = _threshold(thresholds, "hour_profile_shift_hours")
    out = {
        "missing": level_from_value(max_missing_increase, t_miss["warning"], t_miss["alert"]),
        "unseen": level_from_value(unseen_rate, t_unseen["warning"], t_unseen["alert"]),
        "hour_shift": level_from_value(abs(hour_shift), t_shift["warning"], t_shift["alert"]),
        "domain": None,
    }
    if domain_violation_rate is not None and "count_domain_violation_rate" in thresholds:
        t_dom = _threshold(thresholds, "count_domain_violation_rate")
        out["domain"] = level_from_value(domain_violation_rate, t_dom["warning"], t_dom["alert"])
    out["level"] = max_level(out["missing"], out["unseen"], out["hour_shift"], out["domain"])
    return out


def performance_level(mae_degradation: Optional[float], wape_degradation: Optional[float],
                      thresholds: Mapping[str, Any]) -> Optional[str]:
    """Nivel de desempeño: el peor entre la degradación relativa del MAE y la del WAPE (None sin etiquetas).

    Solo cuenta el empeoramiento (una mejora da OK). El WAPE (MAE / demanda media) se agrega porque el
    MAE en bicicletas/hora baja por sí solo cuando baja la demanda, aunque el error relativo empeore.
    """
    if mae_degradation is None and wape_degradation is None:
        return None
    t_mae = _threshold(thresholds, "mae_degradation")
    t_wape = _threshold(thresholds, "wape_degradation")
    return max_level(level_from_value(mae_degradation, t_mae["warning"], t_mae["alert"]),
                     level_from_value(wape_degradation, t_wape["warning"], t_wape["alert"]))


def decide(quality: Mapping[str, Optional[str]], drift: str, performance: Optional[str]) -> Dict[str, str]:
    """Nivel global y acción propuesta. Las reglas se evalúan en orden; gana la primera que aplica.

    R1  calidad (faltantes, dominio o perfil horario) >= AVISO -> revisar el pipeline de variables.
    R2  categorías no vistas >= AVISO                          -> actualizar catálogo de estaciones.
    R3  drift >= AVISO y desempeño = ALERTA                     -> reentrenar y promover challenger.
    R4  drift >= AVISO y desempeño = AVISO                      -> programar reentrenamiento.
    R5  drift = OK y desempeño >= AVISO                         -> posible concept drift.
    R6  drift >= AVISO y desempeño = OK                         -> vigilar.
    R7  sin etiquetas y drift = ALERTA                          -> preparar challenger, esperar etiquetas.
        sin etiquetas y drift = AVISO                           -> vigilar (R6).
    R0  todo OK                                                 -> sin acción.
    """
    q_level = quality.get("level", "OK") or "OK"
    level = max_level(q_level, drift, performance)
    drift_on = _RANK[drift] >= 1
    if max_level(quality.get("missing"), quality.get("hour_shift"), quality.get("domain")) != "OK":
        rule = "R1_pipeline"
    elif quality.get("unseen") not in (None, "OK"):
        rule = "R2_catalogo"
    elif performance is None:
        rule = "R7_sin_etiquetas" if drift == "ALERTA" else ("R6_vigilar" if drift_on else "R0_ok")
    elif drift_on and performance == "ALERTA":
        rule = "R3_reentrenar"
    elif drift_on and performance == "AVISO":
        rule = "R4_programar"
    elif not drift_on and performance != "OK":
        rule = "R5_concepto"
    elif drift_on:
        rule = "R6_vigilar"
    else:
        rule = "R0_ok"
    return {"level": level, "rule": rule, "action": ACTIONS[rule]}
