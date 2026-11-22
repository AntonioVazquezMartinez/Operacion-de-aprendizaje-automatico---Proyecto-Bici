"""Métricas de drift por variable (funciones puras, sin estado ni E/S).

- PSI (Population Stability Index): compara proporciones por intervalo. Los intervalos se definen
  SOLO con la referencia (cuantiles); las variables discretas (pocos valores distintos, p. ej.
  binarias o la hora codificada con seno/coseno) usan cada valor como categoría. Los valores
  faltantes forman una categoría propia, y los valores de la muestra actual que no existen en la
  referencia caen en una categoría "otros". Los intervalos vacíos se sustituyen por un épsilon para
  que el logaritmo sea finito.
- KS (Kolmogorov-Smirnov de dos muestras): con cientos de miles de filas el p-valor es prácticamente
  0 ante cualquier diferencia mínima, así que la decisión usa el estadístico D (tamaño del efecto) y
  el p-valor solo se reporta. Se calcula sobre una submuestra fija (semilla de config.yaml).
- Wasserstein normalizada: distancia de transporte (en unidades de la variable) dividida entre la
  desviación estándar de la referencia, para poder usar un mismo umbral en todas las variables.
- Tasa de faltantes, proporción de categorías no vistas y desfase del perfil horario (coherencia
  entre la hora codificada y la demanda rezagada; detecta relojes o zonas horarias equivocadas, que
  no alteran ninguna distribución marginal).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import stats


def _as_float(values: Iterable) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def is_discrete(reference: Iterable, max_unique: int) -> bool:
    """True si la referencia tiene a lo sumo `max_unique` valores distintos (sin contar faltantes)."""
    ref = _as_float(reference)
    return np.unique(ref[~np.isnan(ref)]).size <= max_unique


def _proportions(reference: np.ndarray, current: np.ndarray, bins: int, max_unique: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Proporciones por categoría/intervalo + [otros, faltantes] para referencia y muestra actual."""
    ref_ok, cur_ok = ~np.isnan(reference), ~np.isnan(current)
    ref_vals, cur_vals = reference[ref_ok], current[cur_ok]
    if ref_vals.size == 0:
        n_bins = 0
        ref_idx = np.empty(0, dtype=int)
        cur_idx = np.full(cur_vals.size, 0, dtype=int)  # todo cae en "otros"
    elif np.unique(ref_vals).size <= max_unique:
        categories = np.unique(ref_vals)
        n_bins = categories.size
        ref_idx = np.searchsorted(categories, ref_vals)
        pos = np.clip(np.searchsorted(categories, cur_vals), 0, n_bins - 1)
        cur_idx = np.where(categories[pos] == cur_vals, pos, n_bins)  # n_bins = "otros"
    else:
        quantiles = np.linspace(0, 1, bins + 1)[1:-1]
        edges = np.unique(np.quantile(ref_vals, quantiles))
        n_bins = edges.size + 1  # intervalos (-inf, e0], (e0, e1], ..., (e_k, inf)
        ref_idx = np.searchsorted(edges, ref_vals, side="left")
        cur_idx = np.searchsorted(edges, cur_vals, side="left")
    missing = n_bins + 1  # n_bins = "otros" (valores ausentes en la referencia), n_bins + 1 = faltantes
    ref_counts = np.bincount(ref_idx, minlength=n_bins + 2).astype(float)
    cur_counts = np.bincount(cur_idx, minlength=n_bins + 2).astype(float)
    ref_counts[missing] += (~ref_ok).sum()
    cur_counts[missing] += (~cur_ok).sum()
    return ref_counts / max(reference.size, 1), cur_counts / max(current.size, 1)


def psi(reference: Iterable, current: Iterable, bins: int = 10, eps: float = 1e-4, max_unique: int = 30) -> float:
    """Population Stability Index: sum((q - p) * ln(q / p)) sobre los intervalos de la referencia.

    `p` = proporciones de la referencia, `q` = de la muestra actual; los intervalos vacíos toman `eps`.
    Es 0 con distribuciones idénticas y crece con el desplazamiento. Convención usual de la industria:
    < 0.10 estable, 0.10-0.25 cambio moderado, > 0.25 cambio importante.
    """
    ref, cur = _as_float(reference), _as_float(current)
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    p, q = _proportions(ref, cur, bins, max_unique)
    p, q = np.maximum(p, eps), np.maximum(q, eps)
    return float(np.sum((q - p) * np.log(q / p)))


def _subsample(values: np.ndarray, max_rows: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if max_rows is None or values.size <= max_rows:
        return values
    return rng.choice(values, size=int(max_rows), replace=False)


def ks_test(reference: Iterable, current: Iterable, sample_rows: Optional[int] = None, seed: int = 42
            ) -> Tuple[float, float]:
    """Estadístico D y p-valor de Kolmogorov-Smirnov de dos muestras (ignora faltantes).

    Con `sample_rows` se usa una submuestra fija de cada lado (misma semilla en cada ejecución).
    """
    ref, cur = _as_float(reference), _as_float(current)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ref, cur = _subsample(ref, sample_rows, rng), _subsample(cur, sample_rows, rng)
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def wasserstein_normalized(reference: Iterable, current: Iterable) -> float:
    """Distancia de Wasserstein-1 dividida entre la desviación estándar de la referencia (ignora faltantes)."""
    ref, cur = _as_float(reference), _as_float(current)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    distance = float(stats.wasserstein_distance(ref, cur))
    std = float(np.std(ref))
    return distance / std if std > 0 else distance


def missing_rate(values: Iterable) -> float:
    """Proporción de valores faltantes (NaN)."""
    arr = _as_float(values)
    return float(np.isnan(arr).mean()) if arr.size else float("nan")


def count_domain_violation_rate(values: np.ndarray) -> float:
    """Proporción de filas con algún conteo fuera de dominio (negativo o no entero); ignora faltantes.

    `values` es una matriz filas x columnas de variables que, por construcción, son conteos de viajes.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] == 0:
        return float("nan")
    ok = np.isnan(arr) | ((arr >= 0) & (np.abs(arr - np.rint(arr)) < 1e-9))
    return float((~ok.all(axis=1)).mean())


def unseen_category_rate(current: Iterable, known: Iterable) -> float:
    """Proporción de filas cuya categoría no estaba en las categorías conocidas por el modelo."""
    cur = np.asarray(list(current) if not isinstance(current, np.ndarray) else current)
    if cur.size == 0:
        return float("nan")
    return float((~np.isin(cur, np.asarray(list(known)))).mean())


def hour_from_cyclic(sin_hour: Iterable, cos_hour: Iterable) -> np.ndarray:
    """Recupera la hora (0-23) desde su codificación cíclica seno/coseno."""
    angle = np.arctan2(_as_float(sin_hour), _as_float(cos_hour))
    return np.mod(np.rint(angle * 24.0 / (2 * np.pi)), 24).astype(int)


def hourly_profile(hours: Iterable, values: Iterable) -> np.ndarray:
    """Media de `values` por hora (vector de 24); horas sin datos o valores faltantes quedan en NaN."""
    h = np.asarray(hours, dtype=int)
    v = _as_float(values)
    ok = ~np.isnan(v)
    sums = np.bincount(h[ok], weights=v[ok], minlength=24)
    counts = np.bincount(h[ok], minlength=24)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def hour_profile_shift(reference_profile: np.ndarray, current_profile: np.ndarray) -> Dict[str, float]:
    """Desfase circular (horas) que mejor alinea el perfil actual con el de referencia.

    Regresa `shift_hours` en [-11, 12] (0 = alineado), la correlación sin desfase y la correlación
    en el mejor desfase. Un reloj corrido N horas produce shift = N aunque las distribuciones
    marginales de cada variable no cambien.
    """
    ref, cur = np.asarray(reference_profile, dtype=float), np.asarray(current_profile, dtype=float)
    if np.isnan(ref).all() or np.isnan(cur).all():
        return {"shift_hours": 0.0, "corr_at_zero": float("nan"), "corr_best": float("nan")}
    ref = np.nan_to_num(ref, nan=np.nanmean(ref))
    cur = np.nan_to_num(cur, nan=np.nanmean(cur))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # El perfil actual en la hora codificada h equivale al de referencia en h - shift.
    corrs = [corr(ref, np.roll(cur, -k)) for k in range(24)]
    best = int(np.argmax(corrs))
    shift = best if best <= 12 else best - 24
    if corrs[0] >= corrs[best] - 1e-12:
        shift = 0
    return {"shift_hours": float(shift), "corr_at_zero": corrs[0], "corr_best": corrs[best]}


def variable_drift(reference: Iterable, current: Iterable, bins: int = 10, eps: float = 1e-4, max_unique: int = 30,
                   ks_sample_rows: Optional[int] = None, seed: int = 42) -> Dict[str, float]:
    """Todas las métricas de drift de una variable numérica."""
    ref, cur = _as_float(reference), _as_float(current)
    ks_stat, ks_p = ks_test(ref, cur, ks_sample_rows, seed)
    return {
        "psi": psi(ref, cur, bins=bins, eps=eps, max_unique=max_unique),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "wasserstein_norm": wasserstein_normalized(ref, cur),
        "missing_rate_reference": missing_rate(ref),
        "missing_rate": missing_rate(cur),
        "mean_reference": float(np.nanmean(ref)) if (~np.isnan(ref)).any() else float("nan"),
        "mean": float(np.nanmean(cur)) if (~np.isnan(cur)).any() else float("nan"),
        "discrete": bool(is_discrete(ref, max_unique)),
    }
