"""Diferencias columna por columna entre la matriz procesada de Windows (/host) y la de Linux (/work)."""
import math
import numpy as np, pandas as pd
w = pd.read_csv("/host/data/processed/mibici_hourly_demand.csv"); l = pd.read_csv("/work/data/processed/mibici_hourly_demand.csv")
print("forma win", w.shape, "linux", l.shape, "| mismas columnas:", list(w.columns) == list(l.columns))
for c in w.columns:
    if w[c].equals(l[c]):
        continue
    a, b = w[c].to_numpy(), l[c].to_numpy()
    if a.dtype.kind == "f":
        d = np.abs(a - b); n = int((a != b).sum())
        ulps = np.unique(np.round((b[a != b] - a[a != b]) / np.array([math.ulp(x) for x in a[a != b]])))
        print(f"{c}: filas distintas {n} de {len(a)}; dif. máx. {d.max()!r}; diferencias en ulp {ulps}")
        print("   valores distintos (win -> linux):", sorted({(float(x), float(y)) for x, y in zip(a[a != b], b[a != b])}))
    else:
        print(c, "difiere (no flotante)", int((a != b).sum()))
