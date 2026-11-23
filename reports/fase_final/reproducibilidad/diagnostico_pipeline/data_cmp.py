"""Compara los CSV generados en Linux (/work) con los del host Windows (/host, solo lectura)."""
import hashlib, sys
import pandas as pd
def sha(b): return hashlib.sha256(b).hexdigest()
pairs = [("data/interim/viajes_cleaned.csv", "data/interim/viajes_cleaned.csv"),
         ("data/processed/mibici_hourly_demand.csv", "data/processed/mibici_hourly_demand.csv"),
         ("data/raw/nomenclatura_estaciones.csv", "data/raw/nomenclatura_estaciones.csv"),
         ("models/champion/stations.csv", "models/champion/stations.csv"),
         ("models/reference_predictions.csv", "models/reference_predictions.csv")]
for w_rel, l_rel in pairs:
    wb = open(f"/host/{w_rel}", "rb").read(); lb = open(f"/work/{l_rel}", "rb").read()
    print(f"== {w_rel}: bytes win {len(wb)} linux {len(lb)} | CRLF win {wb.count(b'\r\n')} linux {lb.count(b'\r\n')}")
    print(f"   sha win {sha(wb)}\n   sha linux {sha(lb)}\n   sha win con CRLF->LF {sha(wb.replace(b'\r\n', b'\n'))}"
          f" | idénticos tras normalizar fin de línea: {wb.replace(b'\r\n', b'\n') == lb}")
w = pd.read_csv("/host/models/reference_predictions.csv"); l = pd.read_csv("/work/models/reference_predictions.csv")
cols = [c for c in w.columns if c != "demand_pred"]
print("referencia: mismas filas/entradas:", w[cols].equals(l[cols]), "| columnas distintas:",
      [c for c in cols if not w[c].equals(l[c])])
