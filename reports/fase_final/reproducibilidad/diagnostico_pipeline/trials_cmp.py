"""Parámetros/métricas de las corridas de entrenamiento en ambas bases MLflow (solo lectura)."""
import sqlite3
for name, path in (("windows", "file:/host/mlruns.db?mode=ro&immutable=1"), ("linux", "file:/work/mlruns.db?mode=ro")):
    con = sqlite3.connect(path, uri=True)
    runs = con.execute("select r.run_uuid, r.name from runs r join experiments e on r.experiment_id=e.experiment_id "
                       "where e.name='MiBici_Fase2_Modelos' order by r.start_time").fetchall()
    print("==", name, len(runs), "corridas")
    for rid, rname in runs:
        params = dict(con.execute("select key, value from params where run_uuid=?", (rid,)).fetchall())
        mets = dict(con.execute("select key, value from latest_metrics where run_uuid=?", (rid,)).fetchall())
        keys = sorted(k for k in params if k in ("learning_rate","max_iter","max_leaf_nodes","min_samples_leaf","l2_regularization","alpha","n_estimators","max_depth"))
        tm = {k: mets[k] for k in sorted(mets) if "best" in k or "trial" in k or k=="val_mae"}
        print(rname, {k: params[k] for k in keys}, tm)
