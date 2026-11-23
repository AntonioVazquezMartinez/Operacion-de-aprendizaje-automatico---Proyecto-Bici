"""Aísla el efecto numérico del efecto de la búsqueda: en Linux reentrena hist_gradient_boosting con los
hiperparámetros EXACTOS del champion de Windows (sin Optuna) y compara métricas y predicciones."""
import json, struct, time
import numpy as np, pandas as pd
from src.config import load_config, set_global_seeds
from src.features import load_features, model_input_columns, temporal_split
from src.modeling.pipeline import build_pipeline, evaluate
cfg = load_config("/work/config_linux.yaml"); set_global_seeds(cfg["seed"])
win = json.load(open("/app/models/champion_metadata.json"))
params = {k: v for k, v in win["hyperparameters"].items() if k not in ("loss", "early_stopping")}
df = load_features(cfg); split = temporal_split(df, cfg); feats = model_input_columns(cfg); y = cfg["features"]["target"]
tr, va, te = (df[split == s].reset_index(drop=True) for s in ("train", "val", "test"))
tv = pd.concat([tr, va], ignore_index=True)
t0 = time.time()
val = evaluate(va[y], build_pipeline(cfg, "hist_gradient_boosting", params).fit(tr[feats], tr[y]).predict(va[feats]), "val")
final = build_pipeline(cfg, "hist_gradient_boosting", params).fit(tv[feats], tv[y])
test = evaluate(te[y], final.predict(te[feats]), "test")
ref = pd.read_csv("/app/models/reference_predictions.csv")
pred = np.clip(final.predict(ref[win["input_columns"]]), 0, None); exp = ref["demand_pred"].to_numpy(float)
same = sum(struct.pack("<d", a) == struct.pack("<d", b) for a, b in zip(pred, exp))
out = {"params": params, "seconds": round(time.time() - t0, 1),
       "metrics_linux_fixed_params": {**val, **test}, "metrics_windows": win["metrics"],
       "abs_diff": {k: abs(v - win["metrics"][k]) for k, v in {**val, **test}.items() if k in win["metrics"]},
       "reference_500": {"max_abs_diff": float(np.abs(pred - exp).max()), "mean_abs_diff": float(np.abs(pred - exp).mean()),
                         "rows_bitwise_identical": int(same)}}
print(json.dumps(out, indent=2))
json.dump(out, open("/diag/fixed_params_linux.json", "w"), indent=2)
