"""Diagnóstico: separa la suma de árboles (escala log, `_raw_predict`) de la función de enlace exp().

Uso: python diag_exp.py <raíz con models/> <salida.json>  (en el venv de Windows y dentro del contenedor).
"""
import json, math, sys, platform
import numpy as np, pandas as pd, mlflow.sklearn, sklearn
root = sys.argv[1]; out = sys.argv[2]
model = mlflow.sklearn.load_model(f"{root}/models/champion")
meta = json.load(open(f"{root}/models/champion_metadata.json"))
ref = pd.read_csv(f"{root}/models/reference_predictions.csv")
X = ref[meta["input_columns"]].astype({c: "int64" for c in ["capacity", "Origen_Id", "is_weekend", "is_holiday"]})
est = model[-1] if hasattr(model, "steps") else model
Xt = model[:-1].transform(X) if hasattr(model, "steps") else X
raw = est._raw_predict(Xt).ravel()
pred = np.clip(model.predict(X), 0, None)
res = {"platform": platform.platform(), "numpy": np.__version__, "sklearn": sklearn.__version__,
       "raw": [float(v).hex() for v in raw], "pred": [float(v).hex() for v in pred],
       "np_exp": [float(v).hex() for v in np.exp(raw)], "math_exp": [math.exp(float(v)).hex() for v in raw]}
json.dump(res, open(out, "w"))
print(type(model), "steps" if hasattr(model, "steps") else "", type(est))
try:
    np.show_runtime()
except Exception as e:
    print(e)
