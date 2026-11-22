"""Compara windows_venv.json y contenedor_linux.json (salidas de diag_exp.py) y mide el error en ulp de cada
exp() contra el valor exacto (decimal con 60 dígitos, correctamente redondeado)."""
import json
import math
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 60
here = Path(__file__).parent
w = json.loads((here / "windows_venv.json").read_text())
l = json.loads((here / "contenedor_linux.json").read_text())
print(f"Windows: {w['platform']} | Linux: {l['platform']}")
for key in ("raw", "np_exp", "math_exp", "pred"):
    diff = [i for i, (a, b) in enumerate(zip(w[key], l[key])) if a != b]
    print(f"{key:<9} filas distintas Windows vs Linux: {len(diff)} de {len(w[key])} {diff}")
for name, r in (("Windows", w), ("Linux", l)):
    print(f"{name}: predict != np.exp(raw): {sum(a != b for a, b in zip(r['pred'], r['np_exp']))}; "
          f"np.exp != math.exp: {sum(a != b for a, b in zip(r['np_exp'], r['math_exp']))}")
print("fila | suma de árboles (log) | Windows | Linux | Linux - Windows (ulp) | error Windows (ulp) | error Linux (ulp)")
for i, (a, b) in enumerate(zip(w["pred"], l["pred"])):
    if a == b:
        continue
    x, fa, fb = float.fromhex(w["raw"][i]), float.fromhex(a), float.fromhex(b)
    exact = Decimal(x).exp()
    err_a = float(abs(Decimal(fa) - exact) / Decimal(math.ulp(fa)))
    err_b = float(abs(Decimal(fb) - exact) / Decimal(math.ulp(fb)))
    print(f"{i} | {x!r} | {fa!r} | {fb!r} | {(fb - fa) / math.ulp(fa):+.0f} | {err_a:.4f} | {err_b:.4f}")
