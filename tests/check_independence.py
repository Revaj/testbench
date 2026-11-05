"""Comprueba que ninguna parte comparte preprocesamiento con otra.

En el stacking con MLP (R3): cada base debe tener su propio ColumnTransformer, sin ningún objeto
transformador en común, y todas deben recibir la tabla cruda (no la salida de otro preprocesamiento).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from variants import VARIANTS  # noqa: E402


def objects(est, seen=None):
    """Todos los estimadores/transformadores alcanzables desde est (recorrido de atributos)."""
    seen = set() if seen is None else seen
    out = []
    stack = [est]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, BaseEstimator):
            out.append(o)
            stack.extend(vars(o).values())
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
    return out


rng = np.random.default_rng(0)
n = 1500
X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
X["lo"] = rng.choice(["x", "y", "z", None], n)
X["hi"] = [f"u{v}" for v in rng.integers(0, 500, n)]
X.loc[rng.random(n) < 0.1, "a"] = np.nan
y = (X["b"] + (X["lo"] == "x") + rng.normal(size=n) > 0.5).astype(int)

m = VARIANTS["R3"]().fit(X, y)
per_base = {name: {id(o) for o in objects(b) if o is not b} for name, b in zip(m.bases_, m.models_)}
ok = True
names = list(per_base)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        shared = per_base[names[i]] & per_base[names[j]]
        print(f"{names[i]:8s} vs {names[j]:8s}: objetos compartidos = {len(shared)}")
        ok &= not shared
for name, b in zip(m.bases_, m.models_):
    ct = b.ct_
    seen_cols = sorted({c for _, _, cols in ct.transformers_ if not isinstance(cols, str) for c in cols})
    raw = all(c in X.columns for c in seen_cols)
    print(f"{name:8s} recibe columnas crudas: {raw}  -> {seen_cols}")
    ok &= raw
print("INDEPENDIENTES" if ok else "COMPARTEN ALGO")
sys.exit(0 if ok else 1)
