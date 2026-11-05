"""Casos raros que podrían tumbar la entrega: clases con 1-3 ejemplos, columnas vacías,
constantes, booleanas, categorías nuevas en test, test lleno de NaN."""
import importlib.util, pickle, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("recipe", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.modules["recipe"] = mod
rng = np.random.default_rng(7)


def frame(n):
    return pd.DataFrame({
        "num": rng.normal(size=n), "num2": rng.exponential(size=n),
        "vacia": np.full(n, np.nan), "const": np.ones(n),
        "flag": rng.random(n) < 0.5,
        "lo": rng.choice(["a", "b", "c", None], n),
        "hi": [f"u{v}" for v in rng.integers(0, 2000, n)],
        "numstr": rng.integers(0, 50, n).astype(str),
    })


cases = []
X = frame(500); y = np.repeat(np.arange(8), [300, 80, 50, 30, 20, 12, 6, 2]); rng.shuffle(y)
cases.append(("8 clases, la menor con 2", X, y))
X = frame(600); y = np.zeros(600, int); y[rng.choice(600, 3, replace=False)] = 1
cases.append(("binario con 3 positivos", X, y))
X = frame(1500); y = (X["num"] + (X["lo"] == "a") + rng.normal(size=1500) > 0.5).astype(int)
cases.append(("binario normal 1500", X, y))
X = frame(3000); y = rng.choice(3, 3000, p=[0.97, 0.022, 0.008])
cases.append(("3 clases, rara con ~24 (justo bajo ES_MIN_CLASS)", X, y))
X = frame(25000); y = np.where(X["num"] > 1.5, 2, np.where(X["num"] > 0.3, 1, 0))
cases.append(("3 clases, 25k filas (ruta holdout)", X, y))

for name, X, y in cases:
    Xte = frame(200)
    Xte.loc[:50, "lo"] = "nunca_visto"; Xte.loc[:50, "hi"] = "nunca_visto"
    Xte["flag"] = Xte["flag"].astype(object); Xte.iloc[60:70] = np.nan
    t = time.perf_counter()
    m = mod.build_model().fit(X, y)
    P = m.predict_proba(Xte)
    P2 = pickle.loads(pickle.dumps(m)).predict_proba(Xte)
    ok = P.shape == (len(Xte), len(np.unique(y))) and np.isfinite(P).all() and np.allclose(P.sum(1), 1) and np.allclose(P, P2)
    print(f"{'OK ' if ok else 'MAL'} {name:50s} {time.perf_counter()-t:5.1f}s {getattr(m.steps[-1][1], 'info_', {})}")
