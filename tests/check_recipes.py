"""Pruebas de humo, de borde y de peor caso con datos SINTÉTICOS (no son evidencia de desempeño).

    python tests/check_recipes.py edge            # casos raros, todas las variantes
    python tests/check_recipes.py smoke           # 80/20 sintético mixto, todas las variantes
    python tests/check_recipes.py stress R1 R3    # peor caso: tiempo de fit y tamaño de pickle
    python tests/check_recipes.py repro           # dos ajustes iguales dan las mismas probabilidades

Verifica en cada caso: forma (n, k), probabilidades finitas que suman 1, orden de classes_,
y que el pickle recargado predice lo mismo. El tamaño de pickle debe ser < 5 MB.
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from variants import VARIANTS  # noqa: E402

ALL = list(VARIANTS)


def check(model, Xte, k):
    P = model.predict_proba(Xte)
    blob = pickle.dumps(model)
    P2 = pickle.loads(blob).predict_proba(Xte)
    ok = (P.shape == (len(Xte), k) and np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-6)
          and np.allclose(P, P2) and len(blob) < 5_000_000)
    return ok, P, len(blob)


def metric(y, P, classes):
    if len(classes) == 2:
        return "auc", roc_auc_score(y == classes[1], P[:, 1])
    return "logloss", log_loss(y, P, labels=classes)


def mixed(n, rng, k=2, p_num=6):
    X = pd.DataFrame(rng.normal(size=(n, p_num)), columns=[f"n{i}" for i in range(p_num)])
    X["lo"] = rng.choice(["a", "b", "c", None], n)
    X["hi"] = np.array([f"id{i}" for i in range(3000)])[rng.integers(0, 3000, n)]
    X["flag"] = rng.random(n) < 0.4
    X.loc[rng.random(n) < 0.1, "n1"] = np.nan
    eff = dict(zip([f"id{i}" for i in range(3000)], rng.normal(0, 1, 3000)))
    score = X["n0"] - 0.7 * X["n2"] ** 2 + (X["lo"] == "a") + X["hi"].map(eff) + rng.normal(0, 1, n)
    if k == 2:
        y = (score > np.quantile(score, 0.7)).astype(int)
    else:
        y = np.digitize(score, np.quantile(score, np.linspace(0, 1, k + 1)[1:-1]))
    return X, np.asarray(y)


def edge():
    rng = np.random.default_rng(7)
    cases = []
    X, _ = mixed(500, rng)
    y = np.repeat(np.arange(8), [300, 80, 50, 30, 20, 12, 6, 2]); rng.shuffle(y)
    X["vacia"] = np.nan; X["const"] = 1.0
    cases.append(("8 clases, la menor con 2", X, y))
    X, _ = mixed(600, rng); y = np.zeros(600, int); y[rng.choice(600, 3, replace=False)] = 1
    cases.append(("binario con 3 positivos", X, y))
    X, _ = mixed(1500, rng); y = np.where(rng.random(1500) < 0.5, "good", "bad")
    cases.append(("etiquetas texto", X, y))
    X, _ = mixed(3000, rng); y = rng.choice(3, 3000, p=[0.97, 0.022, 0.008])
    cases.append(("3 clases, rara ~24", X, y))
    for name, X, y in cases:
        Xte, _ = mixed(200, rng)
        for c in X.columns.difference(Xte.columns):
            Xte[c] = np.nan
        Xte = Xte[X.columns]
        Xte.loc[:50, "lo"] = "nunca_visto"; Xte.loc[:50, "hi"] = "nunca_visto"
        Xte["flag"] = Xte["flag"].astype(object); Xte.iloc[60:70] = np.nan
        for v in ALL:
            t = time.perf_counter()
            try:
                m = VARIANTS[v]().fit(X, y)
                ok, P, size = check(m, Xte, len(np.unique(y)))
                ok = ok and list(m.classes_) == list(np.unique(y))
                print(f"{'OK ' if ok else 'MAL'} {name:28s} {v:9s} {time.perf_counter()-t:5.1f}s {size/1e6:4.2f}MB", flush=True)
            except Exception as e:
                print(f"ERR {name:28s} {v:9s} {type(e).__name__}: {e}", flush=True)


def smoke():
    rng = np.random.default_rng(1)
    for k in (2, 3):
        X, y = mixed(4000, rng, k=k)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        for v in ALL:
            t = time.perf_counter()
            m = VARIANTS[v]().fit(Xtr, ytr)
            ok, P, size = check(m, Xte, k)
            name, val = metric(yte, P, m.classes_)
            last = getattr(m, "base_", None) or getattr(m, "models_", [None])[-1]
            extra = getattr(last, "net_", None)
            vos = f"vos_steps={extra.vos_steps_}" if extra is not None else ""
            print(f"{'OK ' if ok else 'MAL'} k={k} {v:9s} {name}={val:.4f} fit={time.perf_counter()-t:5.1f}s "
                  f"pkl={size/1e6:4.2f}MB {m.info_} {vos}", flush=True)


def stress(variants):
    rng = np.random.default_rng(3)
    # ~100k filas, 5 clases, mixto con alta cardinalidad
    n = 100_000
    big = pd.DataFrame(rng.normal(size=(n, 36)).astype("float32"), columns=[f"g{i}" for i in range(36)])
    big["store"] = np.array([f"s{i}" for i in range(5000)])[rng.integers(0, 5000, n)]
    big["region"] = rng.choice(list("ABCDEFG"), n)
    big["prod"] = np.array([f"p{i}" for i in range(800)])[rng.integers(0, 800, n)]
    big = big.mask(rng.random(big.shape) < 0.03)
    yb = np.digitize(big["g0"].fillna(0) + big["g1"].fillna(0) * big["g2"].fillna(0) + rng.normal(0, 1, n), [-1, 0, 1, 2])
    # ancho: 3,750 x 1,776, binario
    wide = pd.DataFrame(rng.normal(size=(3750, 1776)).astype("float32"), columns=[f"w{i}" for i in range(1776)])
    yw = (wide.iloc[:, :20].sum(axis=1) + rng.normal(0, 3, 3750) > 0).astype(int).to_numpy()
    # ancho multiclase con clase de ~1 %
    wide3 = pd.DataFrame(rng.normal(size=(3850, 1617)).astype("float32"), columns=[f"h{i}" for i in range(1617)])
    y3 = rng.choice(3, 3850, p=[0.94, 0.05, 0.01])
    # 150k x 10 binario desbalanceado
    gm = pd.DataFrame(rng.lognormal(size=(150_000, 10)).astype("float32"), columns=[f"c{i}" for i in range(10)])
    yg = (np.log(gm["c0"]) + rng.normal(0, 1, 150_000) > 1.6).astype(int).to_numpy()
    for name, X, y in [("100k x 39, 5 clases", big, yb), ("3,750 x 1,776 bin", wide, yw),
                       ("3,850 x 1,617, 3 clases", wide3, y3), ("150k x 10 bin", gm, yg)]:
        if os.environ.get("STRESS_ONLY") and os.environ["STRESS_ONLY"] not in name:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        for v in variants:
            t = time.perf_counter()
            m = VARIANTS[v]().fit(Xtr, ytr)
            tf = time.perf_counter() - t
            t = time.perf_counter(); ok, P, size = check(m, Xte, len(np.unique(y))); tp = time.perf_counter() - t
            mn, val = metric(yte, P, m.classes_)
            print(f"{'OK ' if ok else 'MAL'} {name:24s} {v:9s} fit={tf:6.1f}s pred+pickle={tp:5.1f}s pkl={size/1e6:4.2f}MB {mn}={val:.4f}", flush=True)


def repro():
    """Dos ajustes con los mismos datos deben dar las mismas probabilidades (semillas fijas en todo)."""
    rng = np.random.default_rng(11)
    X, y = mixed(3000, rng, k=3)
    Xte, _ = mixed(300, rng, k=3)
    for v in ALL:
        P1 = VARIANTS[v]().fit(X, y).predict_proba(Xte)
        P2 = VARIANTS[v]().fit(X, y).predict_proba(Xte)
        d = np.abs(P1 - P2).max()
        print(f"{'OK ' if d < 1e-9 else 'MAL'} {v:9s} diferencia máxima entre dos ajustes = {d:.1e}", flush=True)


if __name__ == "__main__":
    {"edge": lambda: edge(), "smoke": lambda: smoke(), "repro": lambda: repro(), "stress": lambda: stress(sys.argv[2:] or ["R1", "R3", "MLP-VOS", "LR-SMOTE"])}[sys.argv[1]]()
