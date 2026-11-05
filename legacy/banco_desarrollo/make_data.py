"""Construye el banco de prueba: datasets reales de PMLB + sintéticos para casos extremos.
Cada dataset se guarda como pickle con X_train, y_train, X_test, y_test ya divididos
(misma división para todas las variantes)."""
import os, pickle, sys
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

OUT = "/home/claude/data"
CACHE = "/home/claude/data/pmlb_cache"
os.makedirs(CACHE, exist_ok=True)
rng = np.random.default_rng(123)

# nombre PMLB -> (convertir_a_categoricas_str, fraccion_faltantes, max_filas)
REAL = {
    # binarios
    "adult": (True, 0.0, None),
    "churn": (False, 0.05, None),
    "spambase": (False, 0.0, None),
    "magic": (False, 0.0, None),
    "coil2000": (False, 0.0, None),          # desbalanceado (~6 %)
    "Hill_Valley_with_noise": (False, 0.0, None),  # relación casi lineal
    "madelon": (False, 0.0, None),           # 500 columnas, interacciones
    "GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1": (False, 0.0, None),  # 1000 columnas, epistasis
    "GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1": (False, 0.0, None),  # señal débil
    "kr_vs_kp": (True, 0.0, None),
    "phoneme": (False, 0.0, None),
    "spectf": (False, 0.1, None),            # pequeño
    "dis": (False, 0.0, None),               # muy desbalanceado
    # multiclase
    "waveform_40": (False, 0.0, None),       # 3 clases
    "connect_4": (True, 0.0, None),          # 3 clases, 67k filas
    "splice": (True, 0.0, None),             # 3 clases, 60 categóricas
    "allbp": (False, 0.0, None),             # 3 clases, clase rara
    "page_blocks": (False, 0.0, None),       # 5 clases, desbalance
    "nursery": (True, 0.0, None),
    "shuttle": (False, 0.0, None),           # 7 clases, raras, 58k
    "yeast": (False, 0.0, None),             # 10 -> top 8
    "car_evaluation": (True, 0.0, None),
    "dna": (True, 0.0, None),                # 180 columnas categóricas
}


def order_by_freq(y, kmax=8):
    vc = pd.Series(y).value_counts()
    keep = vc.index[:kmax]
    mask = np.isin(y, keep)
    mapping = {c: i for i, c in enumerate(keep)}
    return mask, np.array([mapping[v] for v in y[mask]], dtype=int)


def save(name, X, y, test_size=0.25, Xte=None, yte=None):
    if Xte is None:
        strat = y if np.bincount(y).min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=strat, random_state=42)
    else:
        Xtr, ytr = X, y
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    with open(f"{OUT}/{name}.pkl", "wb") as f:
        pickle.dump((Xtr, np.asarray(ytr), Xte, np.asarray(yte)), f)
    k = len(np.unique(y))
    print(f"{name:45s} n={len(Xtr)+len(Xte):6d} p={X.shape[1]:5d} k={k} minclass={np.bincount(np.asarray(y)).min()}")


def real():
    for name, (as_cat, miss, maxn) in REAL.items():
        try:
            df = fetch_data(name, local_cache_dir=CACHE)
        except Exception as e:
            print("FALLA", name, e); continue
        y = df.pop("target").to_numpy()
        mask, y = order_by_freq(y)
        X = df[mask].reset_index(drop=True)
        X.columns = [f"c{i}" for i in range(X.shape[1])]
        if as_cat:
            for c in X.columns:
                if X[c].nunique() <= 60:
                    X[c] = ("v" + X[c].astype(str)).astype(object)
        if miss > 0:
            for c in X.columns:
                m = rng.random(len(X)) < miss
                X.loc[m, c] = np.nan
        save(name, X, y)


def synth():
    # 1) alta cardinalidad + categorías desconocidas en test, binario
    n = 20000
    cats = np.array([f"id{i}" for i in range(6000)])
    eff = rng.normal(0, 1.2, len(cats))
    ci = rng.integers(0, len(cats), n)
    zip_ = np.array([f"z{i}" for i in range(400)]); ez = rng.normal(0, 0.8, 400); zi = rng.integers(0, 400, n)
    xnum = rng.normal(size=(n, 6))
    logit = eff[ci] + ez[zi] + xnum[:, 0] - 0.5 * xnum[:, 1] ** 2 - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(xnum, columns=[f"n{i}" for i in range(6)])
    X["cat_hi"] = cats[ci]; X["zip"] = zip_[zi]
    X["cat_lo"] = rng.choice(["a", "b", "c", "d"], n)
    X.loc[rng.random(n) < 0.1, "n2"] = np.nan
    # test: 30 % de filas con categorías que nunca salen en train
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    Xte = Xte.copy(); m = rng.random(len(Xte)) < 0.3
    Xte.loc[m, "cat_hi"] = [f"nuevo{i}" for i in range(m.sum())]
    save("syn_highcard_unseen", Xtr, ytr, Xte=Xte, yte=yte)

    # 2) multiclase 3 clases con clase muy rara (~0.8 %), 8000 filas, faltantes
    X, y = make_classification(n_samples=8000, n_features=30, n_informative=8, n_redundant=4,
                               n_classes=3, weights=[0.85, 0.142, 0.008], class_sep=0.8, random_state=1)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
    X = X.mask(rng.random(X.shape) < 0.08)
    save("syn_rare3", X, y)

    # 3) ancho: 3000 x 1800, binario, poca señal
    X, y = make_classification(n_samples=3000, n_features=1800, n_informative=25, n_redundant=10,
                               flip_y=0.05, random_state=2)
    save("syn_wide1800", pd.DataFrame(X, columns=[f"w{i}" for i in range(1800)]), y)

    # 4) grande: 100k filas, 40 columnas, 5 clases, mixtas con alta cardinalidad
    n = 100_000
    X, y = make_classification(n_samples=n, n_features=36, n_informative=12, n_redundant=6, n_classes=5,
                               weights=[0.4, 0.25, 0.18, 0.12, 0.05], class_sep=0.7, random_state=3)
    X = pd.DataFrame(X, columns=[f"g{i}" for i in range(36)])
    X["store"] = np.array([f"s{i}" for i in range(3000)])[rng.integers(0, 3000, n)]
    X["region"] = rng.choice(list("ABCDEFG"), n)
    X["prod"] = np.array([f"p{i}" for i in range(800)])[rng.integers(0, 800, n)]
    X["flag"] = rng.random(n) < 0.3
    X = X.mask(rng.random(X.shape) < 0.03)
    save("syn_big100k_5c", X, y)

    # 5) pequeño: 500 filas, 8 clases
    X, y = make_classification(n_samples=500, n_features=12, n_informative=8, n_classes=8,
                               weights=[.3, .2, .15, .1, .1, .07, .05, .03], random_state=4)
    save("syn_small8c", pd.DataFrame(X, columns=[f"s{i}" for i in range(12)]), y)


if __name__ == "__main__":
    real()
    if "--synth" in sys.argv: synth()
