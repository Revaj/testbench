"""Corre variantes sobre los 38 datasets de clasificación de TabArena-v0.1 con sus particiones oficiales.

    python bench/run.py --data data --variants R1,MLP-VOS,LR-SMOTE            # todas las particiones
    python bench/run.py --data data --variants R1 --max-splits 1              # solo la primera partición
    python bench/run.py --data data --variants R1 --datasets credit-g,splice

Por cada (variante, dataset, partición) guarda en results/raw/<variante>/<dataset>.json:
  ROC-AUC (binario: clase positiva = classes_[1]; multiclase: uno-contra-todos macro y por clase),
  log-loss, tiempo de fit y de predict_proba, tamaño del pickle y el error si lo hubo.
Se puede interrumpir y volver a correr: salta las particiones ya hechas.
Las variantes corren en un solo proceso, una tras otra, para que los tiempos sean comparables.
"""
import argparse
import json
import pickle
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from variants import VARIANTS  # noqa: E402

TIMEOUT_S = 900   # corte duro por ajuste (el límite del curso es 300 s en un servidor ~5x más lento)


def load(dd):
    schema = json.loads((dd / "schema.json").read_text())
    nominal = [c["name"] for c in schema["columns"] if c["type"] == "nominal"]
    df = pd.read_csv(dd / "data.csv.gz", dtype={c: str for c in nominal}, keep_default_na=False,
                     na_values=[""], low_memory=False)
    for c in df.columns:
        if c not in nominal:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    y = df.pop(schema["target"]).to_numpy()
    splits = pd.read_csv(dd / "splits.csv.gz")
    return df, y, splits, schema


def metrics(y, P, classes):
    out = {}
    if len(classes) == 2:
        out["auc"] = float(roc_auc_score(y == classes[1], P[:, 1]))
    else:
        per = [float(roc_auc_score(y == c, P[:, j])) if 0 < (y == c).sum() < len(y) else None
               for j, c in enumerate(classes)]
        out["auc_per_class"] = per
        out["auc"] = float(np.mean([a for a in per if a is not None]))
    out["logloss"] = float(log_loss(y, np.clip(P, 1e-15, 1), labels=classes))
    return out


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout(f"más de {TIMEOUT_S} s")


def run_split(variant, X, y, tr, te):
    rec = {}
    model = VARIANTS[variant]()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT_S)
    try:
        t0 = time.perf_counter()
        model.fit(X.iloc[tr].reset_index(drop=True), y[tr])
        rec["fit_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        P = model.predict_proba(X.iloc[te].reset_index(drop=True))
        rec["pred_s"] = time.perf_counter() - t0
    finally:
        signal.alarm(0)
    classes = np.asarray(model.classes_)
    if list(classes) != list(np.unique(y[tr])):
        raise ValueError(f"classes_ {list(classes)} no coincide con np.unique(y_train)")
    if P.shape != (len(te), len(classes)) or not np.isfinite(P).all() or not np.allclose(P.sum(1), 1, atol=1e-4):
        raise ValueError(f"predict_proba inválido: forma {P.shape}, suma en [{P.sum(1).min():.4f}, {P.sum(1).max():.4f}]")
    rec["pickle_mb"] = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6
    rec.update(metrics(y[te], P, classes))
    br = getattr(model, "branch_probas", None)
    if br is not None and len(getattr(model, "bases_", [])) > 1:
        rec["branch_auc"] = {k: metrics(y[te], v, classes)["auc"] for k, v in br(X.iloc[te].reset_index(drop=True)).items()}
    info = getattr(model, "info_", None)
    if info:
        rec["info"] = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, list))}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--variants", required=True)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--max-splits", type=int, default=0, help="0 = todas las particiones oficiales")
    ap.add_argument("--out", default=str(ROOT / "results" / "raw"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    data = Path(a.data)
    meta = pd.read_csv(ROOT / "bench" / "TabArena-v0.1_tasks_metadata.csv")
    meta = meta[meta.is_classification]
    order = meta.groupby("dataset_name").num_instances.first().sort_values().index.tolist()   # chicos primero
    names = order if a.datasets == "all" else [n for n in order if n in a.datasets.split(",")]
    for variant in a.variants.split(","):
        outdir = Path(a.out) / variant
        outdir.mkdir(parents=True, exist_ok=True)
        for name in names:
            X, y, splits, schema = load(data / name)
            keys = splits[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"]).to_numpy().tolist()
            if a.max_splits:
                keys = keys[: a.max_splits]
            f = outdir / f"{name}.json"
            res = json.loads(f.read_text()) if f.exists() else {"dataset": name, "variant": variant,
                                                                 "task_id": schema["task_id"], "splits": []}
            done = {(s["repeat"], s["fold"]) for s in res["splits"]}
            for r, k in keys:
                if (r, k) in done:
                    continue
                s = splits[(splits["repeat"] == r) & (splits["fold"] == k)]
                tr = s.rowid[s.type == "TRAIN"].to_numpy()
                te = s.rowid[s.type == "TEST"].to_numpy()
                rec = {"repeat": int(r), "fold": int(k)}
                try:
                    rec.update(run_split(variant, X, y, tr, te))
                    msg = f"auc={rec['auc']:.4f} fit={rec['fit_s']:6.1f}s pkl={rec['pickle_mb']:.2f}MB"
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                    rec["traceback"] = traceback.format_exc()[-2000:]
                    msg = "ERROR " + rec["error"]
                res["splits"].append(rec)
                f.write_text(json.dumps(res, indent=1))
                print(f"{variant:9s} {name:42s} ({r},{k}) {msg}", flush=True)


if __name__ == "__main__":
    main()
