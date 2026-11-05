"""Corre una receta (archivo .py con build_model) sobre todos los datasets del banco.
Uso: python3 run.py receta.py etiqueta [filtro]
Guarda results/<etiqueta>.json con AUC global, AUC por clase, AUC por rama, tiempos y tamaño de pickle."""
import importlib.util, json, os, pickle, sys, time, glob, warnings
import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
path, tag = sys.argv[1], sys.argv[2]
filt = sys.argv[3] if len(sys.argv) > 3 else ""
spec = importlib.util.spec_from_file_location("recipe", path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
sys.modules["recipe"] = mod  # para que pickle encuentre las clases


def auc(y, P):
    k = P.shape[1]
    if k == 2:
        return float(roc_auc_score(y, P[:, 1])), None
    per = [float(roc_auc_score((y == c).astype(int), P[:, c])) if 0 < (y == c).sum() < len(y) else None for c in range(k)]
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(k)))), per


os.makedirs("results", exist_ok=True)
out_file = f"results/{tag}.json"
res = json.load(open(out_file)) if os.path.exists(out_file) else {}
for f in sorted(glob.glob("/home/claude/data/*.pkl")):
    name = os.path.basename(f)[:-4]
    if filt and filt not in name:
        continue
    Xtr, ytr, Xte, yte = pickle.load(open(f, "rb"))
    model = mod.build_model()
    t0 = time.perf_counter(); model.fit(Xtr, ytr); tfit = time.perf_counter() - t0
    t0 = time.perf_counter(); P = model.predict_proba(Xte); tpred = time.perf_counter() - t0
    blob = pickle.dumps(model)
    P2 = pickle.loads(blob).predict_proba(Xte)
    assert P.shape == (len(Xte), len(np.unique(ytr))), P.shape
    assert np.allclose(P, P2) and np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-5)
    a, per = auc(yte, P)
    branches = {}
    fm = model.steps[-1][1]
    if hasattr(fm, "branch_probas"):
        for bname, Pb in fm.branch_probas(Xte).items():
            branches[bname] = auc(yte, Pb)[0]
    elif hasattr(fm, "hgb_"):
        Xt = fm.prep_.transform(Xte)[fm.keep_]
        branches["hgb"] = auc(yte, mod.aligned(fm.hgb_.predict_proba(Xt), fm.hgb_.classes_, fm.classes_))[0]
        branches["lr"] = auc(yte, mod.aligned(fm.lr_.predict_proba(Xt), fm.lr_.classes_, fm.classes_))[0]
    info = getattr(fm, "info_", {})
    res[name] = dict(auc=a, per_class=per, branches=branches, fit_s=tfit, pred_s=tpred,
                     pickle_mb=len(blob) / 1e6, n=len(Xtr), p=Xtr.shape[1], k=P.shape[1], info=info)
    print(f"{name:45s} auc={a:.4f} {' '.join(f'{k}={v:.4f}' for k, v in branches.items())} "
          f"fit={tfit:6.1f}s pred={tpred:5.2f}s pkl={len(blob)/1e6:5.2f}MB {info}", flush=True)
    json.dump(res, open(out_file, "w"), indent=1)
