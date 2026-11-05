import importlib.util, pickle, sys, time, warnings, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("recipe", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.modules["recipe"] = mod
def frame(d):
    out = {}
    for c, (dt, vals) in d.items():
        s = pd.Series(vals)
        out[c] = s.astype(float) if dt.startswith("float") else s.astype(object).where(s.notna(), np.nan) if dt in ("object", "str", "string") else s
    return pd.DataFrame(out)
for n in sys.argv[2:]:
    a, ytr, b, yte = pickle.load(open(f"/home/claude/data311/{n}.pkl", "rb"))
    Xtr, Xte, ytr, yte = frame(a), frame(b), np.array(ytr), np.array(yte)
    m = mod.build_model(); t = time.perf_counter(); m.fit(Xtr, ytr); tf = time.perf_counter() - t
    P = m.predict_proba(Xte); blob = pickle.dumps(m)
    auc = roc_auc_score(yte, P[:, 1]) if P.shape[1] == 2 else roc_auc_score(yte, P, multi_class="ovr")
    print(f"{n:22s} auc={auc:.4f} fit={tf:5.1f}s pkl={len(blob)/1e6:.2f}MB")
