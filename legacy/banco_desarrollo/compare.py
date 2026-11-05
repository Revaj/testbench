import json, sys, numpy as np
tags = sys.argv[1:] or ["A", "B", "C", "BD", "CD"]
R = {t: json.load(open(f"results/{t}.json")) for t in tags}
names = sorted(set.intersection(*[set(r) for r in R.values()]))
print(f"{'dataset':42s} " + " ".join(f"{t:>7s}" for t in tags) + "   " + " ".join(f"fit_{t:>4s}" for t in tags))
for n in names:
    print(f"{n[:42]:42s} " + " ".join(f"{R[t][n]['auc']:7.4f}" for t in tags) + "   "
          + " ".join(f"{R[t][n]['fit_s']:7.1f}" for t in tags))
print("-" * 100)
for t in tags:
    a = np.array([R[t][n]["auc"] for n in names])
    base = np.array([R[tags[0]][n]["auc"] for n in names])
    d = a - base
    print(f"{t:4s} media={a.mean():.4f} mediana={np.median(a):.4f} peor={a.min():.4f} ({names[a.argmin()][:20]})  "
          f"delta_media={d.mean():+.4f} gana={int((d > 1e-4).sum())} pierde={int((d < -1e-4).sum())} peor_delta={d.min():+.4f} ({names[d.argmin()][:20]})  "
          f"fit_max={max(R[t][n]['fit_s'] for n in names):.1f}s fit_total={sum(R[t][n]['fit_s'] for n in names):.0f}s "
          f"pkl_max={max(R[t][n]['pickle_mb'] for n in names):.2f}MB")
