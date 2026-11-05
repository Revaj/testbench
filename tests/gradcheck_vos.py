"""Compara los gradientes analíticos de VOSNet (CE + beta * pérdida VOS) contra diferencias finitas."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "recipe"))
import tabular as T

rng = np.random.default_rng(0)
K, d, B = 3, 7, 16
net = T.VOSNet(use_vos=True)
net._init_params(d, K, rng)
for k in net.p:  # en float64 para que la diferencia finita sea precisa
    net.p[k] = net.p[k].astype(np.float64)
net.p["u"] = rng.uniform(0.2, 1.0, K)
net.p["a"] = np.array([0.7]); net.p["c"] = np.array([0.3])
X = rng.normal(size=(B, d)); y = rng.integers(0, K, B)
V = rng.normal(size=(K, T.MLP_H2)) + 1.0
worst = 0.0
for with_v in (False, True):
    loss, g = net._batch_grads(X, y, V if with_v else None)
    for name, G in g.items():
        P = net.p[name]
        flat = P.reshape(-1)
        for i in rng.choice(flat.size, min(8, flat.size), replace=False):
            old = flat[i]; eps = 1e-6
            flat[i] = old + eps; lp, _ = net._batch_grads(X, y, V if with_v else None)
            flat[i] = old - eps; lm, _ = net._batch_grads(X, y, V if with_v else None)
            flat[i] = old
            num = (lp - lm) / (2 * eps); ana = G.reshape(-1)[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
    print(f"VOS={with_v}: params revisados {sorted(g)}")
print(f"peor error relativo: {worst:.2e}  ->  {'OK' if worst < 1e-4 else 'FALLA'}")
