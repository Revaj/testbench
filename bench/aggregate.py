"""Resume los resultados de bench/run.py: tablas por dataset y por receta, comparaciones pareadas, Elo y gráficas.

    python bench/aggregate.py --raw results/raw --out results

Escribe results/summary.md, results/por_dataset.csv, results/por_receta.csv y las gráficas PNG.
Las métricas de cada dataset son el promedio sobre sus particiones oficiales (30 o 9).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["R1", "MLP-VOS", "LR-SMOTE", "KNN-SMOTE", "ET-BAL", "R3", "R2", "R4", "R1-AVG", "MLP", "KNN", "ET", "B-LIN", "B-HGB", "B-SMALL", "R0", "RF"]
LABEL = {
    "R1": "Stacking (R1)", "R3": "Stacking + MLP-VOS (R3)", "R4": "Stacking + LR-SMOTE (R4)",
    "R1-AVG": "Promedio 3 bases", "R2": "Stacking + MLP (R2)", "MLP-VOS": "MLP-VOS", "MLP": "MLP sin VOS",
    "LR-SMOTE": "Logística + SMOTE", "B-LIN": "Solo logística", "B-HGB": "Solo HGB", "B-SMALL": "Solo ExtraTrees",
    "R0": "Receta original (R0)", "RF": "Random forest (ref.)",
    "KNN-SMOTE": "kNN + SMOTE", "KNN": "kNN sin SMOTE", "ET-BAL": "ExtraTrees balanceado", "ET": "ExtraTrees sin pesos",
}
SEED = 0


def load(raw):
    rows = []
    for f in sorted(Path(raw).glob("*/*.json")):
        r = json.loads(f.read_text())
        for s in r["splits"]:
            rows.append(dict(variant=r["variant"], dataset=r["dataset"], repeat=s["repeat"], fold=s["fold"],
                             auc=s.get("auc"), logloss=s.get("logloss"), fit_s=s.get("fit_s"),
                             pred_s=s.get("pred_s"), pickle_mb=s.get("pickle_mb"), error=s.get("error"),
                             auc_per_class=s.get("auc_per_class"), branch_auc=s.get("branch_auc")))
    return pd.DataFrame(rows)


def per_dataset(df, meta):
    g = df.groupby(["variant", "dataset"])
    out = g.agg(auc=("auc", "mean"), logloss=("logloss", "mean"), fit_s=("fit_s", "mean"), fit_max=("fit_s", "max"),
                pred_s=("pred_s", "mean"), pickle_mb=("pickle_mb", "mean"), pickle_max=("pickle_mb", "max"),
                splits=("fold", "size"), errors=("error", lambda e: int(e.notna().sum()))).reset_index()
    info = meta.groupby("dataset_name").agg(problem=("problem_type", "first"), n=("num_instances", "first"),
                                            p=("num_features", "first"), k=("num_classes", "first"))
    return out.join(info, on="dataset")


def info_n(pdd, d):
    return int(pdd[pdd.dataset == d].n.iloc[0])


def bt_elo(score, higher_better=True, anchor="RF", n_boot=200):
    """Elo tipo TabArena: Bradley-Terry sobre duelos por dataset (empate = 0.5), escala Elo, anclado a RF = 1000.
    IC 95 % por bootstrap sobre datasets."""
    variants = list(score.columns)

    def fit(S):
        V = len(variants)
        W = np.zeros((V, V))
        A = S.to_numpy()
        for i in range(V):
            for j in range(V):
                if i == j:
                    continue
                a, b = A[:, i], A[:, j]
                ok = ~(np.isnan(a) | np.isnan(b))
                d = (a[ok] - b[ok]) * (1 if higher_better else -1)
                W[i, j] = (d > 1e-12).sum() + 0.5 * (np.abs(d) <= 1e-12).sum()
        r = np.zeros(V)
        for _ in range(2000):  # iteración MM de Bradley-Terry con un prior débil (0.5 duelos de empate vs. todos)
            p = np.exp(r)
            num = W.sum(1) + 0.5
            den = np.array([sum((W[i, j] + W[j, i] + 1.0) / (p[i] + p[j]) for j in range(V) if j != i) for i in range(V)])
            new = np.log(num / den)
            new -= new.mean()
            if np.abs(new - r).max() < 1e-9:
                r = new
                break
            r = new
        elo = r * 400 / np.log(10)
        if anchor in variants:
            elo = elo - elo[variants.index(anchor)] + 1000
        return elo

    est = fit(score)
    rng = np.random.default_rng(SEED)
    boots = np.array([fit(score.iloc[rng.integers(0, len(score), len(score))]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"elo": est, "elo_lo": lo, "elo_hi": hi}, index=variants)


def paired(score, a, b, higher_better=True, n_boot=2000):
    if a not in score or b not in score:
        return None
    d = (score[a] - score[b]).dropna() * (1 if higher_better else -1)
    if d.empty:
        return None
    rng = np.random.default_rng(SEED)
    bm = np.array([d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return dict(a=a, b=b, n=len(d), mean=d.mean(), lo=lo, hi=hi, wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()))


def fmt_ci(r, digits=4):
    sig = "sí" if (r["lo"] > 0 or r["hi"] < 0) else "no"
    return (f"{r['mean']:+.{digits}f} [{r['lo']:+.{digits}f}, {r['hi']:+.{digits}f}] | {r['wins']}–{r['losses']} | {sig}")


INK, INK2, MUTED, GRID, BASE, SURF, BLUE, RED = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#2a78d6", "#e34948"


def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def bar_panels(plt, rec, variants, path, refs=(), suptitle=None):
    """Tres paneles (ROC-AUC, fit, pickle) por receta; las de `refs` van en gris como referencia."""
    ink, ink2, muted, grid, base, surf, blue = INK, INK2, MUTED, GRID, BASE, SURF, BLUE
    labels = [LABEL.get(v, v) for v in variants]
    panels = [("auc", "ROC-AUC medio (38 datasets)", "{:.4f}", None),
              ("fit_s", "Fit medio por partición (s)", "{:.1f} s", None),
              ("pickle_mb", "Pickle medio (MB)", "{:.2f} MB", 5.0)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(variants) + 1.6), sharey=True)
    for ax, (col, title, f, limit) in zip(axes, panels):
        vals = rec.loc[variants, col].to_numpy()
        partial = (rec.loc[variants, "datasets"] < rec["datasets"].max()).to_numpy()
        y = np.arange(len(variants))[::-1]
        lo, hi = 0.0, None
        if col == "auc":
            lo = max(0.0, np.floor((vals.min() - 0.02) * 50) / 50)
            hi = min(1.0, vals.max() + 0.035)
            title = f"{title} · eje desde {lo:.2f}"
        elif limit:
            hi = limit * 1.3   # el RF de referencia pasa por mucho: se recorta y se anota
        else:
            hi = vals.max() * 1.3
        shown = np.minimum(vals, hi * 0.97)
        ref = np.array([v in refs for v in variants])
        ax.barh(y, shown, height=0.6, color=np.where(ref, base, np.where(partial, "#a9c8f0", blue)), edgecolor=surf, linewidth=2)
        ax.set_xlim(lo, hi)
        for v, yi, val, sv, pa in zip(variants, y, vals, shown, partial):
            txt = "  " + f.format(val) + (" →" if sv < val else "") + (f"  ({int(rec.loc[v, 'datasets'])} de {int(rec['datasets'].max())})" if pa and col == "auc" else "")
            ax.text(sv, yi, txt, va="center", ha="left", color=ink2, fontsize=9)
        if limit:
            ax.axvline(limit, color=muted, linewidth=1, linestyle=(0, (3, 3)))
            ax.text(limit, len(variants) - 0.4, " límite 5 MB", color=muted, fontsize=8, va="bottom")
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(np.arange(len(variants))[::-1])
        ax.set_yticklabels(labels)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", color=ink, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def charts(rec, pdd, variants, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, surf = INK, INK2, SURF
    _style(plt)
    labels = [LABEL.get(v, v) for v in variants]
    bar_panels(plt, rec, variants, out / "resumen_recetas.png")

    # AUC por dataset. Color = cuánto le falta a la mejor receta de ese dataset: claro = la mejor, rojo = lejos.
    # La mejor de cada dataset (o las empatadas a 3 decimales) va en negritas y con recuadro.
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    M = pdd.pivot(index="dataset", columns="variant", values="auc")[variants]
    order = pdd.drop_duplicates("dataset").set_index("dataset").loc[M.index].sort_values("n").index
    M = M.loc[order]
    G = M.sub(M.max(axis=1), axis=0)
    R = M.round(3).to_numpy()
    best = R == np.nanmax(R, axis=1, keepdims=True)    # igual que los números que se muestran
    worst = 0.10
    cmap = LinearSegmentedColormap.from_list("falta", [surf, "#fbd9d3", "#f08a7e", RED, "#8f1d1c"])
    cmap.set_bad(GRID)
    fig, ax = plt.subplots(figsize=(1.05 * len(variants) + 4, 0.3 * len(M) + 2.2))
    im = ax.imshow(-G.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=worst)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "error", ha="center", va="center", fontsize=7, color=ink2, style="italic")
                continue
            if best[i, j]:
                ax.add_patch(Rectangle((j - 0.47, i - 0.42), 0.94, 0.84, fill=False, edgecolor=ink, linewidth=1.2))
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if -G.iat[i, j] > 0.055 else ink, fontweight="bold" if best[i, j] else "normal")
    wins = np.nan_to_num(best, nan=False).sum(axis=0)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([f"{lab}\nmejor o empatada en {int(w)}" for lab, w in zip(labels, wins)], rotation=30, ha="right", color=ink2)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{d} (n={int(pdd[pdd.dataset == d].n.iloc[0]):,})" for d in M.index], fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[0, 0.025, 0.05, 0.075, worst])
    cb.ax.set_yticklabels(["0 · la mejor", "0.025", "0.050", "0.075", f"≥ {worst:.2f}"])
    cb.set_label("ROC-AUC que le falta para igualar a la mejor receta del dataset", color=ink2)
    cb.outline.set_visible(False)
    ax.set_title("ROC-AUC por dataset. Negritas con recuadro = la mejor receta de ese dataset (o empatada a 3 decimales).\n"
                 "Entre más rojo, más lejos de la mejor; blanco = la mejor o casi empatada. Gris = la receta falló.",
                 color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "auc_por_dataset.png", dpi=160)
    plt.close(fig)


NEW = ["KNN-SMOTE", "KNN", "ET-BAL", "ET"]          # recetas nuevas: van en gráficas aparte
NEW_REFS = ["R1", "LR-SMOTE", "RF"]                 # referencias en gris dentro de esas gráficas
NEW_PAIRS = [("KNN-SMOTE", "KNN", "¿SMOTE ayuda a kNN?"), ("ET-BAL", "ET", "¿Pesos balanceados ayudan a ExtraTrees?")]


def charts_new(rec, auc, pdd, out):
    """nuevas_recetas.png (promedios contra referencias) y efecto_smote_y_pesos.png (diferencia pareada por dataset)."""
    present = [v for v in NEW if v in rec.index]
    if not present:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    made = ["nuevas_recetas.png"]
    bar_panels(plt, rec, present + [v for v in NEW_REFS if v in rec.index], out / "nuevas_recetas.png",
               refs=NEW_REFS, suptitle="kNN y ExtraTrees (azul) frente a recetas anteriores (gris)")
    pairs = [(a, b, t) for a, b, t in NEW_PAIRS if a in auc and b in auc and (auc[a] - auc[b]).notna().any()]
    if not pairs:
        return made
    n_of = pdd.drop_duplicates("dataset").set_index("dataset").n
    order = n_of.loc[auc.index].sort_values().index
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs) + 2.6, 0.27 * len(order) + 2.2), sharey=True,
                             squeeze=False)
    for ax, (a, b, title) in zip(axes[0], pairs):
        d = (auc[a] - auc[b]).loc[order]
        y = np.arange(len(order))[::-1]
        ok = d.notna().to_numpy()
        vals = d.fillna(0).to_numpy()
        lim = max(0.01, float(np.nanmax(np.abs(d)))) * 1.15 if ok.any() else 0.01
        ax.barh(y[ok], vals[ok], height=0.62, color=np.where(vals[ok] >= 0, BLUE, RED), edgecolor=SURF, linewidth=1)
        for yi in y[~ok]:
            ax.text(0, yi, "sin resultado", ha="center", va="center", fontsize=7, color=MUTED, zorder=3,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=SURF, edgecolor="none"))
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_xlim(-lim, lim)
        r = paired(auc, a, b)
        sub = (f"media {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] · gana–pierde {r['wins']}–{r['losses']} · "
               f"{r['n']} datasets") if r else ""
        ax.set_title(f"{title}\n{LABEL.get(a, a)} − {LABEL.get(b, b)} (ROC-AUC)\n{sub}", color=INK, fontsize=10, loc="left")
        ax.text(lim * 0.97, len(order) - 0.3, "mejor con " + LABEL.get(a, a) + " →", ha="right", va="bottom", fontsize=8, color=BLUE)
        ax.text(-lim * 0.97, len(order) - 0.3, "← mejor sin", ha="left", va="bottom", fontsize=8, color=RED)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(order) + 0.6)
    axes[0][0].set_yticks(np.arange(len(order))[::-1])
    axes[0][0].set_yticklabels([f"{x} (n={int(n_of[x]):,})" for x in order], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "efecto_smote_y_pesos.png", dpi=160)
    plt.close(fig)
    return made + ["efecto_smote_y_pesos.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--title", default="Resultados")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "bench" / "TabArena-v0.1_tasks_metadata.csv")
    meta = meta[meta.is_classification]
    df = load(a.raw)
    pdd = per_dataset(df, meta)
    variants = [v for v in ORDER if v in set(pdd.variant)] + sorted(set(pdd.variant) - set(ORDER))
    pdd.to_csv(out / "por_dataset.csv", index=False)

    auc = pdd.pivot(index="dataset", columns="variant", values="auc")[variants]
    ll = pdd.pivot(index="dataset", columns="variant", values="logloss")[variants]
    problem = pdd.drop_duplicates("dataset").set_index("dataset").problem
    binary = problem[problem == "binary"].index
    multi = problem[problem != "binary"].index
    ranks = auc.rank(axis=1, ascending=False)
    # métrica "como TabArena": AUC en binarios (mayor mejor) y log-loss en multiclase (menor mejor) -> error normalizado
    ta = pd.concat([1 - auc.loc[auc.index.intersection(binary)], ll.loc[ll.index.intersection(multi)]])
    rec = pd.DataFrame({
        "datasets": auc.notna().sum(),
        "auc": auc.mean(), "auc_bin": auc.loc[auc.index.intersection(binary)].mean(),
        "auc_multi": auc.loc[auc.index.intersection(multi)].mean(),
        "logloss_multi": ll.loc[ll.index.intersection(multi)].mean(),
        "rank_auc": ranks.mean(), "wins_auc": (ranks == 1).sum(),
        "fit_s": pdd.groupby("variant").fit_s.mean(), "fit_max": pdd.groupby("variant").fit_max.max(),
        "pickle_mb": pdd.groupby("variant").pickle_mb.mean(), "pickle_max": pdd.groupby("variant").pickle_max.max(),
        "errors": pdd.groupby("variant").errors.sum(),
    }).loc[variants]
    # Un dataset con error cuenta como derrota en el Elo (como en TabArena, una falla no se descarta):
    # se le asigna un puntaje peor que cualquier otro.
    full = auc.fillna(-1.0)
    ta_full = ta.loc[auc.index.intersection(ta.index)][variants].fillna(1e9)
    if len(variants) > 1 and len(full) >= 5:
        e1 = bt_elo(full, True)
        e2 = bt_elo(ta_full, False)
        rec["elo_auc"], rec["elo_auc_lo"], rec["elo_auc_hi"] = e1.elo, e1.elo_lo, e1.elo_hi
        rec["elo_tabarena"], rec["elo_ta_lo"], rec["elo_ta_hi"] = e2.elo, e2.elo_lo, e2.elo_hi
    common = auc.dropna().index
    rec["auc_comunes"] = auc.loc[common].mean()
    rec.to_csv(out / "por_receta.csv")

    L = []
    L.append(f"# {a.title}\n")
    n_splits = int(df.groupby(["variant", "dataset"]).size().max()) if len(df) else 0
    L.append(f"Datasets: {auc.shape[0]} ({len(binary)} binarios, {len(multi)} multiclase). Métricas por dataset = promedio "
             f"sobre sus particiones oficiales evaluadas (hasta {n_splits}). ROC-AUC binario sobre la clase `classes_[1]`; "
             "multiclase = promedio uno-contra-todos de las clases presentes en test. Tiempos medidos en un contenedor "
             "de 2 CPU, una variante a la vez.\n")
    L.append("## Promedio por receta\n")
    L.append("\"Datasets\" = datasets con resultado; si una receta falla en alguno, sus promedios son sobre los que sí "
             "terminó (ver Errores), así que no son directamente comparables con los de 38.\n")
    L.append(f"La columna \"AUC {len(common)} comunes\" promedia solo los datasets donde todas las recetas terminaron, "
             "para comparar en igualdad de condiciones.\n")
    L.append(f"| Receta | Datasets | ROC-AUC medio | AUC {len(common)} comunes | AUC binarios | AUC multiclase | Log-loss multiclase | Rango medio (AUC) | "
             "Gana | Fit medio | Fit máx. | Pickle medio | Pickle máx. | Errores |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v, r in rec.iterrows():
        L.append(f"| {LABEL.get(v, v)} | {int(r.datasets)} | **{r.auc:.4f}** | {r.auc_comunes:.4f} | {r.auc_bin:.4f} | {r.auc_multi:.4f} | "
                 f"{r.logloss_multi:.4f} | {r.rank_auc:.2f} | {int(r.wins_auc)} | {r.fit_s:.1f} s | {r.fit_max:.1f} s | "
                 f"{r.pickle_mb:.2f} MB | {r.pickle_max:.2f} MB | {int(r.errors)} |")
    if "elo_auc" in rec:
        L.append("\n## Elo (Bradley-Terry por dataset, RF = 1000, IC 95 % por bootstrap)\n")
        L.append("Dos versiones: con ROC-AUC en los 38 datasets (como el curso) y \"como TabArena\" (ROC-AUC en binarios, "
                 "log-loss en multiclase). Réplica propia del método, no el código oficial. Un dataset con error cuenta "
                 "como derrota.\n")
        L.append("| Receta | Elo (ROC-AUC) | Elo (como TabArena) |")
        L.append("|---|---|---|")
        for v, r in rec.sort_values("elo_auc", ascending=False).iterrows():
            L.append(f"| {LABEL.get(v, v)} | {r.elo_auc:.0f} [{r.elo_auc_lo:.0f}, {r.elo_auc_hi:.0f}] | "
                     f"{r.elo_tabarena:.0f} [{r.elo_ta_lo:.0f}, {r.elo_ta_hi:.0f}] |")
    comps = [("R1", "R0"), ("R1", "RF"), ("R1", "R1-AVG"), ("R2", "R1"), ("R3", "R1"), ("R3", "R2"), ("MLP-VOS", "MLP"),
             ("R4", "R1"), ("LR-SMOTE", "B-LIN"), ("MLP-VOS", "RF"), ("LR-SMOTE", "RF"),
             ("KNN-SMOTE", "KNN"), ("ET-BAL", "ET"), ("KNN-SMOTE", "LR-SMOTE"), ("ET-BAL", "RF"), ("ET-BAL", "R1")]
    rows = [r for r in (paired(auc, x, y) for x, y in comps) if r]
    if rows:
        L.append("\n## Comparaciones pareadas (ROC-AUC, diferencia media por dataset, IC 95 % bootstrap)\n")
        L.append("| A vs. B | Datasets | A − B [IC 95 %] | Gana–pierde | ¿IC excluye 0? |")
        L.append("|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {LABEL.get(r['a'], r['a'])} vs. {LABEL.get(r['b'], r['b'])} | {r['n']} | {fmt_ci(r)} |")
    pc = df[df.auc_per_class.notna()]
    if len(pc):
        L.append("\n## Multiclase: ROC-AUC por clase (uno contra todos, promedio sobre particiones)\n")
        L.append("Clases en el orden de `classes_` (orden de las etiquetas).\n")
        main = [v for v in ["R1", "R3", "MLP-VOS", "LR-SMOTE", "R0", "RF"] if v in variants]
        L.append("| Dataset | " + " | ".join(LABEL.get(v, v) for v in main) + " |")
        L.append("|---|" + "---|" * len(main))
        for d in sorted(pc.dataset.unique(), key=lambda x: info_n(pdd, x)):
            cells = []
            for v in main:
                g = pc[(pc.variant == v) & (pc.dataset == d)].auc_per_class
                if g.empty:
                    cells.append("error")
                    continue
                M = np.array([[np.nan if a is None else a for a in row] for row in g], dtype=float)
                cells.append(" / ".join("—" if np.isnan(a) else f"{a:.3f}" for a in np.nanmean(M, axis=0)))
            L.append(f"| {d} | " + " | ".join(cells) + " |")
    L.append("\n## Por dataset: ROC-AUC · fit medio (s) · pickle medio (MB)\n")
    L.append("| Dataset | Tipo | Filas | Feat. | " + " | ".join(LABEL.get(v, v) for v in variants) + " |")
    L.append("|---|---|---|---|" + "---|" * len(variants))
    info = pdd.drop_duplicates("dataset").set_index("dataset")
    for d in info.sort_values("n").index:
        cells = []
        for v in variants:
            r = pdd[(pdd.variant == v) & (pdd.dataset == d)]
            if r.empty:
                cells.append("—")
                continue
            r = r.iloc[0]
            best = auc.loc[d].max()
            a_ = "error" if pd.isna(r.auc) else (f"**{r.auc:.4f}**" if abs(r.auc - best) < 1e-12 else f"{r.auc:.4f}")
            err = f" ({int(r.errors)} err.)" if r.errors else ""
            cells.append(f"{a_} · {r.fit_s:.1f} · {r.pickle_mb:.2f}{err}")
        L.append(f"| {d} | {info.loc[d].problem} | {int(info.loc[d].n):,} | {int(info.loc[d].p)} | " + " | ".join(cells) + " |")
    errs = df[df.error.notna()]
    if len(errs):
        L.append("\n## Errores\n")
        for (v, d), g in errs.groupby(["variant", "dataset"]):
            L.append(f"- {LABEL.get(v, v)} en {d}: {len(g)} partición(es) — `{g.error.iloc[0][:200]}`")
    L.append("\n![Promedios por receta](resumen_recetas.png)\n\n![ROC-AUC por dataset](auc_por_dataset.png)\n")
    main_v = [v for v in variants if v not in NEW]
    charts(rec, pdd[pdd.variant.isin(main_v)], main_v, out)
    for png in charts_new(rec, auc, pdd, out):
        L.append(f"![Recetas nuevas]({png})\n")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:len(variants) + 6]))


if __name__ == "__main__":
    main()
