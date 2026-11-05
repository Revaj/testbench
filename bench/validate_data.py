"""Valida una carpeta data/ descargada con fetch_data.py, sin confiar en su manifest.

    python bench/validate_data.py [carpeta_data]

Para cada uno de los 38 datasets de clasificación de TabArena-v0.1 comprueba, contra la metadata oficial:
  - que existan data.csv.gz, splits.csv.gz y schema.json, y que se puedan leer;
  - filas, número de features (sin el target) y número de clases;
  - particiones: las mismas (repeat, fold) que TabArena, tamaños train/test exactos, que train y test no
    se traslapen, que test cubra todas las filas una vez por repetición y que los índices estén en rango;
  - md5 de los archivos contra data/manifest.json (si existe).
Sale con código 0 solo si los 38 pasan.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "bench" / "TabArena-v0.1_tasks_metadata.csv"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name, d, meta, manifest):
    issues = []
    m = meta[meta.dataset_name == name]
    first = m.iloc[0]
    for f in ("data.csv.gz", "splits.csv.gz", "schema.json"):
        if not (d / f).exists():
            return [f"falta {f}"], None
    schema = json.loads((d / "schema.json").read_text())
    df = pd.read_csv(d / "data.csv.gz", low_memory=False)
    sp = pd.read_csv(d / "splits.csv.gz")
    target = schema["target"]
    if target not in df.columns:
        issues.append(f"no está la columna target '{target}'")
        return issues, None
    if len(df) != int(first.num_instances):
        issues.append(f"filas {len(df)} != {int(first.num_instances)}")
    if df.shape[1] - 1 != int(first.num_features):
        issues.append(f"features {df.shape[1] - 1} != {int(first.num_features)}")
    k = df[target].nunique(dropna=True)
    if k != int(first.num_classes):
        issues.append(f"clases {k} != {int(first.num_classes)}")
    if df[target].isna().any():
        issues.append(f"target con {int(df[target].isna().sum())} faltantes")
    # La metadata guarda el tamaño PROMEDIO de train/test (p. ej. 598.67 / 299.33 con 898 filas en 3 folds);
    # el tamaño real de cada fold debe quedar a menos de 1 fila de ese promedio.
    want = {(int(r["repeat"]), int(r["fold"])): (float(r.num_instances_train), float(r.num_instances_test)) for _, r in m.iterrows()}
    got = set(map(tuple, sp[["repeat", "fold"]].drop_duplicates().to_numpy().tolist()))
    if got != set(want):
        issues.append(f"particiones {len(got)} != {len(want)} esperadas")
    if sp.rowid.min() < 0 or sp.rowid.max() >= len(df):
        issues.append("rowid fuera de rango")
    for (r, f), (ntr, nte) in want.items():
        s = sp[(sp["repeat"] == r) & (sp["fold"] == f)]
        tr, te = s.rowid[s.type == "TRAIN"].to_numpy(), s.rowid[s.type == "TEST"].to_numpy()
        if abs(len(tr) - ntr) >= 1 or abs(len(te) - nte) >= 1:
            issues.append(f"({r},{f}) train/test {len(tr)}/{len(te)} lejos del promedio {ntr:.2f}/{nte:.2f}")
        if np.intersect1d(tr, te).size:
            issues.append(f"({r},{f}) train y test se traslapan")
    for r in sorted({r for r, _ in want}):
        te_all = sp[(sp["repeat"] == r) & (sp.type == "TEST")].rowid.to_numpy()
        if len(te_all) != len(df) or len(np.unique(te_all)) != len(df):
            issues.append(f"repetición {r}: test no cubre cada fila exactamente una vez")
    if manifest and name in manifest:
        e = manifest[name]
        for f, key in (("data.csv.gz", "md5_data"), ("splits.csv.gz", "md5_splits")):
            if key in e and md5(d / f) != e[key]:
                issues.append(f"md5 de {f} no coincide con el manifest")
    info = dict(rows=len(df), features=df.shape[1] - 1, classes=int(k), splits=len(got),
                minority=float(df[target].value_counts(normalize=True).min()),
                missing=float(df.drop(columns=[target]).isna().mean().mean()),
                cat_cols=int(sum(c["type"] == "nominal" for c in schema["columns"] if c["name"] != target)))
    return issues, info


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
    meta = pd.read_csv(META)
    meta = meta[meta.is_classification]
    names = sorted(meta.dataset_name.unique(), key=str.lower)
    man_path = data / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    ok, rows = 0, []
    for name in names:
        d = data / name
        if not d.exists():
            print(f"FALTA {name}")
            continue
        try:
            issues, info = check(name, d, meta, manifest)
        except Exception as e:
            issues, info = [f"{type(e).__name__}: {e}"], None
        ok += not issues
        tag = "OK " if not issues else "MAL"
        extra = (f"{info['rows']:7d} x {info['features']:5d}  k={info['classes']}  splits={info['splits']:2d}  "
                 f"minoritaria={info['minority']:.3f}  faltantes={info['missing']:.3f}  categóricas={info['cat_cols']}") if info else ""
        print(f"{tag} {name:42s} {extra} {'; '.join(issues)}")
        if info:
            rows.append(dict(dataset=name, **info))
    extras = sorted(p.name for p in data.iterdir() if p.is_dir() and p.name not in set(names))
    if extras:
        print("Carpetas que no son de TabArena-v0.1 clasificación:", ", ".join(extras))
    print(f"\n{ok}/{len(names)} datasets válidos")
    return 0 if ok == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
