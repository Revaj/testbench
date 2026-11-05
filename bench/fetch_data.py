"""Descarga los 38 datasets de clasificación de TabArena-v0.1 y sus particiones oficiales desde OpenML.

Uso (desde la raíz del repo clonado, en una máquina con acceso a openml.org):
    python bench/fetch_data.py                  # todos; si se corta, vuelve a correrlo: salta los ya verificados
    python bench/fetch_data.py credit-g splice  # solo algunos (por dataset_name)
    python bench/fetch_data.py --force          # repetir aunque ya estén descargados
    python bench/fetch_data.py --jobs=8         # descargas en paralelo (por defecto 4)

Requisitos: Python 3.11, numpy y pandas. Sin el paquete `openml` ni `tabarena`:
usa solo la API REST pública de OpenML (JSON + ARFF) con urllib.

Salida (carpeta data/, ignorada por git):
    data/<dataset_name>/data.csv.gz    todas las columnas (features + target), faltantes vacíos
    data/<dataset_name>/schema.json    orden y tipo de columnas, categorías, target, task/dataset id
    data/<dataset_name>/splits.csv.gz  particiones oficiales: type (TRAIN/TEST), rowid, repeat, fold
    data/manifest.json                 md5 y verificación contra la metadata oficial

Verifica contra bench/TabArena-v0.1_tasks_metadata.csv (copia sin modificar del repo oficial
autogluon/tabarena, Apache-2.0): filas, columnas, número de clases y, por partición,
tamaños de train/test. Si algo no coincide lo marca en el manifest y en la salida.
"""
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "bench" / "TabArena-v0.1_tasks_metadata.csv"
OUT = ROOT / "data"
API = "https://www.openml.org/api/v1/json"
UA = {"User-Agent": "testbench-fetch/0.1"}


def get(url, binary=False, tries=8):
    """Descarga con reintentos y espera creciente (3, 6, 12, ... hasta 60 s).
    Aguanta cortes de red (connection reset, DNS caído un momento)."""
    for i in range(tries):
        try:
            buf = io.BytesIO()
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                shown = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    buf.write(chunk)
                    if buf.tell() - shown >= 10 << 20:  # avisa cada 10 MB para que se vea que avanza
                        shown = buf.tell()
                        print(f"   ... {shown / 1e6:.0f} MB", flush=True)
            raw = buf.getvalue()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return raw if binary else raw.decode("utf-8")
        except Exception as e:
            if i == tries - 1:
                raise RuntimeError(f"Fallo al descargar {url}: {e}") from e
            wait = min(60, 3 * 2 ** i)
            print(f"   reintento {i + 1}/{tries - 1} en {wait}s ({type(e).__name__}: {e})", flush=True)
            time.sleep(wait)


def get_json(url):
    return json.loads(get(url))


# ---------------------------------------------------------------- ARFF
def _unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1].replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
    return v


def _split_values(text):
    """Separa por comas respetando comillas simples o dobles."""
    return next(csv.reader([text], quotechar="'", escapechar="\\", skipinitialspace=True))


def _parse_header(header):
    attrs = []
    for line in io.StringIO(header):
        s = line.strip()
        if not s.lower().startswith("@attribute"):
            continue
        rest = s[len("@attribute"):].strip()
        if rest[0] in "'\"":
            q = rest[0]
            end = rest.index(q, 1)
            while rest[end - 1] == "\\":
                end = rest.index(q, end + 1)
            name, typ = _unquote(rest[: end + 1]), rest[end + 1:].strip()
        else:
            name, typ = rest.split(None, 1)
        if typ.startswith("{"):
            attrs.append((name, "nominal", [_unquote(c) for c in _split_values(typ.strip()[1:-1])]))
        else:
            attrs.append((name, typ.split()[0].lower(), None))
    return attrs


def parse_arff(text):
    m = re.search(r"^[ \t]*@data[ \t]*$", text, flags=re.IGNORECASE | re.MULTILINE)
    attrs = _parse_header(text[: m.start()])
    names = [a[0] for a in attrs]
    body = re.sub(r"^[ \t]*%.*\n?", "", text[m.end():], flags=re.MULTILINE)
    first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    if first.startswith("{"):  # formato disperso (raro): lector lento en Python
        rows = []
        for ln in body.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            row = ["0"] * len(names)
            for item in _split_values(ln[1:-1].strip()) if ln[1:-1].strip() else []:
                idx, val = item.strip().split(None, 1)
                row[int(idx)] = val
            rows.append(row)
        df = pd.DataFrame(rows, columns=names, dtype=object).replace("?", None)
    else:  # formato denso: lector en C de pandas (rápido)
        df = pd.read_csv(io.StringIO(body), header=None, names=names, quotechar="'", escapechar="\\",
                         skipinitialspace=True, dtype=str, na_values=["?"], keep_default_na=False,
                         skip_blank_lines=True)
    for name, typ, _ in attrs:
        if typ in ("numeric", "real", "integer"):
            df[name] = pd.to_numeric(df[name], errors="coerce")
        else:
            col = df[name].astype(object)
            mask = col.notna()
            col[mask] = col[mask].astype(str).str.strip().str.replace(r'^"(.*)"$', r"\1", regex=True)
            df[name] = col
    return df, attrs


# ---------------------------------------------------------------- descarga de una tarea
def fetch_task(task_id):
    task = get_json(f"{API}/task/{task_id}")["task"]
    inputs = {i["name"]: i for i in task["input"]}
    did = int(inputs["source_data"]["data_set"]["data_set_id"])
    target = inputs["source_data"]["data_set"]["target_feature"]
    splits_url = inputs["estimation_procedure"]["estimation_procedure"]["data_splits_url"]

    desc = get_json(f"{API}/data/{did}")["data_set_description"]
    feats = get_json(f"{API}/data/features/{did}")["data_features"]["feature"]
    drop = {f["name"] for f in feats if str(f.get("is_ignore", "false")) == "true"
            or str(f.get("is_row_identifier", "false")) == "true"}
    drop |= set(filter(None, str(desc.get("row_id_attribute") or "").split(",")))
    drop |= set(filter(None, str(desc.get("ignore_attribute") or "").split(",")))
    drop.discard(target)

    print("   bajando datos...", flush=True)
    raw = get(desc["url"])
    print(f"   leyendo ARFF ({len(raw) / 1e6:.1f} MB)...", flush=True)
    df, attrs = parse_arff(raw)
    split_df, _ = parse_arff(get(splits_url))
    split_df.columns = [c.lower() for c in split_df.columns]
    for c in ("rowid", "repeat", "fold"):
        split_df[c] = split_df[c].astype(int)
    split_df["type"] = split_df["type"].astype(str).str.upper()

    df = df[[c for c in df.columns if c not in drop]]
    schema = {
        "task_id": task_id, "dataset_id": did, "dataset_version": desc.get("version"),
        "target": target, "dropped": sorted(drop),
        "columns": [{"name": n, "type": "nominal" if t == "nominal" else "numeric", "categories": cats}
                    for n, t, cats in attrs if n in df.columns],
        "source_url": desc["url"], "splits_url": splits_url,
    }
    return df, split_df, schema


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(name, df, split_df, schema, meta):
    m = meta[meta.dataset_name == name]
    first = m.iloc[0]
    issues = []
    if len(df) != int(first.num_instances):
        issues.append(f"filas {len(df)} != {int(first.num_instances)}")
    n_feat = df.shape[1] - 1
    if n_feat != int(first.num_features):
        issues.append(f"features {n_feat} != {int(first.num_features)}")
    k = df[schema["target"]].nunique(dropna=True)
    if k != int(first.num_classes):
        issues.append(f"clases {k} != {int(first.num_classes)}")
    got = split_df.groupby(["repeat", "fold"]).type.value_counts().unstack(fill_value=0)
    if len(got) != len(m):
        issues.append(f"particiones {len(got)} != {len(m)}")
    for _, r in m.iterrows():
        key = (int(r["repeat"]), int(r["fold"]))
        if key not in got.index:
            issues.append(f"falta partición {key}")
            continue
        tr, te = int(got.loc[key].get("TRAIN", 0)), int(got.loc[key].get("TEST", 0))
        # la metadata guarda el tamaño PROMEDIO por fold (p. ej. 598.67): el real debe quedar a < 1 fila
        mtr, mte = float(r.num_instances_train), float(r.num_instances_test)
        if abs(tr - mtr) >= 1 or abs(te - mte) >= 1:
            issues.append(f"partición {key}: train/test {tr}/{te} lejos del promedio {mtr:.2f}/{mte:.2f}")
    return issues


def _one(name, meta, manifest, lock, man_path):
    rows = meta[meta.dataset_name == name]
    if rows.empty:
        print(f"?? {name}: no está en la metadata de clasificación"); return
    tid = int(rows.iloc[0].task_id_str)
    prev = manifest.get(name, {})
    if prev.get("verified") and (OUT / name / "data.csv.gz").exists() and "--force" not in sys.argv:
        print(f"OK  {name:45s} ya descargado y verificado (usa --force para repetir)"); return
    t0 = time.time()
    print(f"--> {name} (task {tid}, {int(rows.iloc[0].num_instances)} filas x {int(rows.iloc[0].num_features)} columnas)", flush=True)
    try:
        df, split_df, schema = fetch_task(tid)
    except Exception as e:
        print(f"ERROR {name} (task {tid}): {e}", flush=True)
        with lock:
            manifest[name] = {"task_id": tid, "error": str(e)}
            man_path.write_text(json.dumps(manifest, indent=1))
        return
    # OpenML guarda 10 repeticiones para todos; TabArena-v0.1 usa solo las primeras 3 en datasets con >= 2,500 filas
    n_rep = int(rows["repeat"].max()) + 1
    schema["openml_repeats"] = int(split_df["repeat"].max()) + 1
    schema["tabarena_repeats"] = n_rep
    split_df = split_df[split_df["repeat"] < n_rep].reset_index(drop=True)
    d = OUT / name
    d.mkdir(exist_ok=True)
    df.to_csv(d / "data.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    split_df.to_csv(d / "splits.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (d / "schema.json").write_text(json.dumps(schema, indent=1, ensure_ascii=False))
    issues = verify(name, df, split_df, schema, meta)
    entry = {
        "task_id": tid, "dataset_id": schema["dataset_id"], "rows": int(len(df)), "features": int(df.shape[1] - 1),
        "classes": int(df[schema["target"]].nunique()), "splits": int(split_df.groupby(["repeat", "fold"]).ngroups),
        "md5_data": md5(d / "data.csv.gz"), "md5_splits": md5(d / "splits.csv.gz"),
        "verified": not issues, "issues": issues,
    }
    with lock:
        manifest[name] = entry
        man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    tag = "OK " if not issues else "DIF"
    print(f"{tag} {name:45s} task={tid} {len(df):7d}x{df.shape[1]-1:5d} "
          f"k={entry['classes']} splits={entry['splits']} {time.time()-t0:5.1f}s {'; '.join(issues)}", flush=True)


def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    meta = pd.read_csv(META)
    meta = meta[meta.is_classification]
    jobs = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--jobs=")), 4)
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(meta.groupby("dataset_name").num_instances.first().sort_values().index)
    OUT.mkdir(exist_ok=True)
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    lock = threading.Lock()
    print(f"Descargando {len(names)} datasets con {jobs} descargas en paralelo (cambia con --jobs=N)", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(lambda nm: _one(nm, meta, manifest, lock, man_path), names))
    ok = sum(v.get("verified", False) for v in manifest.values())
    bad = [k for k, v in manifest.items() if not v.get("verified")]
    print(f"\n{ok}/{len(manifest)} verificados. Manifest: {man_path}")
    if bad:
        print("Pendientes (vuelve a correr el script; retoma donde se quedó):", ", ".join(sorted(bad)))


if __name__ == "__main__":
    main()
