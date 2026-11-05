"""Genera archivos de entrega autocontenidos (un solo .py con build_model) a partir de recipe/tabular.py.
Cada uno es una copia exacta con VARIANT fijo, así no hay código duplicado que se desincronice.

    python tools/make_submissions.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "recipe" / "tabular.py").read_text()
TARGETS = {
    "receta_stacking.py": "R1",          # stacking: lineal + HGB + ExtraTrees, meta-modelo sobre OOF
    "receta_stacking_vos.py": "R3",      # R1 + MLP con VOS como 4.a base
    "mlp_vos.py": "MLP-VOS",         # MLP NumPy con VOS, sola
    "logistic_smote.py": "LR-SMOTE", # regresión logística + SMOTE con corrección de prior
}
assert SRC.count('VARIANT = "R1"\n') == 1
out = ROOT / "submissions"
out.mkdir(exist_ok=True)
for name, variant in TARGETS.items():
    header = f"# GENERADO por tools/make_submissions.py desde recipe/tabular.py (VARIANT={variant}). No editar a mano.\n"
    (out / name).write_text(header + SRC.replace('VARIANT = "R1"\n', f'VARIANT = "{variant}"\n'))
    print("escrito", out / name)
