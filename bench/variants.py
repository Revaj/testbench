"""Todas las variantes a comparar, construidas a partir de las tres recetas de recipe/.

    from variants import VARIANTS;  model = VARIANTS["R3"]()

R2 y R3 son el stacking con una 4.a base externa (la MLP de mlp_vos.py sin y con VOS) y R4 (demostrativa)
con logística + SMOTE-NC como 4.a base: así cada receta sigue en su propio archivo, con su propio
preprocesamiento, y no hay código duplicado del stacking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "recipe"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extratrees  # noqa: E402
import knn_smote  # noqa: E402
import logistic_smote  # noqa: E402
import mlp_vos  # noqa: E402
import stacking  # noqa: E402

import baselines  # noqa: E402  (referencias, no recetas)

VARIANTS = {
    "RF": lambda: baselines.DefaultRF(),                 # referencia: ancla del Elo de TabArena
    "R0": lambda: baselines.original(),                 # referencia: la receta original
    "R1": lambda: stacking.build_model("R1"),            # receta de stacking
    "R1-AVG": lambda: stacking.build_model("R1-AVG"),    # mismas 3 bases, promedio simple
    "R2": lambda: stacking.build_model("R1", extra_bases={"mlp": mlp_vos.MLPBase(use_vos=False)}),
    "R3": lambda: stacking.build_model("R1", extra_bases={"mlp_vos": mlp_vos.MLPBase(use_vos=True)}),
    "MLP": lambda: mlp_vos.build_model(use_vos=False),   # control de MLP-VOS
    "MLP-VOS": lambda: mlp_vos.build_model(use_vos=True),
    "LR-SMOTE": lambda: logistic_smote.build_model(),
    # Demostrativa: logística + SMOTE-NC como 4.a rama del stacking, con su propio pipeline encapsulado.
    "R4": lambda: stacking.build_model("R1", extra_bases={"lr_smote": logistic_smote.LRSmoteBase()}),
    # kNN con/sin SMOTE y ExtraTrees con/sin pesos balanceados (cada par: misma receta, una sola diferencia)
    "KNN-SMOTE": lambda: knn_smote.build_model(use_smote=True),
    "KNN": lambda: knn_smote.build_model(use_smote=False),
    "ET-BAL": lambda: extratrees.build_model(class_weight="balanced"),
    "ET": lambda: extratrees.build_model(class_weight=None),
    "B-LIN": lambda: stacking.build_model("B-LIN"),
    "B-HGB": lambda: stacking.build_model("B-HGB"),
    "B-SMALL": lambda: stacking.build_model("B-SMALL"),
}
