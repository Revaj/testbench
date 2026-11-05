"""Referencias que NO son recetas: solo sirven para comparar.

RF   : RandomForestClassifier por defecto de scikit-learn (100 árboles), el ancla del Elo de TabArena.
       Preprocesamiento mínimo propio: categóricas -> código ordinal, faltantes -> -1 en categóricas y
       mediana en numéricas. Su pickle puede pasar de 5 MB: es solo referencia, no una entrega.
R0   : la receta original (legacy/submission_original.py), sin cambios.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent


class DefaultRF(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        cat = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c]) or not pd.api.types.is_numeric_dtype(X[c])]
        num = [c for c in X.columns if c not in cat]
        parts = []
        if num:
            parts.append(("num", SimpleImputer(strategy="median", keep_empty_features=True), num))
        if cat:
            parts.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                encoded_missing_value=-1), cat))
        self.ct_ = ColumnTransformer(parts)
        Z = self.ct_.fit_transform(X.astype({c: object for c in cat}))
        self.rf_ = RandomForestClassifier(random_state=0, n_jobs=-1).fit(Z, y)
        self.cat_ = cat
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X).astype({c: object for c in self.cat_})
        P = np.zeros((len(X), len(self.classes_)))
        P[:, self.rf_.classes_] = self.rf_.predict_proba(self.ct_.transform(X))
        return P


def _load_original():
    path = ROOT / "legacy" / "submission_original.py"
    spec = importlib.util.spec_from_file_location("submission_original", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_original"] = mod
    spec.loader.exec_module(mod)
    return mod


def original():
    return _load_original().build_model()
