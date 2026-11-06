"""
ExtraTrees con o sin pesos de clase balanceados para clasificación tabular (solo NumPy, pandas y scikit-learn;
Python 3.11, sklearn >= 1.4).

    model = build_model();  model.fit(X, y);  model.predict_proba(X_test)   # columnas en el orden de classes_

`build_model(class_weight=None)` es el mismo bosque sin pesos: el control del experimento.
`class_weight="balanced"` hace que cada error en una clase rara cuente más al elegir los cortes, sin inventar filas.

Preprocesamiento propio y mínimo (los árboles no necesitan escalar): mediana + indicador de faltante en numéricas;
categóricas como código ordinal por frecuencia (faltante = -1) y log de frecuencia para las de más de 30 valores.
Tamaño acotado: árboles de 256 hojas, tantos como quepan en ~3.5 MB (entre 30 y 300).

Ningún hiperparámetro se eligió mirando datos de evaluación: son reglas fijas por tamaño.
"""
import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder

CLASS_WEIGHT = "balanced"
SEED = 0

ET_BYTES = 3_500_000
ET_LEAVES = 256
ET_MAX_ROWS_PER_TREE = 50_000


# ================================================================ utilidades (no preprocesan nada)
def _clip_p(P):
    return np.clip(P, 1e-7, 1.0)


def _align(P, model_classes, n_classes):
    """Coloca las columnas de predict_proba de un modelo en las posiciones 0..K-1."""
    out = np.zeros((P.shape[0], n_classes))
    out[:, np.asarray(model_classes, dtype=int)] = P
    s = out.sum(axis=1, keepdims=True)
    return np.where(s > 0, out / np.where(s > 0, s, 1), 1.0 / n_classes)


def _size(obj):
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def _frame(X):
    X = pd.DataFrame(X).reset_index(drop=True)
    X.columns = [str(c) for c in X.columns]
    return X


def _as_str(s):
    s = pd.Series(s)
    out = pd.Series(s.astype(object).astype(str).to_numpy(dtype=object), index=s.index, dtype=object)
    out[s.isna().to_numpy()] = np.nan
    return out


def _raw_types(X):
    """Solo inspección de la tabla cruda: columnas numéricas y categóricas (con su número de valores).
    Cada modelo decide después, con sus propios umbrales, cómo tratar cada columna."""
    num, cat = [], {}
    for c in X.columns:
        s = X[c]
        if not s.notna().any():
            continue
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            cat[c] = int(s.nunique(dropna=True))
        else:
            num.append(c)
    return num, cat


def _stratified_subsample(y, m):
    try:
        idx, _ = train_test_split(np.arange(len(y)), train_size=m, stratify=y, random_state=SEED)
    except ValueError:  # clases con un solo ejemplo
        idx = np.random.default_rng(SEED).choice(len(y), m, replace=False)
    return np.sort(idx)


def _target_encoder():
    """TargetEncoder con validación cruzada interna (fit_transform nunca usa la etiqueta de la propia fila),
    reproducible en todas las versiones: hasta sklearn 1.8 con shuffle + random_state; desde 1.9 (donde esos
    parámetros son obsoletos y, si no, baraja al azar) con un StratifiedKFold con semilla fija."""
    if inspect.signature(TargetEncoder).parameters["shuffle"].default != "deprecated":
        return TargetEncoder(cv=5, shuffle=True, random_state=SEED)
    return TargetEncoder(cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


# ================================================================ piezas de preprocesamiento
# Son herramientas; cada modelo crea SUS PROPIAS instancias con SUS PROPIOS parámetros.
class Clipper(TransformerMixin, BaseEstimator):
    def __init__(self, bound=5.0):
        self.bound = bound

    def fit(self, X, y=None):
        self.n_features_in_ = np.shape(X)[1]
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=np.float32), -self.bound, self.bound)


class CategoryCapper(TransformerMixin, BaseEstimator):
    """Conserva las `max_cats` categorías más frecuentes con al menos `min_count` filas.
    El resto (y las nunca vistas) -> '__rare__'; faltante -> '__nan__'. Devuelve texto."""

    def __init__(self, max_cats=1000, min_count=1):
        self.max_cats = max_cats
        self.min_count = min_count

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = []
        for c in X.columns:
            vc = _as_str(X[c]).value_counts()
            self.keep_.append(pd.Index(vc[vc >= self.min_count].index[: self.max_cats]))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.empty(X.shape, dtype=object)
        for j, c in enumerate(X.columns):
            v = _as_str(X[c])
            arr = np.where(v.isin(self.keep_[j]).to_numpy(), v.to_numpy(dtype=object), "__rare__")
            arr[v.isna().to_numpy()] = "__nan__"
            out[:, j] = arr
        return out


class CountEncoder(TransformerMixin, BaseEstimator):
    """log(1 + frecuencia en train) de cada categoría. Guarda, de forma compacta (hash de 64 bits + conteo),
    las categorías con al menos 2 filas, a lo más `max_total` sumando columnas y `max_cats` por columna;
    las demás (cola, únicas o nunca vistas) reciben el conteo medio de la cola."""

    def __init__(self, max_cats=1000, max_total=20_000):
        self.max_cats = max_cats
        self.max_total = max_total

    @staticmethod
    def _hash(v):
        return pd.util.hash_array(v.to_numpy(dtype=object), categorize=True)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cap = int(max(50, min(self.max_cats, self.max_total // max(1, X.shape[1]))))
        self.keys_, self.vals_, self.tail_ = [], [], []
        for c in X.columns:
            vc = _as_str(X[c]).fillna("__nan__").value_counts()
            keep = vc[vc >= 2].iloc[:cap]
            rest = vc.iloc[len(keep):]
            h = self._hash(pd.Series(keep.index))
            order = np.argsort(h)
            self.keys_.append(h[order])
            self.vals_.append(np.log1p(keep.to_numpy(dtype=np.float32))[order])
            self.tail_.append(float(np.log1p(rest.mean())) if len(rest) else 0.0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros(X.shape, dtype=np.float32)
        for j, c in enumerate(X.columns):
            keys = self.keys_[j]
            h = self._hash(_as_str(X[c]).fillna("__nan__"))
            if len(keys) == 0:
                out[:, j] = self.tail_[j]
                continue
            pos = np.minimum(np.searchsorted(keys, h), len(keys) - 1)
            hit = keys[pos] == h
            out[:, j] = np.where(hit, self.vals_[j][pos], self.tail_[j])
        return out


class FreqOrdinal(TransformerMixin, BaseEstimator):
    """Código entero por rango de frecuencia (0 = más frecuente) para las `max_cats` categorías más comunes;
    rara o nunca vista -> max_cats; faltante -> `nan_value`. Devuelve float32."""

    def __init__(self, max_cats=254, nan_value=np.nan):
        self.max_cats = max_cats
        self.nan_value = nan_value

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = []
        for c in X.columns:
            vc = _as_str(X[c]).value_counts()
            self.maps_.append(pd.Series(np.arange(min(len(vc), self.max_cats), dtype=np.float32),
                                        index=vc.index[: self.max_cats]))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.empty(X.shape, dtype=np.float32)
        for j, c in enumerate(X.columns):
            v = _as_str(X[c])
            code = np.array(v.map(self.maps_[j]).to_numpy(dtype=np.float32), copy=True)
            code[np.isnan(code)] = self.max_cats
            code[v.isna().to_numpy()] = self.nan_value
            out[:, j] = code
        return out


# ================================================================ modelo
class ExtraTreesModel(ClassifierMixin, BaseEstimator):
    MAX_CATS = 1_000
    COUNT_MIN = 30

    def __init__(self, class_weight="default"):
        self.class_weight = class_weight

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._fit(X, y)

    def _fit(self, X, y):
        X = _frame(X)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        K = len(self.classes_)
        cw = CLASS_WEIGHT if self.class_weight == "default" else self.class_weight
        num, cat = _raw_types(X)
        many = [c for c, k in cat.items() if k > self.COUNT_MIN]
        parts = []
        if num:
            parts.append(("num", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), num))
        if cat:
            parts.append(("cat", FreqOrdinal(self.MAX_CATS, nan_value=-1.0), list(cat)))
        if many:
            parts.append(("cnt", CountEncoder(), many))
        self.ct_ = ColumnTransformer(parts) if parts else None
        Z = self._encode(X, y, fit=True)
        per_tree = (2 * ET_LEAVES - 1) * (64 + 8 * K) + 1_500
        n_est = int(np.clip(ET_BYTES // per_tree, 30, 300))
        self.et_ = ExtraTreesClassifier(
            n_estimators=n_est, max_leaf_nodes=ET_LEAVES, min_samples_leaf=2, max_features="sqrt",
            bootstrap=True, max_samples=min(1.0, ET_MAX_ROWS_PER_TREE / len(y)), class_weight=cw,
            n_jobs=-1, random_state=SEED,
        ).fit(Z, y)
        self.info_ = {"class_weight": str(cw), "trees": n_est}
        return self

    def _encode(self, X, y=None, fit=False):
        if self.ct_ is None:
            return np.zeros((len(X), 1), np.float32)
        Z = self.ct_.fit_transform(X, y) if fit else self.ct_.transform(X)
        return np.asarray(Z, dtype=np.float32)

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = self.et_.predict_proba(self._encode(_frame(X)))
        return _align(P, self.et_.classes_, len(self.classes_))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model(class_weight="default"):
    return ExtraTreesModel(class_weight=class_weight)
