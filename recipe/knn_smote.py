"""
kNN con o sin SMOTE para clasificación tabular (solo NumPy, pandas y scikit-learn; Python 3.11, sklearn >= 1.4).

    model = build_model();  model.fit(X, y);  model.predict_proba(X_test)   # columnas en el orden de classes_

`build_model(use_smote=False)` es el mismo kNN sin SMOTE: el control del experimento.

Preprocesamiento propio, pensado para distancias: numéricas con mediana y estandarizado recortado; indicadores de
faltante; categóricas de <= 30 valores en one-hot escalado para que una categoría distinta pese como una desviación
estándar; alta cardinalidad con target encoding (validación cruzada) + frecuencia, estandarizados. Si quedan más de
32 dimensiones se proyectan con PCA a 32 (en alta dimensión las distancias pierden sentido y el modelo pesa más).

SMOTE (Chawla et al., 2002) se aplica en ese espacio ya numérico, solo con las filas de entrenamiento. Como kNN guarda
sus ejemplos en el pickle, se conservan a lo más los que caben en el presupuesto (muestra estratificada).
Las probabilidades se corrigen al prior original (pi / pi') porque SMOTE cambia las proporciones de clase, y se suavizan
con un vecino virtual que vota con el prior: (k·p + pi) / (k + 1). Sin eso kNN da probabilidades exactamente 0, que
destruyen el log-loss en multiclase; en binario no cambia el orden (la AUC queda igual).

Ningún hiperparámetro se eligió mirando datos de evaluación: son reglas fijas por tamaño.
"""
import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

USE_SMOTE = True
SEED = 0

KNN_MAX_DIMS = 32
KNN_BYTES = 3_000_000        # bytes para los ejemplos guardados (float32)
KNN_MAX_ROWS = 30_000        # tope de ejemplos guardados (tiempo de predicción)
SMOTE_K = 5
SMOTE_MAX_BASE = 5_000       # puntos base por clase para buscar vecinos
SMOTE_MAX_ROWS = 300_000


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


# ================================================================ SMOTE (espacio numérico)
def smote(Z, y, rng, k=SMOTE_K):
    """Lleva cada clase con >= 2 filas al tamaño de la mayoritaria (tope SMOTE_MAX_ROWS) interpolando entre un
    punto y uno de sus k vecinos más cercanos de la misma clase."""
    counts = np.bincount(y)
    target = counts.max()
    room = max(0, SMOTE_MAX_ROWS - len(y))
    need_total = sum(max(0, target - c) for c in counts if c >= 2)
    scale = min(1.0, room / need_total) if need_total else 0.0
    newZ, newy = [], []
    for c, n_c in enumerate(counts):
        need = int((target - n_c) * scale)
        if need <= 0 or n_c < 2:
            continue
        rows = np.where(y == c)[0]
        if len(rows) > SMOTE_MAX_BASE:
            rows = rng.choice(rows, SMOTE_MAX_BASE, replace=False)
        Zc = Z[rows]
        kk = min(k, len(rows) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Zc).kneighbors(Zc, return_distance=False)[:, 1:]
        base = rng.integers(0, len(rows), need)
        nb = nn[base, rng.integers(0, kk, need)]
        lam = rng.random((need, 1)).astype(np.float32)
        newZ.append((Zc[base] + lam * (Zc[nb] - Zc[base])).astype(np.float32))
        newy.append(np.full(need, c))
    if not newZ:
        return Z, y
    return np.vstack([Z] + newZ), np.concatenate([y] + newy)


# ================================================================ modelo
class KNNModel(ClassifierMixin, BaseEstimator):
    LOW_MAX = 30
    HC_MAX_CATS = 2_000
    OHE_MIN_FREQ, OHE_MAX_CATS = 10, 30

    def __init__(self, use_smote=None):
        self.use_smote = use_smote

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._fit(X, y)

    def _fit(self, X, y):
        X = _frame(X)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        K = len(self.classes_)
        use_smote = USE_SMOTE if self.use_smote is None else self.use_smote
        num, cat = _raw_types(X)
        low = [c for c, k in cat.items() if k <= self.LOW_MAX]
        high = [c for c, k in cat.items() if k > self.LOW_MAX]
        parts = []
        if num:
            parts.append(("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                               StandardScaler(), Clipper(5.0)), num))
            parts.append(("miss", MissingIndicator(features="missing-only", error_on_new=False), num))
        if low:
            parts.append(("low", make_pipeline(CategoryCapper(), OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=self.OHE_MIN_FREQ,
                max_categories=self.OHE_MAX_CATS, sparse_output=False, dtype=np.float32),
                Scale(1 / np.sqrt(2.0))), low))
        if high:
            parts.append(("high_te", make_pipeline(CategoryCapper(self.HC_MAX_CATS, min_count=2), _target_encoder(),
                                                   StandardScaler(), Clipper(5.0)), high))
            parts.append(("high_cnt", make_pipeline(CountEncoder(), StandardScaler(), Clipper(5.0)), high))
        self.ct_ = ColumnTransformer(parts, sparse_threshold=0.0) if parts else None
        Z = self._encode(X, y, fit=True)
        self.pca_ = None
        if Z.shape[1] > KNN_MAX_DIMS:
            sub = Z if len(Z) <= 20_000 else Z[np.random.default_rng(SEED).choice(len(Z), 20_000, replace=False)]
            self.pca_ = PCA(KNN_MAX_DIMS, svd_solver="randomized", random_state=SEED).fit(sub)
            Z = self.pca_.transform(Z).astype(np.float32)
        Zs, ys = smote(Z, y, np.random.default_rng(SEED)) if use_smote else (Z, y)
        cap = int(min(KNN_MAX_ROWS, KNN_BYTES // (4 * Z.shape[1])))
        if len(ys) > cap:
            idx = _stratified_subsample(ys, cap)
            Zs, ys = Zs[idx], ys[idx]
        k = int(np.clip(round(np.sqrt(len(ys)) / 2), 5, 50))
        self.knn_ = KNeighborsClassifier(n_neighbors=min(k, len(ys)), weights="distance", algorithm="brute").fit(Zs, ys)
        pi = np.bincount(y, minlength=K) / len(y)
        pi_s = np.bincount(ys, minlength=K) / len(ys)
        cls = self.knn_.classes_
        self.ratio_ = np.maximum(pi[cls], 1e-12) / np.maximum(pi_s[cls], 1e-12)
        self.prior_ = pi[cls] / pi[cls].sum()
        self.info_ = {"use_smote": bool(use_smote), "k": int(self.knn_.n_neighbors), "stored_rows": int(len(ys)),
                      "dims": int(Zs.shape[1])}
        return self

    def _encode(self, X, y=None, fit=False):
        if self.ct_ is None:
            return np.zeros((len(X), 1), np.float32)
        Z = self.ct_.fit_transform(X, y) if fit else self.ct_.transform(X)
        return np.asarray(Z, dtype=np.float32)

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = self._encode(_frame(X))
            if self.pca_ is not None:
                Z = self.pca_.transform(Z).astype(np.float32)
            P = self.knn_.predict_proba(Z) * self.ratio_      # corrección al prior original
        s = P.sum(axis=1, keepdims=True)
        P = np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / P.shape[1])
        k = self.knn_.n_neighbors                              # un vecino "virtual" con el prior: evita ceros exactos
        P = (k * P + self.prior_) / (k + 1)
        return _align(P, self.knn_.classes_, len(self.classes_))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class Scale(TransformerMixin, BaseEstimator):
    def __init__(self, factor=1.0):
        self.factor = factor

    def fit(self, X, y=None):
        self.n_features_in_ = np.shape(X)[1]
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        return np.asarray(X, dtype=np.float32) * self.factor


def build_model(use_smote=None):
    return KNNModel(use_smote=USE_SMOTE if use_smote is None else use_smote)
