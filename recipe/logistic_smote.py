"""
Regresión logística + SMOTE-NC para clasificación tabular (solo NumPy, pandas, SciPy y scikit-learn;
Python 3.11, sklearn >= 1.4). SMOTE-NC está implementado aquí (sin imbalanced-learn).

    model = build_model();  model.fit(X, y);  model.predict_proba(X_test)   # columnas en el orden de classes_

Preprocesamiento propio, pensado para SMOTE y para un modelo lineal (ver LogisticSmoteModel). SMOTE se aplica
solo a las filas de entrenamiento. Como cambia las proporciones de clase, los logits se corrigen al prior
original (+ log pi/pi'), necesario para log-loss.

Ningún hiperparámetro se eligió mirando datos de evaluación.
"""
import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

SEED = 0

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
    """log(1 + frecuencia en train) de cada categoría. Guarda a lo más `max_cats` conteos por columna;
    las demás (de la cola o nunca vistas) reciben el conteo medio de la cola (0 si no hay cola)."""

    def __init__(self, max_cats=5000):
        self.max_cats = max_cats

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.counts_, self.tail_ = [], []
        for c in X.columns:
            vc = _as_str(X[c]).fillna("__nan__").value_counts()
            self.counts_.append(vc.iloc[: self.max_cats])
            self.tail_.append(float(vc.iloc[self.max_cats:].mean()) if len(vc) > self.max_cats else 0.0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros(X.shape, dtype=np.float32)
        for j, c in enumerate(X.columns):
            v = _as_str(X[c]).fillna("__nan__")
            out[:, j] = np.log1p(v.map(self.counts_[j]).fillna(self.tail_[j]).to_numpy(dtype=np.float32))
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


# ================================================================ SMOTE-NC (numéricas + categóricas)
def smote_nc(N, C, y, rng, k=SMOTE_K):
    """SMOTE-NC (Chawla et al., 2002, sección 6.1), solo con filas de entrenamiento.

    N: bloque numérico ya escalado; C: bloque categórico como códigos enteros.
    Cada clase con menos filas que la mayoritaria se lleva a ese tamaño (tope SMOTE_MAX_ROWS):
      - vecinos de la misma clase con distancia euclídea en N, sumando por cada categoría distinta la
        mediana de las desviaciones estándar numéricas de esa clase (al cuadrado), como en el paper;
      - numéricas del punto sintético: interpolación entre el punto y un vecino;
      - categóricas del punto sintético: la categoría más frecuente entre sus k vecinos.
    Así las categorías siguen siendo válidas (no salen 'medio one-hot'). Clases con 1 fila no se sobremuestrean.
    """
    counts = np.bincount(y)
    target = counts.max()
    room = max(0, SMOTE_MAX_ROWS - len(y))
    need_total = sum(max(0, target - c) for c in counts if c >= 2)
    scale = min(1.0, room / need_total) if need_total else 0.0
    newN, newC, newy = [], [], []
    for c, n_c in enumerate(counts):
        need = int((target - n_c) * scale)
        if need <= 0 or n_c < 2:
            continue
        rows = np.where(y == c)[0]
        if len(rows) > SMOTE_MAX_BASE:
            rows = rng.choice(rows, SMOTE_MAX_BASE, replace=False)
        Nc, Cc = N[rows], C[rows]
        med = float(np.median(Nc.std(axis=0))) if Nc.shape[1] else 1.0
        med = med if med > 0 else 1.0
        if Cc.shape[1]:
            onehot = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(Cc)
            space = np.hstack([Nc, onehot * (med / np.sqrt(2.0))])  # cada categoría distinta suma med^2
        else:
            space = Nc
        kk = min(k, len(rows) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(space).kneighbors(space, return_distance=False)[:, 1:]
        base = rng.integers(0, len(rows), need)
        nb = nn[base, rng.integers(0, kk, need)]
        lam = rng.random((need, 1)).astype(np.float32)
        newN.append((Nc[base] + lam * (Nc[nb] - Nc[base])).astype(np.float32))
        if Cc.shape[1]:
            out = np.empty((need, Cc.shape[1]), Cc.dtype)
            step = max(1, int(2e7 // (kk * kk * Cc.shape[1])))  # por bloques para acotar memoria
            for s0 in range(0, need, step):
                vals = Cc[nn[base[s0:s0 + step]]]                 # (filas, kk, n_cat)
                agree = (vals[:, :, None, :] == vals[:, None, :, :]).sum(axis=2)
                pick = agree.argmax(axis=1)                       # moda; empate -> vecino más cercano
                out[s0:s0 + step] = np.take_along_axis(vals, pick[:, None, :], axis=1)[:, 0, :]
            newC.append(out)
        else:
            newC.append(np.zeros((need, 0), C.dtype))
        newy.append(np.full(need, c))
    if not newN:
        return N, C, y
    return np.vstack([N] + newN), np.vstack([C] + newC), np.concatenate([y] + newy)


# ================================================================ modelo
class LogisticSmoteModel(ClassifierMixin, BaseEstimator):
    """Regresión logística L2 entrenada con SMOTE-NC, con SU PROPIO preprocesamiento:
      - bloque numérico (para distancias e interpolación de SMOTE): mediana, estandarización con recorte
        (SMOTE usa distancia euclídea: todas las columnas deben pesar parecido); para categóricas de más de
        30 valores, target encoding con validación cruzada + log de frecuencia, también estandarizados.
      - bloque categórico (SMOTE lo copia, no lo interpola): categorías de <= 30 valores como códigos e
        indicadores de faltante. Después de SMOTE se pasa a one-hot (raras agrupadas) para la logística.
    Como SMOTE cambia las proporciones de clase, los logits se corrigen al prior original (+ log pi/pi'),
    necesario para log-loss; en AUC binario no cambia el orden."""

    LOW_MAX = 30
    HC_MAX_CATS = 2_000
    OHE_MIN_FREQ, OHE_MAX_CATS = 10, 30

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._fit(X, y)

    def _fit(self, X, y):
        X = _frame(X)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        K = len(self.classes_)
        num, cat = _raw_types(X)
        low = [c for c, k in cat.items() if k <= self.LOW_MAX]
        high = [c for c, k in cat.items() if k > self.LOW_MAX]
        nparts, cparts = [], []
        if num:
            nparts.append(("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                                StandardScaler(), Clipper(5.0)), num))
            cparts.append(("miss", MissingIndicator(features="missing-only", error_on_new=False), num))
        if high:
            nparts.append(("high_te", make_pipeline(CategoryCapper(self.HC_MAX_CATS, min_count=2), _target_encoder(),
                                                    StandardScaler()), high))
            nparts.append(("high_cnt", make_pipeline(CountEncoder(), StandardScaler()), high))
        if low:
            cparts.append(("low", FreqOrdinal(max_cats=1_000, nan_value=-1.0), low))
        self.num_ct_ = ColumnTransformer(nparts, sparse_threshold=0.0) if nparts else None
        self.cat_ct_ = ColumnTransformer(cparts, sparse_threshold=0.0) if cparts else None
        N = self._num(X, y, fit=True)
        C = self._cat(X, y, fit=True)
        Ns, Cs, ys = smote_nc(N, C, y, np.random.default_rng(SEED))
        self.ohe_ = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=self.OHE_MIN_FREQ,
                                  max_categories=self.OHE_MAX_CATS, dtype=np.float32).fit(Cs) if C.shape[1] else None
        Z = self._design(Ns, Cs)
        C_reg = 1.0 if len(y) >= 10 * Z.shape[1] else 0.1
        self.lr_ = LogisticRegression(C=C_reg, max_iter=500, random_state=SEED).fit(Z, ys)
        pi = np.bincount(y, minlength=K) / len(y)
        pi_s = np.bincount(ys, minlength=K) / len(ys)
        cls = self.lr_.classes_
        self.shift_ = np.log(np.maximum(pi[cls], 1e-12) / np.maximum(pi_s[cls], 1e-12))
        self.info_ = {"synthetic_rows": int(len(ys) - len(y))}
        return self

    def _num(self, X, y=None, fit=False):
        if self.num_ct_ is None:
            return np.zeros((len(X), 0), np.float32)
        Z = self.num_ct_.fit_transform(X, y) if fit else self.num_ct_.transform(X)
        return np.asarray(Z, dtype=np.float32)

    def _cat(self, X, y=None, fit=False):
        if self.cat_ct_ is None:
            return np.zeros((len(X), 0), np.int32)
        Z = self.cat_ct_.fit_transform(X, y) if fit else self.cat_ct_.transform(X)
        return np.asarray(Z, dtype=np.float32).astype(np.int32)

    def _design(self, N, C):
        if self.ohe_ is None:
            return N if N.shape[1] else np.zeros((len(N), 1), np.float32)
        return sparse.hstack([sparse.csr_matrix(N), self.ohe_.transform(C)]).tocsr()

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = _frame(X)
            d = self.lr_.decision_function(self._design(self._num(X), self._cat(X)))
        K = len(self.classes_)
        if d.ndim == 1:  # binario: logit de la clase 1
            d = d + self.shift_[1] - self.shift_[0]
            p1 = 1.0 / (1.0 + np.exp(-np.clip(d, -50, 50)))
            P = np.column_stack([1 - p1, p1])
        else:
            L = d + self.shift_
            P = np.exp(L - L.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
        return _align(P, self.lr_.classes_, K)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model():
    return LogisticSmoteModel()


# ================================================================ uso como rama de un stacking
class LRSmoteBase(BaseEstimator):
    """Adaptador para usar esta receta como base extra de stacking.StackingModel (interfaz
    fit(X_crudo, y, K, budget) / predict_proba con K columnas). Conserva TODO su preprocesamiento y su
    SMOTE-NC dentro de su fit: en el stacking eso ocurre solo con las filas de entrenamiento de cada fold,
    nunca con las filas cuyas predicciones ve el meta-modelo."""

    def fit(self, X, y, K, budget=None):
        self.K_ = K
        self.model_ = LogisticSmoteModel().fit(X, y)
        return self

    def predict_proba(self, X):
        return _align(self.model_.predict_proba(X), self.model_.classes_.astype(int), self.K_)
