"""
Regresión logística + SMOTE para clasificación tabular (solo NumPy, pandas, SciPy y scikit-learn;
Python 3.11, sklearn >= 1.4). SMOTE está implementado aquí (sin imbalanced-learn).

    model = build_model();  model.fit(X, y);  model.predict_proba(X_test)   # columnas en el orden de classes_

SMOTE se aplica solo a las filas de entrenamiento, en el espacio ya codificado. Como cambia las proporciones
de clase, los logits se corrigen al prior original (+ log pi/pi'), necesario para log-loss.

Ningún hiperparámetro se eligió mirando datos de evaluación.
"""
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, TargetEncoder

SEED = 0

# ---------------------------------------------------------------- reglas fijas (no ajustadas con datos)
LOW_CARD_MAX = 30            # <= 30 categorías: one-hot / categórica nativa; > 30: target encoding
HC_MAX_CATS_TOTAL = 30_000   # categorías guardadas (sumando columnas de alta cardinalidad)
HC_MAX_CATS_COL = 2_000
OHE_MIN_FREQ, OHE_MAX_CATS = 10, 30
CLIP = 5.0

SMOTE_K = 5
SMOTE_MAX_BASE = 5_000       # puntos base por clase para buscar vecinos
SMOTE_MAX_ROWS = 300_000


# ================================================================ utilidades
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


def _lr_C(n, p):
    # más regularización cuando hay pocas filas por columna codificada
    return 1.0 if n >= 10 * max(p, 1) else 0.1


class Clipper(TransformerMixin, BaseEstimator):
    def __init__(self, bound=CLIP):
        self.bound = bound

    def fit(self, X, y=None):
        self.n_features_in_ = np.shape(X)[1]
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=np.float32), -self.bound, self.bound)


def _as_str(s):
    out = pd.Series(s.astype(object).astype(str).to_numpy(dtype=object), index=s.index, dtype=object)
    out[s.isna().to_numpy()] = np.nan
    return out


# ================================================================ featurizer compartido
class Featurizer(BaseEstimator):
    """Tipos de columna + codificación de alta cardinalidad (target encoding OOF + frecuencia).

    Se ajusta dentro de cada fold. Las filas de entrenamiento reciben target encoding
    con validación cruzada (TargetEncoder.fit_transform), así ninguna fila ve su propia etiqueta.
    El resto del preprocesamiento es propio de cada base.
    """

    def fit_transform(self, X, y):
        X = self._frame(X)
        self.num_, self.low_, self.high_, self.keep_ = [], [], [], {}
        for c in X.columns:
            s = X[c]
            if not s.notna().any():
                continue
            if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                (self.low_ if s.nunique(dropna=True) <= LOW_CARD_MAX else self.high_).append(c)
            else:
                self.num_.append(c)
        per_col = int(min(HC_MAX_CATS_COL, HC_MAX_CATS_TOTAL // max(1, len(self.high_))))
        self.counts_, self.te_ = {}, None
        if self.high_:
            H = pd.DataFrame(index=X.index)
            for c in self.high_:
                vc = _as_str(X[c]).value_counts()
                keep = vc[vc >= 2].index[:per_col]
                self.keep_[c] = pd.Index(keep)
                H[c] = self._cap(X[c], self.keep_[c])
                self.counts_[c] = H[c].value_counts()
            self.te_ = TargetEncoder(target_type="auto", smooth="auto", cv=5, shuffle=True, random_state=SEED)
            T = self.te_.fit_transform(H, y)
            return self._assemble(X, H, T)
        return self._assemble(X, None, None)

    def transform(self, X):
        X = self._frame(X)
        if not self.high_:
            return self._assemble(X, None, None)
        H = pd.DataFrame({c: self._cap(X[c], self.keep_[c]) for c in self.high_}, index=X.index)
        return self._assemble(X, H, self.te_.transform(H))

    @staticmethod
    def _frame(X):
        X = pd.DataFrame(X).copy()
        X.columns = [str(c) for c in X.columns]
        return X

    @staticmethod
    def _cap(s, keep):
        v = _as_str(s)
        miss = v.isna()
        v = v.where(v.isin(keep) | miss, "__rare__")
        return v.where(~miss, "__nan__").astype(object)

    def _assemble(self, X, H, T):
        out = {}
        for c in self.num_:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
        if H is not None:
            T = np.asarray(T, dtype="float32")
            per = T.shape[1] // len(self.high_)
            for i, c in enumerate(self.high_):
                for j in range(per):
                    out[f"{c}__te{j}"] = T[:, i * per + j]
                out[f"{c}__cnt"] = np.log1p(H[c].map(self.counts_[c]).fillna(0).to_numpy(dtype="float32"))
        for c in self.low_:
            out[c] = _as_str(X[c])
        F = pd.DataFrame(out, index=X.index)
        self.num_out_ = [c for c in F.columns if c not in self.low_]
        return F


# ================================================================ preprocesamientos por base
def _linear_ct(num, low, dense):
    parts = []
    if num:
        parts.append(("num", make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            RobustScaler(), Clipper()), num))
    if low:
        parts.append(("low", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="__nan__", keep_empty_features=True),
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=OHE_MIN_FREQ,
                          max_categories=OHE_MAX_CATS, sparse_output=not dense, dtype=np.float32)), low))
    return ColumnTransformer(parts, sparse_threshold=0.0 if dense else 0.3)


# ================================================================ LR + SMOTE
def smote(X, y, rng, k=SMOTE_K):
    """SMOTE (Chawla et al., 2002) en el espacio ya codificado, solo con filas de entrenamiento.

    Cada clase con menos filas que la mayoritaria se lleva a ese tamaño interpolando entre un
    punto y uno de sus k vecinos más cercanos de la misma clase. Clases con 1 fila no se sobremuestrean.
    Para acotar tiempo, los vecinos se buscan entre a lo más SMOTE_MAX_BASE puntos por clase.
    """
    counts = np.bincount(y)
    target = counts.max()
    room = max(0, SMOTE_MAX_ROWS - len(y))
    need_total = sum(max(0, target - c) for c in counts if c >= 2)
    scale = min(1.0, room / need_total) if need_total else 0.0
    new_X, new_y = [], []
    for c, n_c in enumerate(counts):
        need = int((target - n_c) * scale)
        if need <= 0 or n_c < 2:
            continue
        rows = np.where(y == c)[0]
        if len(rows) > SMOTE_MAX_BASE:
            rows = rng.choice(rows, SMOTE_MAX_BASE, replace=False)
        Xc = X[rows]
        Xc = Xc.toarray() if sparse.issparse(Xc) else np.asarray(Xc)
        kk = min(k, len(rows) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc).kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.integers(0, len(rows), need)
        nb = nn[base, rng.integers(0, kk, need)]
        lam = rng.random((need, 1)).astype(np.float32)
        new_X.append((Xc[base] + lam * (Xc[nb] - Xc[base])).astype(np.float32))
        new_y.append(np.full(need, c))
    if not new_X:
        return X, y
    S = np.vstack(new_X)
    Xn = sparse.vstack([X, sparse.csr_matrix(S)]).tocsr() if sparse.issparse(X) else np.vstack([X, S])
    return Xn, np.concatenate([y] + new_y)


class LRSmoteBase(BaseEstimator):
    """Regresión logística con SMOTE. Como SMOTE cambia las proporciones de clase, las probabilidades
    se corrigen al prior original sumando log(pi_c / pi'_c) a cada logit (necesario para log-loss;
    en AUC binario no cambia el orden)."""

    def fit(self, F, y, K, budget=None, low=()):
        low = [c for c in F.columns if c in set(low)]
        num = [c for c in F.columns if c not in set(low)]
        self.ct_ = _linear_ct(num, low, dense=False)
        Z = self.ct_.fit_transform(F)
        Zs, ys = smote(Z, y, np.random.default_rng(SEED))
        self.lr_ = LogisticRegression(C=_lr_C(Z.shape[0], Z.shape[1]), max_iter=500, random_state=SEED).fit(Zs, ys)
        pi = np.bincount(y, minlength=K) / len(y)
        pi_s = np.bincount(ys, minlength=K) / len(ys)
        cls = self.lr_.classes_
        self.shift_ = np.log(np.maximum(pi[cls], 1e-12) / np.maximum(pi_s[cls], 1e-12))
        self.n_synthetic_ = len(ys) - len(y)
        self.K_ = K
        return self

    def predict_proba(self, F):
        d = self.lr_.decision_function(self.ct_.transform(F))
        if d.ndim == 1:  # binario: logit de la clase 1
            d = d + self.shift_[1] - self.shift_[0]
            p1 = 1.0 / (1.0 + np.exp(-d))
            P = np.column_stack([1 - p1, p1])
        else:
            L = d + self.shift_
            P = np.exp(L - L.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
        return _align(P, self.lr_.classes_, self.K_)



# ================================================================ modelo
class LogisticSmoteModel(ClassifierMixin, BaseEstimator):
    """Featurizer + regresión logística entrenada con SMOTE y corrección de prior."""

    def __init__(self):
        pass

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._fit(X, y)

    def _fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        K = len(self.classes_)
        self.feat_ = Featurizer()
        F = self.feat_.fit_transform(X, y)
        self.base_ = LRSmoteBase().fit(F, y, K, low=self.feat_.low_)
        self.info_ = {"synthetic_rows": int(self.base_.n_synthetic_)}
        return self

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = self.base_.predict_proba(self.feat_.transform(pd.DataFrame(X).reset_index(drop=True)))
        P = np.clip(np.nan_to_num(P, nan=0.0), 0, None)
        s = P.sum(axis=1, keepdims=True)
        return np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / P.shape[1])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model():
    return LogisticSmoteModel()
