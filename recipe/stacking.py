"""
Receta de stacking para clasificación tabular (solo NumPy, pandas y scikit-learn; Python 3.11, sklearn >= 1.4).

    model = build_model();  model.fit(X, y);  model.predict_proba(X_test)   # columnas en el orden de classes_

Tres bases diversas. CADA UNA TIENE SU PROPIO PREPROCESAMIENTO, ajustado desde la tabla cruda dentro de su
fit, para que lo que necesita un modelo no perjudique a otro:
  - lineal (regresión logística L2): one-hot, indicadores de faltante, escalado robusto con recorte,
    target encoding para alta cardinalidad
  - no lineal (HistGradientBoosting): categóricas y faltantes nativos, sin escalado
  - datos pequeños (ExtraTrees acotado): códigos ordinales + frecuencia, sin target encoding
Un meta-modelo logístico L2 aprende a mezclarlas usando SOLO predicciones fuera de muestra (OOF): en cada
fold cada base ajusta desde cero su preprocesamiento y su modelo; luego se reentrenan con todas las filas.

Variantes (argumento `variant`): "R1" (receta), "R1-AVG" (promedio simple de las 3 bases),
"B-LIN", "B-HGB", "B-SMALL" (cada base sola, para diagnóstico).
`extra_bases` añade bases externas con la misma interfaz fit(X_crudo, y, K, budget) / predict_proba(X_crudo),
cada una con su propio preprocesamiento (lo usa el experimento con MLP + VOS).

Ningún hiperparámetro se eligió mirando datos de evaluación: son valores por defecto razonables o
reglas fijas por tamaño del dataset.
"""
import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, TargetEncoder

VARIANT = "R1"
SEED = 0

SIZE_BUDGET = 4_400_000      # bytes del pickle completo (límite duro: 5 MB)
SIZE_CAP = 4_600_000         # tope medido: si el pickle real lo pasa, el HGB se reajusta con menos árboles
ET_BUDGET = 700_000
EXTRA_BUDGET = 600_000       # presupuesto por defecto de cada base externa
HGB_BYTES_PER_TREE = 4_500
HGB_MIN_BUDGET = 400_000
OOF_MAX_ROWS = 30_000        # filas usadas para las predicciones OOF del meta-modelo
META_C = 1.0


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


# ================================================================ bases: cada una con SU preprocesamiento
# Cada base recibe la tabla CRUDA y ajusta su propio preprocesamiento dentro de su fit (y por tanto
# dentro de cada fold del stacking). Ninguna comparte codificación, imputación ni escalado con otra.
class _PriorFallback:
    """Si un modelo no tiene ninguna columna utilizable, predice la proporción de clases de train."""

    def _set_prior(self, y, K):
        self.prior_ = np.bincount(y, minlength=K) / len(y)

    def _prior(self, n):
        return np.tile(self.prior_, (n, 1))


class LinearBase(_PriorFallback, BaseEstimator):
    """Regresión logística L2: relaciones lineales, datasets anchos o dispersos.
    Preprocesamiento propio: mediana + indicador de faltante, escalado robusto con recorte (los modelos
    lineales sufren con valores extremos), one-hot de categorías con <= 30 valores (raras agrupadas),
    y para alta cardinalidad target encoding con validación cruzada + log de frecuencia, estandarizados."""

    LOW_MAX = 30
    HC_MAX_CATS = 2_000
    OHE_MIN_FREQ, OHE_MAX_CATS = 10, 30

    def fit(self, X, y, K, budget=None):
        num, cat = _raw_types(X)
        low = [c for c, k in cat.items() if k <= self.LOW_MAX]
        high = [c for c, k in cat.items() if k > self.LOW_MAX]
        parts = []
        if num:
            parts.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                               RobustScaler(), Clipper(5.0)), num))
        if low:
            parts.append(("low", make_pipeline(CategoryCapper(), OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=self.OHE_MIN_FREQ,
                max_categories=self.OHE_MAX_CATS, dtype=np.float32)), low))
        if high:
            parts.append(("high_te", make_pipeline(CategoryCapper(self.HC_MAX_CATS, min_count=2), _target_encoder(),
                                                   StandardScaler(), Clipper(5.0)), high))
            parts.append(("high_cnt", make_pipeline(CountEncoder(), StandardScaler(), Clipper(5.0)), high))
        self.K_ = K
        if not parts:
            self.ct_ = None
            self._set_prior(y, K)
            return self
        self.ct_ = ColumnTransformer(parts, sparse_threshold=0.3)
        Z = self.ct_.fit_transform(X, y)
        C = 1.0 if Z.shape[0] >= 10 * Z.shape[1] else 0.1   # más regularización con pocas filas por columna
        self.lr_ = LogisticRegression(C=C, max_iter=500, random_state=SEED).fit(Z, y)
        return self

    def predict_proba(self, X):
        if self.ct_ is None:
            return self._prior(len(X))
        return _align(self.lr_.predict_proba(self.ct_.transform(X)), self.lr_.classes_, self.K_)


class HGBBase(_PriorFallback, BaseEstimator):
    """HistGradientBoosting: relaciones no lineales e interacciones, datos mixtos y con faltantes.
    Preprocesamiento propio: numéricas tal cual (los faltantes los maneja el modelo), categóricas como
    categóricas NATIVAS (las 254 más frecuentes por columna + 'otras'); las de más de 254 valores además
    con target encoding con validación cruzada y log de frecuencia. Sin escalado (los árboles no lo necesitan).
    Con más de 256 columnas numéricas usa las 256 de mayor F de ANOVA (calculada solo con estas filas)."""

    NATIVE_MAX = 254
    HC_MAX_CATS = 2_000
    MAX_NUM_FEATS = 256

    def fit(self, X, y, K, budget=2_500_000, iter_cap=None):
        num, cat = _raw_types(X)
        if len(num) > self.MAX_NUM_FEATS:
            A = X[num].to_numpy(dtype=np.float32)
            med = np.nan_to_num(np.nanmedian(A, axis=0))
            A = np.where(np.isnan(A), med[None, :], A)
            score = np.nan_to_num(f_classif(A, y)[0], nan=0.0, posinf=0.0)
            num = [num[i] for i in np.sort(np.argsort(-score, kind="stable")[: self.MAX_NUM_FEATS])]
        high = [c for c, k in cat.items() if k > self.NATIVE_MAX]
        parts = []
        if num:
            parts.append(("num", "passthrough", num))
        if cat:
            parts.append(("cat", FreqOrdinal(self.NATIVE_MAX, nan_value=np.nan), list(cat)))
        if high:
            parts.append(("high_te", make_pipeline(CategoryCapper(self.HC_MAX_CATS, min_count=2), _target_encoder()), high))
            parts.append(("high_cnt", CountEncoder(), high))
        self.K_ = K
        if not parts:
            self.ct_ = None
            self._set_prior(y, K)
            return self
        self.ct_ = ColumnTransformer(parts)
        Z = np.asarray(self.ct_.fit_transform(X, y), dtype=np.float32)
        mask = [False] * len(num) + [True] * len(cat) + [False] * (Z.shape[1] - len(num) - len(cat))
        n, counts = len(y), np.bincount(y, minlength=K)
        tpi = 1 if K == 2 else K
        budget = budget - 255 * 8 * Z.shape[1] - 20_000          # umbrales del binning
        max_iter = int(np.clip(max(budget, HGB_MIN_BUDGET) // (HGB_BYTES_PER_TREE * tpi), 50, 1000))
        es = n >= 1500 and counts[counts > 0].min() >= 20
        if es:
            lr = float(np.clip(50.0 / max_iter, 0.05, 0.2))
        else:  # pocos datos: sin early stopping, pocas iteraciones y aprendizaje lento
            max_iter, lr = min(max_iter, 150), 0.05
        if iter_cap is not None:
            # misma configuración y misma semilla con menos iteraciones: los primeros árboles son los mismos
            max_iter = int(max(10, min(max_iter, iter_cap)))
        self.hgb_ = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=max_iter, max_leaf_nodes=31,
            min_samples_leaf=int(min(20, max(2, n // 50))), l2_regularization=1.0,
            categorical_features=mask if any(mask) else None,
            early_stopping=es, validation_fraction=0.1 if es else None, n_iter_no_change=30,
            random_state=SEED,
        ).fit(Z, y)
        return self

    def predict_proba(self, X):
        if self.ct_ is None:
            return self._prior(len(X))
        Z = np.asarray(self.ct_.transform(X), dtype=np.float32)
        return _align(self.hgb_.predict_proba(Z), self.hgb_.classes_, self.K_)


class SmallDataBase(_PriorFallback, BaseEstimator):
    """ExtraTrees de tamaño acotado: promedia muchos árboles aleatorios, sobreajusta poco con pocos datos.
    Preprocesamiento propio y mínimo: mediana + indicador de faltante, categóricas como código ordinal por
    frecuencia (faltante = -1) y log de frecuencia para las de más de 30 valores. Sin target encoding
    (con pocas filas es fácil que sobreajuste) y sin escalado."""

    MAX_CATS = 1_000
    COUNT_MIN = 30
    LEAVES = 64

    def fit(self, X, y, K, budget=ET_BUDGET):
        num, cat = _raw_types(X)
        many = [c for c, k in cat.items() if k > self.COUNT_MIN]
        parts = []
        if num:
            parts.append(("num", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), num))
        if cat:
            parts.append(("cat", FreqOrdinal(self.MAX_CATS, nan_value=-1.0), list(cat)))
        if many:
            parts.append(("cnt", CountEncoder(), many))
        self.K_ = K
        if not parts:
            self.ct_ = None
            self._set_prior(y, K)
            return self
        self.ct_ = ColumnTransformer(parts)
        Z = np.asarray(self.ct_.fit_transform(X, y), dtype=np.float32)
        per_tree = (2 * self.LEAVES - 1) * (64 + 8 * K) + 1_500
        n_est = int(np.clip(budget // per_tree, 20, 150))
        self.et_ = ExtraTreesClassifier(
            n_estimators=n_est, max_leaf_nodes=self.LEAVES, min_samples_leaf=2, max_features="sqrt",
            bootstrap=True, max_samples=min(1.0, 20_000 / len(y)), n_jobs=-1, random_state=SEED,
        ).fit(Z, y)
        return self

    def predict_proba(self, X):
        if self.ct_ is None:
            return self._prior(len(X))
        Z = np.asarray(self.ct_.transform(X), dtype=np.float32)
        return _align(self.et_.predict_proba(Z), self.et_.classes_, self.K_)


# ================================================================ stacking
BASES = {"lin": LinearBase, "hgb": HGBBase, "et": SmallDataBase}
VARIANTS = {
    "R1": (["lin", "hgb", "et"], "meta"),
    "R1-AVG": (["lin", "hgb", "et"], "avg"),
    "B-LIN": (["lin"], None),
    "B-HGB": (["hgb"], None),
    "B-SMALL": (["et"], None),
}


def _meta_features(Ps):
    cols = []
    for P in Ps:
        L = np.log(_clip_p(P))
        cols.append((L[:, 1] - L[:, 0])[:, None] if P.shape[1] == 2 else L - L.mean(axis=1, keepdims=True))
    return np.hstack(cols)


class StackingModel(ClassifierMixin, BaseEstimator):
    def __init__(self, variant=None, extra_bases=None):
        self.variant = variant
        self.extra_bases = extra_bases

    def _make(self, name):
        return BASES[name]() if name in BASES else clone(self.extra_bases[name])

    def _fit_bases(self, X, y, K, final):
        bases = {}
        # primero las bases de tamaño fijo; el HGB usa lo que sobra del presupuesto de pickle
        for name in [b for b in self.bases_ if b != "hgb"]:
            budget = ET_BUDGET if name == "et" else EXTRA_BUDGET
            bases[name] = self._make(name).fit(X, y, K, budget=budget)
        if "hgb" in self.bases_:
            used = sum(_size(b) for b in bases.values()) if final else 0
            budget = SIZE_BUDGET - used - 100_000 if final else SIZE_BUDGET - 1_500_000
            bases["hgb"] = HGBBase().fit(X, y, K, budget=budget)
        return [bases[b] for b in self.bases_]

    @staticmethod
    def _predict_bases(bases, X):
        return [b.predict_proba(X) for b in bases]

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergencia / folds con clases raras: ya se manejan
            return self._fit(X, y)

    def _fit(self, X, y):
        X = _frame(X)
        self.variant_ = self.variant or VARIANT
        names, self.blend_ = VARIANTS[self.variant_]
        self.bases_ = list(names) + list(self.extra_bases or {})
        if self.extra_bases and self.blend_ is None:
            self.blend_ = "meta"
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        K = len(self.classes_)
        self.meta_ = None
        self.info_ = {"variant": self.variant_, "bases": self.bases_}

        if self.blend_ == "meta" and len(self.bases_) > 1:
            # predicciones fuera de muestra: cada base ajusta su preprocesamiento y su modelo solo con el fold
            idx = np.arange(len(y)) if len(y) <= OOF_MAX_ROWS else _stratified_subsample(y, OOF_MAX_ROWS)
            Xo, yo = X.iloc[idx].reset_index(drop=True), y[idx]
            # 5 folds si es chico; 3 si es mediano o ancho (cota de tiempo)
            n_splits = 5 if len(yo) < 5_000 and len(yo) * X.shape[1] <= 2_000_000 else 3
            n_splits = int(max(2, min(n_splits, np.bincount(yo).max())))
            P_oof = [np.zeros((len(yo), K)) for _ in self.bases_]
            for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(Xo, yo):
                bases = self._fit_bases(Xo.iloc[tr].reset_index(drop=True), yo[tr], K, final=False)
                for P, Pb in zip(P_oof, self._predict_bases(bases, Xo.iloc[va].reset_index(drop=True))):
                    P[va] = Pb
            if len(np.unique(yo)) == K:
                self.meta_ = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED).fit(_meta_features(P_oof), yo)
            self.info_.update(oof_rows=int(len(yo)), folds=n_splits, meta=self.meta_ is not None)

        self.models_ = self._fit_bases(X, y, K, final=True)
        self._enforce_size(X, y, K)
        return self

    def _enforce_size(self, X, y, K):
        """El presupuesto del HGB es una estimación; aquí se mide el pickle real. Si pasa de SIZE_CAP, se
        reajusta el HGB (mismas filas, misma semilla) con menos iteraciones, proporcional al exceso."""
        if "hgb" not in self.bases_:
            return
        j = self.bases_.index("hgb")
        for _ in range(3):
            total = _size(self)
            if total <= SIZE_CAP:
                break
            hgb = self.models_[j]
            if getattr(hgb, "hgb_", None) is None:
                break
            size_h = _size(hgb)
            keep = max(0.05, (size_h - (total - SIZE_CAP) - 50_000) / size_h)
            n_iter = int(hgb.hgb_.n_iter_)
            if n_iter <= 10:
                break
            self.models_[j] = HGBBase().fit(X, y, K, iter_cap=int(n_iter * keep))
            self.info_["hgb_iter_recortado"] = [n_iter, int(self.models_[j].hgb_.n_iter_)]
        total = _size(self)
        if total > SIZE_CAP and "et" in self.bases_:
            # último recurso: un ExtraTrees con menos árboles (el promedio de los primeros m árboles es un
            # bosque válido, igual que si se hubiera entrenado con n_estimators = m)
            et = self.models_[self.bases_.index("et")].et_
            size_e = _size(et)
            m = int(max(10, len(et.estimators_) * max(0.05, (size_e - (total - SIZE_CAP) - 50_000) / size_e)))
            et.estimators_ = et.estimators_[:m]
            et.n_estimators = m
            self.info_["et_arboles_recortado"] = m
        self.info_["pickle_bytes"] = _size(self)

    def branch_probas(self, X):
        return dict(zip(self.bases_, self._predict_bases(self.models_, _frame(X))))

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._predict_proba(X)

    def _predict_proba(self, X):
        Ps = self._predict_bases(self.models_, _frame(X))
        if len(Ps) == 1:
            P = Ps[0]
        elif self.meta_ is not None:
            P = _align(self.meta_.predict_proba(_meta_features(Ps)), self.meta_.classes_, len(self.classes_))
        else:
            P = np.mean(Ps, axis=0)
        P = np.clip(np.nan_to_num(P, nan=0.0), 0, None)
        s = P.sum(axis=1, keepdims=True)
        return np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / P.shape[1])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model(variant=None, extra_bases=None):
    return StackingModel(variant=variant or VARIANT, extra_bases=extra_bases)
