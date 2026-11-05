"""
Receta de stacking para clasificación tabular (solo NumPy, pandas y scikit-learn; Python 3.11, sklearn >= 1.4).

    model = build_model();  model.fit(X, y);  model.predict_proba(X_test)   # columnas en el orden de classes_

Tres bases diversas, cada una con su propio preprocesamiento:
  - lineal: regresión logística L2 (one-hot, indicadores de faltante, escalado robusto)
  - no lineal: HistGradientBoosting (categóricas y faltantes nativos, early stopping interno)
  - datos pequeños: ExtraTrees de tamaño acotado
Un meta-modelo logístico L2 aprende a mezclarlas usando SOLO predicciones fuera de muestra (OOF): en cada
fold se ajustan desde cero el featurizer, el target encoding y las bases; luego las bases se reentrenan
con todas las filas.

Variantes (argumento `variant`): "R1" (receta), "R1-AVG" (promedio simple de las 3 bases),
"B-LIN", "B-HGB", "B-SMALL" (cada base sola, para diagnóstico).
`extra_bases` permite añadir bases externas con la misma interfaz (lo usa el experimento con MLP + VOS).

Ningún hiperparámetro se eligió mirando datos de evaluación: son valores por defecto razonables o
reglas fijas por tamaño del dataset.
"""
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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, TargetEncoder

VARIANT = "R1"
SEED = 0

# ---------------------------------------------------------------- reglas fijas (no ajustadas con datos)
LOW_CARD_MAX = 30            # <= 30 categorías: one-hot / categórica nativa; > 30: target encoding
HC_MAX_CATS_TOTAL = 30_000   # categorías guardadas (sumando columnas de alta cardinalidad)
HC_MAX_CATS_COL = 2_000
OHE_MIN_FREQ, OHE_MAX_CATS = 10, 30
CLIP = 5.0

SIZE_BUDGET = 4_400_000      # bytes del pickle completo (límite duro: 5 MB)
ET_BUDGET = 700_000
EXTRA_BUDGET = 600_000       # presupuesto por defecto de cada base externa
HGB_BYTES_PER_TREE = 4_500
HGB_MIN_BUDGET = 400_000
HGB_MAX_NUM_FEATS = 256      # con más columnas numéricas, el HGB usa las 256 con mayor F de ANOVA (en el fold)

OOF_MAX_ROWS = 30_000        # filas usadas para las predicciones OOF del meta-modelo
META_C = 1.0


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


class LinearBase(BaseEstimator):
    """Relaciones lineales; buena con muchas columnas o datos dispersos."""

    def fit(self, F, y, K, budget=None, low=()):
        low = [c for c in F.columns if c in set(low)]
        num = [c for c in F.columns if c not in set(low)]
        self.ct_ = _linear_ct(num, low, dense=False)
        Z = self.ct_.fit_transform(F)
        self.lr_ = LogisticRegression(C=_lr_C(Z.shape[0], Z.shape[1]), max_iter=500, random_state=SEED).fit(Z, y)
        self.K_ = K
        return self

    def predict_proba(self, F):
        return _align(self.lr_.predict_proba(self.ct_.transform(F)), self.lr_.classes_, self.K_)


class HGBBase(BaseEstimator):
    """Relaciones no lineales e interacciones; categóricas y faltantes nativos."""

    def fit(self, F, y, K, budget=2_500_000, low=()):
        low = [c for c in F.columns if c in set(low)]
        num = [c for c in F.columns if c not in set(low)]
        if len(num) > HGB_MAX_NUM_FEATS:
            # muchas columnas: el binning de HGB guarda ~255 umbrales por columna (tamaño) y cada
            # iteración recorre todas (tiempo). Filtro univariado ajustado solo con este fold.
            A = F[num].to_numpy(dtype=np.float32)
            med = np.nan_to_num(np.nanmedian(A, axis=0))
            A = np.where(np.isnan(A), med[None, :], A)
            score = np.nan_to_num(f_classif(A, y)[0], nan=0.0, posinf=0.0)
            num = [num[i] for i in np.sort(np.argsort(-score, kind="stable")[:HGB_MAX_NUM_FEATS])]
        parts = []
        if num:
            parts.append(("num", "passthrough", num))
        if low:
            parts.append(("low", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                                encoded_missing_value=np.nan, max_categories=250), low))
        self.ct_ = ColumnTransformer(parts)
        Z = np.asarray(self.ct_.fit_transform(F), dtype=np.float32)
        mask = [False] * len(num) + [True] * len(low)
        n, counts = len(y), np.bincount(y, minlength=K)
        tpi = 1 if K == 2 else K
        budget = budget - 255 * 8 * (len(num) + len(low)) - 20_000   # umbrales del binning
        max_iter = int(np.clip(max(budget, HGB_MIN_BUDGET) // (HGB_BYTES_PER_TREE * tpi), 50, 1000))
        es = n >= 1500 and counts[counts > 0].min() >= 20
        if es:
            lr = float(np.clip(50.0 / max_iter, 0.05, 0.2))
        else:  # pocos datos: sin early stopping, pocas iteraciones y aprendizaje lento
            max_iter, lr = min(max_iter, 150), 0.05
        self.hgb_ = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=max_iter, max_leaf_nodes=31,
            min_samples_leaf=int(min(20, max(2, n // 50))), l2_regularization=1.0,
            categorical_features=mask if any(mask) else None,
            early_stopping=es, validation_fraction=0.1 if es else None, n_iter_no_change=30,
            random_state=SEED,
        ).fit(Z, y)
        self.K_ = K
        return self

    def predict_proba(self, F):
        Z = np.asarray(self.ct_.transform(F), dtype=np.float32)
        return _align(self.hgb_.predict_proba(Z), self.hgb_.classes_, self.K_)


class SmallDataBase(BaseEstimator):
    """ExtraTrees de tamaño acotado: promedia muchos árboles aleatorios, sobreajusta poco con pocos datos."""

    def fit(self, F, y, K, budget=ET_BUDGET, low=()):
        low = [c for c in F.columns if c in set(low)]
        num = [c for c in F.columns if c not in set(low)]
        parts = []
        if num:
            parts.append(("num", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), num))
        if low:
            parts.append(("low", make_pipeline(
                SimpleImputer(strategy="constant", fill_value="__nan__", keep_empty_features=True),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), low))
        self.ct_ = ColumnTransformer(parts)
        Z = np.asarray(self.ct_.fit_transform(F), dtype=np.float32)
        leaves = 64
        per_tree = (2 * leaves - 1) * (64 + 8 * K) + 1_500
        n_est = int(np.clip(budget // per_tree, 20, 150))
        n = len(y)
        self.et_ = ExtraTreesClassifier(
            n_estimators=n_est, max_leaf_nodes=leaves, min_samples_leaf=2, max_features="sqrt",
            bootstrap=True, max_samples=min(1.0, 20_000 / n), n_jobs=-1, random_state=SEED,
        ).fit(Z, y)
        self.K_ = K
        return self

    def predict_proba(self, F):
        Z = np.asarray(self.ct_.transform(F), dtype=np.float32)
        return _align(self.et_.predict_proba(Z), self.et_.classes_, self.K_)


def _stratified_subsample(y, m):
    try:
        idx, _ = train_test_split(np.arange(len(y)), train_size=m, stratify=y, random_state=SEED)
    except ValueError:  # clases con un solo ejemplo
        idx = np.random.default_rng(SEED).choice(len(y), m, replace=False)
    return np.sort(idx)


# ================================================================ modelo
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
        feat = Featurizer()
        F = feat.fit_transform(X, y)
        low = feat.low_
        bases = {}
        # primero las bases de tamaño fijo; el HGB usa lo que sobra del presupuesto
        for name in [b for b in self.bases_ if b != "hgb"]:
            budget = ET_BUDGET if name == "et" else EXTRA_BUDGET
            bases[name] = self._make(name).fit(F, y, K, budget=budget, low=low)
        if "hgb" in self.bases_:
            used = _size(feat) + sum(_size(b) for b in bases.values()) if final else 0
            budget = SIZE_BUDGET - used - 100_000 if final else SIZE_BUDGET - 1_500_000
            bases["hgb"] = HGBBase().fit(F, y, K, budget=budget, low=low)
        return feat, [bases[b] for b in self.bases_]

    @staticmethod
    def _predict_bases(feat, bases, X):
        F = feat.transform(X)
        return [b.predict_proba(F) for b in bases]

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergencia / folds con clases raras: ya se manejan
            return self._fit(X, y)

    def _fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
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
            # predicciones fuera de muestra: todo (featurizer, target encoding, bases) se ajusta en el fold
            idx = np.arange(len(y)) if len(y) <= OOF_MAX_ROWS else _stratified_subsample(y, OOF_MAX_ROWS)
            Xo, yo = X.iloc[idx].reset_index(drop=True), y[idx]
            # 5 folds si es chico; 3 si es mediano o ancho (cota de tiempo)
            n_splits = 5 if len(yo) < 5_000 and len(yo) * X.shape[1] <= 2_000_000 else 3
            n_splits = int(max(2, min(n_splits, np.bincount(yo).max())))
            P_oof = [np.zeros((len(yo), K)) for _ in self.bases_]
            for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(Xo, yo):
                feat, bases = self._fit_bases(Xo.iloc[tr], yo[tr], K, final=False)
                for P, Pb in zip(P_oof, self._predict_bases(feat, bases, Xo.iloc[va])):
                    P[va] = Pb
            if len(np.unique(yo)) == K:
                self.meta_ = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED).fit(_meta_features(P_oof), yo)
            self.info_.update(oof_rows=int(len(yo)), folds=n_splits, meta=self.meta_ is not None)

        self.feat_, self.models_ = self._fit_bases(X, y, K, final=True)
        return self

    def branch_probas(self, X):
        return dict(zip(self.bases_, self._predict_bases(self.feat_, self.models_, pd.DataFrame(X).reset_index(drop=True))))

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._predict_proba(X)

    def _predict_proba(self, X):
        Ps = self._predict_bases(self.feat_, self.models_, pd.DataFrame(X).reset_index(drop=True))
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
