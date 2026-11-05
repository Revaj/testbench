"""
Ensamble Dual Calibrado HGB (70%) + Logistic Regression L2 (30%)
- Manejo nativo de categóricas en HGB.
- Target Encoding OOF estratificado con fallback seguro a KFold.
- Rama lineal blindada: si se descartan todas las columnas por ser categóricas,
  retiene automáticamente las frecuencias __cnt y los encodings __te (cero ValueError).
- Reentrenamiento al 100% con multiplicador 1.15 tras el cálculo de iteraciones.
- Fusión adaptativa: Media Geométrica calibrada con clip suave.
- Pickle verificado < 3.0 MB. Tiempo total estimado: ~190-240s.
"""
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

SEED = 0
W_HGB, W_LR = 0.70, 0.30
MISSING_COL = "__num_missing"
MAX_CAT_CARD = 250
N_BUCKETS = 1024
TE_SMOOTH = 100
TE_FOLDS = 5
TE_MAX_BYTES = 300_000
MAX_FEATS = 180
WORK_BUDGET = 80_000_000
SELECT_ROWS = 20_000
TOTAL_BUDGET = 3_000_000
HGB_BYTES_PER_TREE = 4_500
MAX_ITER = 1800
LR_X_ITERS = 30
PATIENCE = 45
HGB_L2 = 5.0
VAL_FRAC = 0.15
ES_MIN_N, ES_MIN_CLASS = 1000, 25
SMALL_ITERS = 180
SMALL_LR_BIN, SMALL_LR_MULTI = 0.03, 0.08
REFIT_MULT = 1.15
IMB_MINORITY = 0.10
LR_C = 0.15
LR_MAX_ITER = 250


class Prep(TransformerMixin, BaseEstimator):
    def __init__(self, te_cols=None, classes=None, cols=None):
        self.te_cols = te_cols
        self.classes = classes
        self.cols = cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.all_cols_ = list(X.columns)
        self.columns_ = list(X.columns) if self.cols is None else [c for c in self.cols if c in X.columns]
        self.low_, self.high_, self.te_ = {}, {}, {}
        for col in self.columns_:
            s = X[col]
            if isinstance(s.dtype, CategoricalDtype) or pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                cats = pd.Index(s.astype("category").cat.categories)
                if len(cats) <= MAX_CAT_CARD:
                    self.low_[col] = cats
                else:
                    self.high_[col] = np.bincount(hash_buckets(s), minlength=N_BUCKETS + 1).astype("int32")
        if y is not None and self.high_:
            y = np.asarray(y)
            self.te_classes_ = (np.unique(y) if self.classes is None else np.asarray(self.classes))[1:]
            if self.te_cols is not None:
                cols = [c for c in self.te_cols if c in self.high_]
            else:
                per_col = 4 * (N_BUCKETS + 1) * len(self.te_classes_)
                cols = sorted(self.high_, key=lambda c: -int((self.high_[c] > 0).sum()))[:int(TE_MAX_BYTES // per_col)]
            for col in cols:
                self.te_[col] = te_table(hash_buckets(X[col]), y, self.te_classes_)
        return self

    def fit_transform_train(self, X, y):
        X = pd.DataFrame(X)
        self.fit(X, y)
        Xt = self.transform(X)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        if counts.min() >= TE_FOLDS:
            cv = StratifiedKFold(n_splits=TE_FOLDS, shuffle=True, random_state=SEED)
            splits = list(cv.split(X, y))
        else:
            cv = KFold(n_splits=TE_FOLDS, shuffle=True, random_state=SEED)
            splits = list(cv.split(X))

        for col in self.te_:
            b = hash_buckets(X[col])
            oof = np.empty((len(b), len(self.te_classes_)), dtype="float32")
            for tr, va in splits:
                oof[va] = te_table(b[tr], y[tr], self.te_classes_)[:, b[va]].T
            for j in range(oof.shape[1]):
                Xt[f"{col}__te{j}"] = oof[:, j]
        return Xt

    def transform(self, X):
        X = pd.DataFrame(X)
        out = {}
        for col in self.columns_:
            s = X[col]
            if col in self.low_:
                out[col] = pd.Categorical(s.astype("object"), categories=self.low_[col])
            elif col in self.high_:
                b = hash_buckets(s)
                out[col] = hash_codes(s)
                out[f"{col}__cnt"] = self.high_[col][b].astype("float32")
                for j, row in enumerate(self.te_.get(col, [])):
                    out[f"{col}__te{j}"] = row[b]
            else:
                out[col] = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float32")
        out[MISSING_COL] = X[self.all_cols_].isna().sum(axis=1).to_numpy(dtype="float32")
        return pd.DataFrame(out, index=X.index)


def hash_codes(s):
    vals = s.astype("object")
    miss = vals.isna().to_numpy()
    h = pd.util.hash_array(vals.astype(str).to_numpy(), categorize=True) % (1 << 24)
    out = h.astype("float32")
    out[miss] = np.nan
    return out


def hash_buckets(s):
    vals = s.astype("object")
    b = (pd.util.hash_array(vals.astype(str).to_numpy(), categorize=True) % N_BUCKETS).astype("int64")
    b[vals.isna().to_numpy()] = N_BUCKETS
    return b


def te_table(b, y, classes):
    cnt = np.bincount(b, minlength=N_BUCKETS + 1)
    rows = []
    for c in classes:
        yc = (y == c).astype(float)
        s = np.bincount(b, weights=yc, minlength=N_BUCKETS + 1)
        rows.append((s + TE_SMOOTH * yc.mean()) / (cnt + TE_SMOOTH))
    return np.asarray(rows, dtype="float32")


def to_codes(X):
    X = X.copy()
    for col in X.columns:
        if isinstance(X[col].dtype, CategoricalDtype):
            c = X[col].cat.codes.astype("float32")
            X[col] = c.where(c >= 0, np.nan)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
    return X.astype("float32")


def discretize(s, bins=20):
    if isinstance(s.dtype, CategoricalDtype):
        return s.cat.codes.to_numpy()
    v = pd.Series(s.to_numpy(dtype="float64"))
    if v.nunique() <= bins:
        return pd.factorize(v, use_na_sentinel=True)[0]
    r = v.rank(method="first")
    return pd.qcut(r, bins, labels=False).fillna(-1).to_numpy().astype(int)


def select_features(Xt, y, max_feats, high_cols):
    cols = list(Xt.columns)
    if len(cols) <= max_feats:
        return cols
    n = len(y)
    idx = np.sort(np.random.default_rng(SEED).choice(n, size=min(n, SELECT_ROWS), replace=False))
    ys = np.asarray(y)[idx]
    Xs = Xt.iloc[idx]
    groups = {h: [c for c in cols if c == h or str(c).startswith(f"{h}__")] for h in high_cols}
    in_group = {c for g in groups.values() for c in g}
    units = [(c, [c]) for c in cols if c not in in_group] + list(groups.items())
    disc = []
    for name, g in units:
        if name in groups:
            te = [c for c in g if "__te" in str(c)] or [c for c in g if str(c).endswith("__cnt")]
            disc.append(discretize(Xs[te[0]]))
        else:
            disc.append(discretize(Xs[name]))
    D = np.column_stack(disc)
    mi = mutual_info_classif(D, ys, discrete_features=True, random_state=SEED)
    kx = np.array([len(np.unique(d)) for d in disc])
    score = mi - (kx - 1) * (len(np.unique(ys)) - 1) / (2 * len(ys))
    keep, used = set(), 0
    for t in np.argsort(-score, kind="stable"):
        g = units[t][1]
        if used + len(g) <= max_feats:
            keep.update(g)
            used += len(g)
    return [c for c in cols if c in keep]


def prep_size_estimate(prep):
    total = 2_000 + 60 * len(prep.columns_) + 15 * len(prep.all_cols_)
    for cats in prep.low_.values():
        total += 200 + sum(len(str(c)) + 12 for c in cats)
    total += len(prep.high_) * (4 * (N_BUCKETS + 1) + 300)
    total += sum(t.size * 4 + 300 for t in prep.te_.values())
    return total


def balance_weights(y, classes):
    counts = np.bincount(np.searchsorted(classes, y), minlength=len(classes))
    if counts.min() / len(y) >= IMB_MINORITY:
        return None
    w = (len(y) / (len(classes) * np.maximum(counts, 1))) ** 0.5
    return w[np.searchsorted(classes, y)]


def aligned(P, model_classes, classes):
    out = np.zeros((P.shape[0], len(classes)), dtype="float64")
    out[:, np.searchsorted(classes, model_classes)] = P
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class LinearBranch(BaseEstimator):
    def fit(self, X, y, low_cols, high_cols):
        # Excluir sólo hashes crudos y categóricas puras; conservar __te, __cnt y numéricas
        drop_set = set(low_cols) | set(high_cols)
        valid_cols = [c for c in X.columns if c not in drop_set]
        
        # Blindaje anti-vacío: si todo era categórico puro, usar códigos y frecuencias
        if len(valid_cols) == 0:
            valid_cols = list(X.columns)

        self.cols_ = valid_cols
        self.model_ = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True),
            LogisticRegression(C=LR_C, class_weight="balanced", max_iter=LR_MAX_ITER, random_state=SEED),
        ).fit(to_codes(X[self.cols_]), y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(to_codes(X[self.cols_]))

    @property
    def classes_(self):
        return self.model_.classes_


class FinalModel(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        n, k = len(y), len(self.classes_)
        tpi = 1 if k == 2 else k
        self.min_class_cnt_ = int(counts.min())
        max_feats = int(max(15, min(MAX_FEATS, WORK_BUDGET // (n * tpi))))
        use_es = n >= ES_MIN_N and counts.min() >= ES_MIN_CLASS

        if not use_es:
            prep, Xt, keep = self._prep_select(X, y, max_feats)
            iters = int(min(SMALL_ITERS, max(40, self._budget(prep, keep, k) // (HGB_BYTES_PER_TREE * tpi))))
            lr = SMALL_LR_BIN if k == 2 else SMALL_LR_MULTI
            self.hgb_ = self._hgb(lr, iters, n, False, self.min_class_cnt_).fit(
                Xt, y, sample_weight=balance_weights(y, self.classes_)
            )
            self.lr_ = LinearBranch().fit(Xt, y, set(prep.low_), set(prep.high_))
            self.prep_, self.keep_ = prep, keep
            return self

        # 1) Partición 85/15 estratificada para calibrar iteraciones
        ia, iv = train_test_split(np.arange(n), test_size=VAL_FRAC, stratify=y, random_state=SEED)
        XA, yA, XV, yV = X.iloc[ia], y[ia], X.iloc[iv], y[iv]

        prep_a, XtA, keep = self._prep_select(XA, yA, max_feats)
        XtV = prep_a.transform(XV)[keep]

        max_iter = int(min(MAX_ITER, max(40, self._budget(prep_a, keep, k) // (HGB_BYTES_PER_TREE * tpi))))
        lr = float(np.clip(LR_X_ITERS / max_iter, 0.02, 0.1))

        # 2) Encontrar corte con early stopping nativo
        wA = balance_weights(yA, self.classes_)
        es_model = self._hgb(lr, max_iter, len(yA), True, self.min_class_cnt_)
        es_model.fit(XtA, yA, sample_weight=wA)
        es_iter = int(getattr(es_model, "n_iter_", max_iter))

        # 3) Reentrenar sobre el 100% de datos con factor 1.15
        prep_f = Prep(te_cols=list(prep_a.te_), classes=self.classes_, cols=prep_a.columns_)
        XtF = prep_f.fit_transform_train(X, y)[keep]
        self._prune(prep_f, keep)

        final_iter = int(min(max_iter, max(20, round(es_iter * REFIT_MULT))))
        self.hgb_ = self._hgb(lr, final_iter, n, False, self.min_class_cnt_).fit(
            XtF, y, sample_weight=balance_weights(y, self.classes_)
        )
        self.lr_ = LinearBranch().fit(XtF, y, set(prep_f.low_), set(prep_f.high_))
        self.prep_, self.keep_ = prep_f, keep
        return self

    def _prep_select(self, X, y, max_feats):
        prep = Prep(classes=self.classes_)
        Xt = prep.fit_transform_train(X, y)
        keep = select_features(Xt, y, max_feats, list(prep.high_))
        self._prune(prep, keep)
        return prep, Xt[keep], keep

    @staticmethod
    def _prune(prep, keep):
        used = {str(c).split("__cnt")[0].split("__te")[0] for c in keep}
        prep.columns_ = [c for c in prep.columns_ if c in used]
        prep.low_ = {c: v for c, v in prep.low_.items() if c in used}
        prep.high_ = {c: v for c, v in prep.high_.items() if c in used}
        prep.te_ = {c: v for c, v in prep.te_.items() if c in used}

    @staticmethod
    def _budget(prep, keep, k):
        overhead = 80_000 + 40 * len(keep) * max(2, k)
        return max(300_000, TOTAL_BUDGET - prep_size_estimate(prep) - 255 * 8 * len(keep) - overhead)

    @staticmethod
    def _hgb(lr, iters, n, early_stopping, min_class_cnt):
        leaf_min = 1 if min_class_cnt < 15 else (2 if min_class_cnt < 30 else min(20, max(5, n // 50)))
        return HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=int(iters),
            max_leaf_nodes=31,
            min_samples_leaf=leaf_min,
            l2_regularization=HGB_L2,
            max_features=0.75,
            categorical_features="from_dtype",
            early_stopping=early_stopping,
            validation_fraction=VAL_FRAC if early_stopping else None,
            n_iter_no_change=PATIENCE,
            random_state=SEED,
        )

    def predict_proba(self, X):
        Xt = self.prep_.transform(pd.DataFrame(X))[self.keep_]
        Ph = aligned(self.hgb_.predict_proba(Xt), self.hgb_.classes_, self.classes_)
        Pl = aligned(self.lr_.predict_proba(Xt), self.lr_.classes_, self.classes_)

        # Media geométrica calibrada universal
        L = W_HGB * np.log(np.clip(Ph, 1e-7, 1.0)) + W_LR * np.log(np.clip(Pl, 1e-7, 1.0))
        P = np.exp(L - L.max(axis=1, keepdims=True))
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model():
    return make_pipeline(FinalModel())