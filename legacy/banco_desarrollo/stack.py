"""
Receta con mezcla aprendida fuera de muestra (OOF / holdout).

Bases (las mismas que la receta original, con dos arreglos):
  - HGB con categóricas nativas + target encoding OOF de alta cardinalidad.
  - Regresión logística L2, ahora con one-hot de las categóricas de baja cardinalidad
    (antes las descartaba y en datasets 100 % categóricos quedaba constante, AUC 0.5).
  - (opcional, MODE con "D") ExtraTrees de tamaño acotado.

Mezcla (MODE):
  "A"  pesos fijos 70/30 (receta original, para comparar)
  "B"  pesos de la media geométrica elegidos maximizando el ROC-AUC de predicciones OOF
  "C"  stacking: regresión logística L2 sobre los logits OOF de cada base
  "BD" / "CD" igual, con ExtraTrees como tercera base

Las predicciones que ve el meta-modelo salen de modelos cuyo preprocesamiento, selección de
variables, target encoder y early stopping se ajustaron sólo con las filas de entrenamiento de
ese fold (o de la partición A cuando se usa holdout); nunca con las filas que predicen.
"""
import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

SEED = 0
MODE = os.environ.get("RECIPE_MODE", "C")
W_HGB, W_LR, W_ET = 0.70, 0.30, 0.0
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
OHE_MAX_CATS = 30
# --- mezcla aprendida ---
HOLDOUT_MIN_N = 20_000       # desde aquí basta el 15 % de holdout que ya existía
HOLDOUT_MIN_CELLS = 3_000_000  # o si filas x columnas es grande (datasets anchos)
OOF_FOLDS_SMALL, OOF_FOLDS_MED = 5, 3
META_C = 1.0
GRID_STEP = 0.1
GRID_TOL = 5e-4              # empate: preferir el punto más cercano a 70/30
# --- ExtraTrees ---
ET_BYTES = 900_000
ET_LEAVES = 128


class Prep(TransformerMixin, BaseEstimator):
    def __init__(self, te_cols=None, classes=None, cols=None, high_cols=None):
        self.te_cols = te_cols
        self.classes = classes
        self.cols = cols
        self.high_cols = high_cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.all_cols_ = list(X.columns)
        self.columns_ = list(X.columns) if self.cols is None else [c for c in self.cols if c in X.columns]
        # Arreglo: una columna 100 % vacía en train tumba el binning de HGB
        # ("window shape cannot be larger than input array shape").
        self.columns_ = [c for c in self.columns_ if X[c].notna().any()]
        self.low_, self.high_, self.te_ = {}, {}, {}
        for col in self.columns_:
            s = X[col]
            if isinstance(s.dtype, CategoricalDtype) or pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                cats = pd.Index(s.astype("category").cat.categories)
                # Arreglo: al reentrenar se respeta la partición baja/alta decidida en la fase A,
                # para que las columnas de keep_ sigan existiendo con el mismo significado.
                is_high = (col in self.high_cols) if self.high_cols is not None else len(cats) > MAX_CAT_CARD
                if not is_high and len(cats) <= MAX_CAT_CARD:
                    self.low_[col] = cats
                elif not is_high:
                    # pasó de 250 a >250 al ver más filas: conservar las 250 más frecuentes
                    self.low_[col] = pd.Index(s.value_counts().index[:MAX_CAT_CARD])
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
            splits = list(StratifiedKFold(n_splits=TE_FOLDS, shuffle=True, random_state=SEED).split(X, y))
        else:
            splits = list(KFold(n_splits=TE_FOLDS, shuffle=True, random_state=SEED).split(X))
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


def logp(P):
    return np.log(np.clip(P, 1e-7, 1.0))


def geo_blend(Ps, w):
    L = sum(wi * logp(P) for wi, P in zip(w, Ps) if wi != 0)
    P = np.exp(L - L.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)


def auc_score(y, P, classes):
    if len(classes) == 2:
        return roc_auc_score(y == classes[1], P[:, 1])
    present = [j for j, c in enumerate(classes) if 0 < (y == c).sum() < len(y)]
    return float(np.mean([roc_auc_score(y == classes[j], P[:, j]) for j in present]))


def meta_features(Ps):
    feats = []
    for P in Ps:
        L = logp(P)
        if P.shape[1] == 2:
            feats.append((L[:, 1] - L[:, 0])[:, None])
        else:
            feats.append(L - L.mean(axis=1, keepdims=True))
    return np.hstack(feats)


class LinearBranch(BaseEstimator):
    def fit(self, X, y, low_cols, high_cols):
        num = [c for c in X.columns if c not in low_cols and c not in high_cols]
        cat = [c for c in X.columns if c in low_cols]
        self.num_, self.cat_ = num, cat
        parts = []
        if num:
            parts.append(("num", make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)), num))
        if cat:
            parts.append(("cat", make_pipeline(
                SimpleImputer(strategy="constant", fill_value="__nan__", keep_empty_features=True),
                OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=OHE_MAX_CATS,
                              min_frequency=5, dtype=np.float32)), cat))
        self.model_ = make_pipeline(
            ColumnTransformer(parts, sparse_threshold=0.3),
            LogisticRegression(C=LR_C, class_weight="balanced", max_iter=LR_MAX_ITER, random_state=SEED),
        ).fit(self._frame(X), y)
        return self

    def _frame(self, X):
        out = to_codes(X[self.num_]) if self.num_ else pd.DataFrame(index=X.index)
        for c in self.cat_:
            out[c] = X[c].astype("object").where(X[c].notna(), "__nan__").astype(str)
        return out

    def predict_proba(self, X):
        return self.model_.predict_proba(self._frame(X))

    @property
    def classes_(self):
        return self.model_.classes_


class TreeBranch(BaseEstimator):
    """ExtraTrees con tamaño acotado. Faltantes -> centinela (versión-segura, no depende
    del soporte nativo de NaN que sólo existe en versiones recientes de scikit-learn)."""

    def fit(self, X, y, k, sample_weight=None):
        leaves = ET_LEAVES
        bytes_per_tree = (2 * leaves) * (64 + 8 * k) + 1_000
        n_est = int(np.clip(ET_BYTES // bytes_per_tree, 20, 150))
        self.model_ = ExtraTreesClassifier(
            n_estimators=n_est, max_leaf_nodes=leaves, min_samples_leaf=3, max_features="sqrt",
            bootstrap=True, max_samples=min(1.0, 20_000 / len(y)), n_jobs=1, random_state=SEED,
        ).fit(self._X(X), y, sample_weight=sample_weight)
        return self

    @staticmethod
    def _X(X):
        return np.nan_to_num(to_codes(X).to_numpy(), nan=-9.0e8)

    def predict_proba(self, X):
        return self.model_.predict_proba(self._X(X))

    @property
    def classes_(self):
        return self.model_.classes_


class FinalModel(ClassifierMixin, BaseEstimator):
    def __init__(self, mode=None):
        self.mode = mode

    # ---------- entrenamiento de las bases sobre un subconjunto ----------
    def _fit_bases(self, X, y, max_feats, use_es, with_et):
        """Ajusta prep + selección + bases sólo con (X, y). Devuelve un objeto para predecir."""
        k = len(self.classes_)
        tpi = 1 if k == 2 else k
        prep, Xt, keep = self._prep_select(X, y, max_feats)
        cnt = np.unique(y, return_counts=True)[1].min()
        if use_es:
            max_iter = int(min(MAX_ITER, max(40, self._budget(prep, keep, k) // (HGB_BYTES_PER_TREE * tpi))))
            lr = float(np.clip(LR_X_ITERS / max_iter, 0.02, 0.1))
            hgb = self._hgb(lr, max_iter, len(y), True, int(cnt))
        else:
            iters = int(min(SMALL_ITERS, max(40, self._budget(prep, keep, k) // (HGB_BYTES_PER_TREE * tpi))))
            lr = SMALL_LR_BIN if k == 2 else SMALL_LR_MULTI
            hgb = self._hgb(lr, iters, len(y), False, int(cnt))
        bw = balance_weights(y, self.classes_)
        hgb.fit(Xt, y, sample_weight=bw)
        lin = LinearBranch().fit(Xt, y, set(prep.low_), set(prep.high_))
        et = TreeBranch().fit(Xt, y, k, sample_weight=bw) if with_et else None
        return dict(prep=prep, keep=keep, hgb=hgb, lin=lin, et=et, lr=lr)

    def _predict_bases(self, b, X):
        Xt = b["prep"].transform(pd.DataFrame(X))[b["keep"]]
        out = [aligned(b["hgb"].predict_proba(Xt), b["hgb"].classes_, self.classes_),
               aligned(b["lin"].predict_proba(Xt), b["lin"].classes_, self.classes_)]
        if b["et"] is not None:
            out.append(aligned(b["et"].predict_proba(Xt), b["et"].classes_, self.classes_))
        return out

    def _oof(self, X, y, max_feats, use_es, with_et):
        """Predicciones fuera de muestra para el meta-modelo."""
        n = len(y)
        counts = np.unique(y, return_counts=True)[1]
        if use_es and n >= HOLDOUT_MIN_N:
            ia, iv = train_test_split(np.arange(n), test_size=VAL_FRAC, stratify=y, random_state=SEED)
            folds = [(ia, iv)]
        else:
            K = OOF_FOLDS_SMALL if n * X.shape[1] <= 2_000_000 and n < 5_000 else OOF_FOLDS_MED
            if counts.min() >= K:
                folds = list(StratifiedKFold(K, shuffle=True, random_state=SEED + 1).split(X, y))
            else:
                folds = list(KFold(K, shuffle=True, random_state=SEED + 1).split(X))
        idx, preds, iters = [], None, []
        for tr, va in folds:
            fold_es = use_es and np.unique(y[tr], return_counts=True)[1].min() >= ES_MIN_CLASS and len(tr) >= ES_MIN_N
            b = self._fit_bases(X.iloc[tr], y[tr], max_feats, fold_es, with_et)
            P = self._predict_bases(b, X.iloc[va])
            preds = P if preds is None else [np.vstack([a, c]) for a, c in zip(preds, P)]
            idx.append(va)
        idx = np.concatenate(idx)
        return y[idx], preds

    def _learn_mix(self, y_oof, P_oof, mode):
        nb = len(P_oof)
        prior = np.array([W_HGB, W_LR, W_ET][:nb])
        prior = prior / prior.sum()
        # B: rejilla en el símplex de pesos
        steps = np.round(np.arange(0, 1 + 1e-9, GRID_STEP), 6)
        grid = [np.array(w) for w in (
            [(a, 1 - a) for a in steps] if nb == 2 else
            [(a, b_, 1 - a - b_) for a in steps for b_ in steps if a + b_ <= 1 + 1e-9])]
        scores = np.array([auc_score(y_oof, geo_blend(P_oof, np.maximum(w, 0)), self.classes_) for w in grid])
        best = scores.max()
        cand = [i for i in range(len(grid)) if scores[i] >= best - GRID_TOL]
        i_best = min(cand, key=lambda i: np.abs(grid[i] - prior).sum())
        self.w_ = np.maximum(grid[i_best], 0)
        self.info_ = {"w": [round(float(v), 2) for v in self.w_], "oof_auc_w": round(float(scores[i_best]), 4),
                      "oof_auc_prior": round(float(auc_score(y_oof, geo_blend(P_oof, prior), self.classes_)), 4),
                      "oof_n": int(len(y_oof))}
        self.meta_ = None
        if mode.startswith("C") or mode.startswith("S"):
            Z = meta_features(P_oof)
            cls, cnt = np.unique(y_oof, return_counts=True)
            if len(cls) == len(self.classes_):
                meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
                if mode.startswith("C"):
                    self.meta_ = meta.fit(Z, y_oof)
                elif cnt.min() >= 5:
                    # S: el meta-modelo sólo se usa si su AUC con validación cruzada sobre las
                    # predicciones OOF supera a la mejor mezcla de la rejilla.
                    cvm = StratifiedKFold(5, shuffle=True, random_state=SEED + 2)
                    Pm = np.zeros((len(y_oof), len(self.classes_)))
                    for tr, va in cvm.split(Z, y_oof):
                        mm = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED).fit(Z[tr], y_oof[tr])
                        Pm[va] = aligned(mm.predict_proba(Z[va]), mm.classes_, self.classes_)
                    a_meta = float(auc_score(y_oof, Pm, self.classes_))
                    self.info_["oof_auc_meta_cv"] = round(a_meta, 4)
                    if a_meta > scores[i_best] + GRID_TOL:
                        self.meta_ = meta.fit(Z, y_oof)
            self.info_["uses_meta"] = self.meta_ is not None

    # ---------- fit principal ----------
    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        mode = self.mode or MODE
        self.mode_ = mode
        self.classes_, counts = np.unique(y, return_counts=True)
        n, k = len(y), len(self.classes_)
        tpi = 1 if k == 2 else k
        self.min_class_cnt_ = int(counts.min())
        max_feats = int(max(15, min(MAX_FEATS, WORK_BUDGET // (n * tpi))))
        use_es = n >= ES_MIN_N and counts.min() >= ES_MIN_CLASS
        with_et = "D" in mode
        self.info_ = {}
        self.w_ = np.array([W_HGB, W_LR] + ([W_ET] if with_et else []))
        self.meta_ = None

        # Con muchos datos (o muy anchos) basta el 15 % de holdout que la receta ya separa
        # para el early stopping: se reutilizan los modelos entrenados en A (casi sin costo).
        holdout_mix = mode != "A" and use_es and (n >= HOLDOUT_MIN_N or n * X.shape[1] > HOLDOUT_MIN_CELLS)
        if mode != "A" and not holdout_mix:
            y_oof, P_oof = self._oof(X, y, max_feats, use_es, with_et)
            self._learn_mix(y_oof, P_oof, mode)

        # ----- modelos finales: idénticos a la receta original -----
        if not use_es:
            prep, Xt, keep = self._prep_select(X, y, max_feats)
            iters = int(min(SMALL_ITERS, max(40, self._budget(prep, keep, k) // (HGB_BYTES_PER_TREE * tpi))))
            lr = SMALL_LR_BIN if k == 2 else SMALL_LR_MULTI
            bw = balance_weights(y, self.classes_)
            self.hgb_ = self._hgb(lr, iters, n, False, self.min_class_cnt_).fit(Xt, y, sample_weight=bw)
            self.lr_ = LinearBranch().fit(Xt, y, set(prep.low_), set(prep.high_))
            self.et_ = TreeBranch().fit(Xt, y, k, sample_weight=bw) if with_et else None
            self.prep_, self.keep_ = prep, keep
            return self

        ia, iv = train_test_split(np.arange(n), test_size=VAL_FRAC, stratify=y, random_state=SEED)
        XA, yA = X.iloc[ia], y[ia]
        prep_a, XtA, keep = self._prep_select(XA, yA, max_feats)
        max_iter = int(min(MAX_ITER, max(40, self._budget(prep_a, keep, k) // (HGB_BYTES_PER_TREE * tpi))))
        lr = float(np.clip(LR_X_ITERS / max_iter, 0.02, 0.1))
        wA = balance_weights(yA, self.classes_)
        es_model = self._hgb(lr, max_iter, len(yA), True, self.min_class_cnt_)
        es_model.fit(XtA, yA, sample_weight=wA)
        es_iter = int(getattr(es_model, "n_iter_", max_iter))
        if holdout_mix:
            # V nunca se usó para ajustar prep_a, selección, target encoder ni early stopping
            XV, yV = X.iloc[iv], y[iv]
            XtV = prep_a.transform(XV)[keep]
            lin_a = LinearBranch().fit(XtA, yA, set(prep_a.low_), set(prep_a.high_))
            P_v = [aligned(es_model.predict_proba(XtV), es_model.classes_, self.classes_),
                   aligned(lin_a.predict_proba(XtV), lin_a.classes_, self.classes_)]
            if with_et:
                et_a = TreeBranch().fit(XtA, yA, k, sample_weight=wA)
                P_v.append(aligned(et_a.predict_proba(XtV), et_a.classes_, self.classes_))
            self._learn_mix(yV, P_v, mode)

        prep_f = Prep(te_cols=list(prep_a.te_), classes=self.classes_, cols=prep_a.columns_,
                      high_cols=set(prep_a.high_))
        XtF = prep_f.fit_transform_train(X, y)[keep]
        self._prune(prep_f, keep)
        final_iter = int(min(max_iter, max(20, round(es_iter * REFIT_MULT))))
        bw = balance_weights(y, self.classes_)
        self.hgb_ = self._hgb(lr, final_iter, n, False, self.min_class_cnt_).fit(XtF, y, sample_weight=bw)
        self.lr_ = LinearBranch().fit(XtF, y, set(prep_f.low_), set(prep_f.high_))
        self.et_ = TreeBranch().fit(XtF, y, k, sample_weight=bw) if with_et else None
        self.prep_, self.keep_ = prep_f, keep
        self.info_["iters"] = final_iter
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
            learning_rate=lr, max_iter=int(iters), max_leaf_nodes=31, min_samples_leaf=leaf_min,
            l2_regularization=HGB_L2, max_features=0.75, categorical_features="from_dtype",
            early_stopping=early_stopping, validation_fraction=VAL_FRAC if early_stopping else None,
            n_iter_no_change=PATIENCE, random_state=SEED,
        )

    def _base_probas(self, X):
        Xt = self.prep_.transform(pd.DataFrame(X))[self.keep_]
        out = [aligned(self.hgb_.predict_proba(Xt), self.hgb_.classes_, self.classes_),
               aligned(self.lr_.predict_proba(Xt), self.lr_.classes_, self.classes_)]
        if self.et_ is not None:
            out.append(aligned(self.et_.predict_proba(Xt), self.et_.classes_, self.classes_))
        return out

    def branch_probas(self, X):
        Ps = self._base_probas(X)
        return dict(zip(["hgb", "lr", "et"], Ps))

    def predict_proba(self, X):
        Ps = self._base_probas(X)
        if self.meta_ is not None:
            P = aligned(self.meta_.predict_proba(meta_features(Ps)), self.meta_.classes_, self.classes_)
        else:
            P = geo_blend(Ps, self.w_)
        P = np.clip(P, 0, None)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model():
    return make_pipeline(FinalModel())
