"""
MLP en NumPy con VOS (Virtual Outlier Synthesis; Du et al., ICLR 2022) para clasificación tabular
(solo NumPy, pandas, SciPy y scikit-learn; Python 3.11, sklearn >= 1.4).

    model = build_model();  model.fit(X, y);  model.predict_proba(X_test)   # columnas en el orden de classes_

`build_model(use_vos=False)` da exactamente la misma red sin el término VOS: es el control del experimento.
`MLPBase` también se puede usar como base extra del stacking (ver bench/variants.py).

Ningún hiperparámetro se eligió mirando datos de evaluación: son los del paper o reglas fijas por tamaño.
"""
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, TargetEncoder

USE_VOS = True
SEED = 0

# ---------------------------------------------------------------- reglas fijas (no ajustadas con datos)
LOW_CARD_MAX = 30            # <= 30 categorías: one-hot / categórica nativa; > 30: target encoding
HC_MAX_CATS_TOTAL = 30_000   # categorías guardadas (sumando columnas de alta cardinalidad)
HC_MAX_CATS_COL = 2_000
OHE_MIN_FREQ, OHE_MAX_CATS = 10, 30
CLIP = 5.0

MLP_BUDGET = 600_000
MLP_MAX_ROWS = 50_000
MLP_MAX_PARAMS = 150_000     # float32 -> ~0.6 MB
MLP_H2 = 64                  # penúltima capa: espacio donde VOS estima gaussianas
MLP_FLOP_BUDGET = 4e11       # tope de cómputo de entrenamiento por ajuste
VOS_BETA = 0.1               # peso de la pérdida de incertidumbre (paper)
VOS_QUEUE = 1000             # tamaño de cola por clase (paper, clasificación)
VOS_SAMPLE_FROM = 10_000     # candidatos por clase; se toma el de menor verosimilitud (select=1)
VOS_START_FRAC = 0.4         # el paper empieza en la época 40 de 100


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


# ================================================================ MLP en NumPy con/sin VOS
class VOSNet:
    """MLP de dos capas ocultas (ReLU) entrenada con Adam, con regularización VOS opcional.

    VOS (Du et al., ICLR 2022, Algoritmo 1), en el espacio de la penúltima capa z:
      1. Cola FIFO por clase con las z (sin gradiente) de los ejemplos de entrenamiento.
      2. Media por clase y covarianza compartida (+1e-4 I, como el código oficial).
      3. Por clase se muestrean VOS_SAMPLE_FROM puntos de N(mu_k, Sigma) y se toma el de menor
         verosimilitud. Como -log N = ||L^-1 (v - mu)||^2 / 2 + cte, el de menor verosimilitud es el de
         mayor radio: se muestrea el radio como el máximo de VOS_SAMPLE_FROM variables chi(D) y una
         dirección uniforme. Es la misma distribución que muestrear y quedarse con el mínimo.
      4. Energía libre con pesos aprendidos: -E(x) = log sum_k relu(u_k) exp(f_k(x)).
      5. Pérdida de incertidumbre: regresión logística sobre -E (ID = 1, virtuales = 0).
         Los logits de los virtuales salen de la última capa aplicada a v.
      6. Pérdida total = CE + beta * L_incertidumbre; VOS se activa a partir de VOS_START_FRAC de
         las épocas (calendario fijo, ver fit) y cuando todas las colas están llenas.
    Desviación documentada: la cola de cada clase mide min(VOS_QUEUE, n_clase) para que las clases
    raras también se llenen.
    """

    def __init__(self, use_vos, seed=SEED):
        self.use_vos = use_vos
        self.seed = seed

    # ------------------------------------------------ forward / backward
    def _forward(self, X):
        p = self.p
        a1 = X @ p["W1"] + p["b1"]
        h1 = np.maximum(a1, 0)
        a2 = h1 @ p["W2"] + p["b2"]
        z = np.maximum(a2, 0)
        f = z @ p["W3"] + p["b3"]
        return a1, h1, a2, z, f

    @staticmethod
    def _softmax(f):
        e = np.exp(f - f.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _neg_energy(self, f):
        w = np.maximum(self.p["u"], 0)
        m = f.max(axis=1, keepdims=True)
        T = (w * np.exp(f - m)).sum(axis=1, keepdims=True) + 1e-12
        S = (m + np.log(T)).ravel()
        return S, np.exp(f - S[:, None])  # S = -E ; dS/dw_k = exp(f_k - S)

    def predict_proba(self, X):
        return self._softmax(self._forward(np.asarray(X, dtype=np.float32))[-1])

    # ------------------------------------------------ entrenamiento
    def _init_params(self, d, K, rng):
        h2 = MLP_H2
        h1 = int(np.clip((MLP_MAX_PARAMS - h2 * (K + 1)) // (d + 1 + h2), 16, 128))
        self.p = {"W1": (rng.normal(0, np.sqrt(2.0 / d), (d, h1))).astype(np.float32), "b1": np.zeros(h1, np.float32),
                  "W2": (rng.normal(0, np.sqrt(2.0 / h1), (h1, h2))).astype(np.float32), "b2": np.zeros(h2, np.float32),
                  "W3": (rng.normal(0, np.sqrt(1.0 / h2), (h2, K))).astype(np.float32), "b3": np.zeros(K, np.float32),
                  "u": rng.uniform(0, 1, K).astype(np.float32),      # pesos de la energía (código oficial: uniforme)
                  "a": np.ones(1, np.float32), "c": np.zeros(1, np.float32)}  # logística sobre -E
        return h1, h2

    def fit(self, X, y, K):
        """Dos pasadas, idénticas para la variante con y sin VOS:
        1. Sin VOS, con 10 % de validación y early stopping: fija el número de épocas E.
        2. Reentrena en todas las filas durante E épocas (calendario fijo, como el paper) y se queda con
           los pesos finales. Si use_vos, VOS se activa desde la época floor(VOS_START_FRAC * E).
        Así la variante VOS nunca termina con pesos de antes de que VOS actúe."""
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self.K = K
        h1, h2 = self._init_params(d, K, np.random.default_rng(self.seed))
        batch = int(np.clip(n // 10, 32, 256))
        flops_epoch = 6.0 * n * (d * h1 + h1 * h2 + h2 * K)
        max_epochs = int(np.clip(min(4000 * batch / n, MLP_FLOP_BUDGET / flops_epoch), 8, 200))
        counts = np.bincount(y, minlength=K)
        if n >= 100 and counts[counts > 0].min() >= 2:
            tr, va = train_test_split(np.arange(n), test_size=0.1, stratify=y, random_state=self.seed)
            best_ep = self._train(X[tr], y[tr], K, max_epochs, False, (X[va], y[va]))
            epochs = max(5, best_ep)
        else:
            epochs = max(5, max_epochs // 2)
        self._init_params(d, K, np.random.default_rng(self.seed + 1))
        self._train(X, y, K, epochs, self.use_vos, None)
        self.epochs_run_ = epochs
        return self

    def _virtual_outliers(self, queues, qcap, ks, rng):
        h2 = MLP_H2
        mus = np.stack([queues[k].mean(axis=0) for k in ks])
        C = sum(((queues[k] - mus[j]).T @ (queues[k] - mus[j])) for j, k in enumerate(ks))
        C = C / sum(qcap[k] for k in ks) + 1e-4 * np.eye(h2, dtype=np.float32)
        L = np.linalg.cholesky(C.astype(np.float64))
        dirs = rng.normal(size=(len(ks), h2))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        # máximo de VOS_SAMPLE_FROM chi^2(D) por CDF inversa: F^-1(U^(1/N))
        radii = np.sqrt(chi2.ppf(rng.random(len(ks)) ** (1.0 / VOS_SAMPLE_FROM), df=h2))
        return (mus + (dirs * radii[:, None]) @ L.T).astype(np.float32)

    def _batch_grads(self, xb, yb, V=None):
        """Pérdida (CE + beta * incertidumbre si hay virtuales V) y gradientes analíticos (sin L2)."""
        B = len(yb)
        a1, h1a, a2, z, f = self._forward(xb)
        P = self._softmax(f)
        loss = -np.mean(np.log(_clip_p(P[np.arange(B), yb])))
        G = P.copy()
        G[np.arange(B), yb] -= 1.0
        G /= B
        g = {}
        gW3_extra = gb3_extra = None
        if V is not None:
            fv = V @ self.p["W3"] + self.p["b3"]
            S_id, E_id = self._neg_energy(f)
            S_v, E_v = self._neg_energy(fv)
            S = np.concatenate([S_id, S_v])
            lab = np.concatenate([np.ones(B), np.zeros(len(V))])
            a, c = self.p["a"][0], self.p["c"][0]
            q = np.clip(a * S + c, -50, 50)
            sig = 1.0 / (1.0 + np.exp(-q))
            loss += VOS_BETA * np.mean(np.logaddexp(0, -q) * lab + np.logaddexp(0, q) * (1 - lab))
            gs = VOS_BETA * (sig - lab) / len(S)
            g["a"] = np.array([np.sum(gs * S)], np.float32)
            g["c"] = np.array([np.sum(gs)], np.float32)
            w = np.maximum(self.p["u"], 0)
            dS = gs * a
            G += (dS[:B, None] * (w * E_id)).astype(np.float32)       # dS/df_k = w_k exp(f_k - S)
            Gv = (dS[B:, None] * (w * E_v)).astype(np.float32)
            gW3_extra, gb3_extra = V.T @ Gv, Gv.sum(axis=0)
            du = (dS[:B, None] * E_id).sum(axis=0) + (dS[B:, None] * E_v).sum(axis=0)
            g["u"] = (du * (self.p["u"] > 0)).astype(np.float32)
        g["W3"] = z.T @ G
        g["b3"] = G.sum(axis=0)
        if gW3_extra is not None:
            g["W3"] += gW3_extra
            g["b3"] += gb3_extra
        dz = (G @ self.p["W3"].T) * (a2 > 0)
        g["W2"], g["b2"] = h1a.T @ dz, dz.sum(axis=0)
        dh = (dz @ self.p["W2"].T) * (a1 > 0)
        g["W1"], g["b1"] = xb.T @ dh, dh.sum(axis=0)
        return loss, g

    def _train(self, X, y, K, epochs, use_vos, val):
        rng = np.random.default_rng(self.seed + 2)
        n = len(y)
        h2 = MLP_H2
        batch = int(np.clip(n // 10, 32, 256))
        start = int(VOS_START_FRAC * epochs)
        patience = max(5, epochs // 10)

        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        s = {k: np.zeros_like(v) for k, v in self.p.items()}
        t, lr, b1, b2, wd = 0, 1e-3, 0.9, 0.999, 1e-4

        qcap = np.minimum(VOS_QUEUE, np.maximum(np.bincount(y, minlength=K), 1))
        queues = [np.zeros((qcap[k], h2), np.float32) for k in range(K)]
        qfill, qpos = np.zeros(K, int), np.zeros(K, int)
        present = np.bincount(y, minlength=K) > 0

        best, best_ep, bad = np.inf, epochs, 0
        self.vos_steps_ = 0
        for ep in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                xb, yb = X[idx], y[idx]
                z = self._forward(xb)[3]
                # --- cola por clase (z sin gradiente)
                if use_vos:
                    for k in np.unique(yb):
                        rows = z[yb == k][-qcap[k]:]
                        pos = (qpos[k] + np.arange(len(rows))) % qcap[k]
                        queues[k][pos] = rows
                        qpos[k] = (qpos[k] + len(rows)) % qcap[k]
                        qfill[k] = min(qfill[k] + len(rows), qcap[k])
                V = None
                if use_vos and ep >= start and np.all(qfill[present] >= qcap[present]):
                    V = self._virtual_outliers(queues, qcap, np.where(present)[0], rng)
                    self.vos_steps_ += 1
                _, g = self._batch_grads(xb, yb, V)
                # --- Adam (+ L2 en las matrices)
                t += 1
                for k, gk in g.items():
                    if k in ("W1", "W2", "W3"):
                        gk = gk + wd * self.p[k]
                    m[k] = b1 * m[k] + (1 - b1) * gk
                    s[k] = b2 * s[k] + (1 - b2) * gk * gk
                    mh, sh = m[k] / (1 - b1 ** t), s[k] / (1 - b2 ** t)
                    self.p[k] = (self.p[k] - lr * mh / (np.sqrt(sh) + 1e-8)).astype(np.float32)
            if val is not None:
                Xv, yv = val
                Pv = self.predict_proba(Xv)
                loss = -np.mean(np.log(_clip_p(Pv[np.arange(len(yv)), yv])))
                if loss < best - 1e-4:
                    best, best_ep, bad = loss, ep + 1, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        return best_ep


class MLPBase(BaseEstimator):
    def __init__(self, use_vos=False):
        self.use_vos = use_vos

    def fit(self, F, y, K, budget=MLP_BUDGET, low=()):
        low = [c for c in F.columns if c in set(low)]
        num = [c for c in F.columns if c not in set(low)]
        idx = np.arange(len(y))
        if len(y) > MLP_MAX_ROWS:
            idx = _stratified_subsample(y, MLP_MAX_ROWS)
        F, y = F.iloc[idx], y[idx]
        self.ct_ = _linear_ct(num, low, dense=True)
        Z = self.ct_.fit_transform(F).astype(np.float32)
        self.net_ = VOSNet(self.use_vos).fit(Z, y, K)
        self.K_ = K
        return self

    def predict_proba(self, F):
        return self.net_.predict_proba(self.ct_.transform(F).astype(np.float32))


def _stratified_subsample(y, m):
    try:
        idx, _ = train_test_split(np.arange(len(y)), train_size=m, stratify=y, random_state=SEED)
    except ValueError:  # clases con un solo ejemplo
        idx = np.random.default_rng(SEED).choice(len(y), m, replace=False)
    return np.sort(idx)



# ================================================================ modelo
class MLPVOSModel(ClassifierMixin, BaseEstimator):
    """Featurizer + MLPBase. use_vos=False es el control idéntico sin VOS."""

    def __init__(self, use_vos=None):
        self.use_vos = use_vos

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
        self.base_ = MLPBase(use_vos=USE_VOS if self.use_vos is None else self.use_vos).fit(F, y, K, low=self.feat_.low_)
        self.info_ = {"use_vos": self.base_.use_vos, "epochs": self.base_.net_.epochs_run_, "vos_steps": self.base_.net_.vos_steps_}
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


def build_model(use_vos=None):
    return MLPVOSModel(use_vos=USE_VOS if use_vos is None else use_vos)
