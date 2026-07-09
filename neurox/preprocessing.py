"""
neurox.preprocessing
--------------------
Utilidades para preparar datos antes de entrenar.

Funciones disponibles:
    - normalize(X)               → estandariza columnas numéricas (media 0, std 1)
    - one_hot(y, n_classes)      → convierte vector de clases a matriz one-hot
    - encode_labels(y)           → convierte etiquetas string/int a índices 0..N
    - train_test_split(X, y)     → divide datos en train y test
    - to_numpy(df, cols)         → extrae columnas de un DataFrame a numpy

Ejemplo:
    from neurox.preprocessing import normalize, one_hot, train_test_split

    X_norm          = normalize(X)
    y_ohe           = one_hot(y, n_classes=4)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, seed=42)
"""

import numpy as np


# ─── Normalización ───────────────────────────────────────────────────────────

class Normalizer:
    """
    Estandariza columnas: (X - media) / std.
    Guarda los parámetros del fit para aplicar el mismo transform al test set.

    Uso:
        norm = Normalizer()
        X_train = norm.fit_transform(X_train)
        X_test  = norm.transform(X_test)
    """

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray) -> 'Normalizer':
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1  # evita división por cero en columnas constantes
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Llama a fit() antes de transform().")
        return (np.array(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Versión rápida sin guardar parámetros (útil cuando train y test
    se normalizan juntos o solo hay un split).
    """
    return Normalizer().fit_transform(X)


# ─── One-hot encoding ────────────────────────────────────────────────────────

def one_hot(y: np.ndarray, n_classes: int = None) -> np.ndarray:
    """
    Convierte un vector de clases enteras a matriz one-hot.

    Parámetros
    ----------
    y         : np.ndarray de enteros, shape (m,)
    n_classes : int — número de clases. Si es None, se infiere de max(y)+1.

    Retorna
    -------
    np.ndarray, shape (m, n_classes)

    Ejemplo
    -------
    y = np.array([0, 2, 1, 0])
    one_hot(y, n_classes=3)
    # [[1,0,0],[0,0,1],[0,1,0],[1,0,0]]
    """
    y = np.array(y, dtype=int)
    if n_classes is None:
        n_classes = y.max() + 1
    return np.eye(n_classes)[y]


# ─── Label encoding ──────────────────────────────────────────────────────────

class LabelEncoder:
    """
    Convierte etiquetas (strings o cualquier tipo) a enteros 0..N-1
    y permite invertir la transformación.

    Uso:
        le = LabelEncoder()
        y_int = le.fit_transform(y_string)
        y_orig = le.inverse_transform(y_int)
    """

    def __init__(self):
        self.classes_  = None   # array de clases únicas ordenadas
        self._mapping  = {}     # clase → índice
        self._inverse  = {}     # índice → clase

    def fit(self, y) -> 'LabelEncoder':
        self.classes_ = np.array(sorted(set(y)))
        self._mapping  = {c: i for i, c in enumerate(self.classes_)}
        self._inverse  = {i: c for c, i in self._mapping.items()}
        return self

    def transform(self, y) -> np.ndarray:
        if not self._mapping:
            raise RuntimeError("Llama a fit() antes de transform().")
        return np.array([self._mapping[v] for v in y], dtype=int)

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_int) -> np.ndarray:
        return np.array([self._inverse[i] for i in y_int])

    @property
    def n_classes(self):
        return len(self.classes_) if self.classes_ is not None else 0


# ─── Train / Test split ──────────────────────────────────────────────────────

def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.1,
                     seed: int = None):
    """
    Divide X e y en conjuntos de train y test de forma aleatoria.

    Parámetros
    ----------
    X         : np.ndarray, shape (m, features)
    y         : np.ndarray, shape (m,) o (m, clases)
    test_size : float — fracción del dataset para test (0 < test_size < 1)
    seed      : int|None — semilla para reproducibilidad

    Retorna
    -------
    X_train, X_test, y_train, y_test
    """
    X, y = np.array(X), np.array(y)
    m = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(m)
    n_test = int(m * test_size)
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ─── K-Fold (validación cruzada) ─────────────────────────────────────────────

def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 3, seed: int = None):
    """
    Genera k particiones (train, val) para validación cruzada.

    Parámetros
    ----------
    X, y : np.ndarray
    k    : int — número de folds (ej. 3 para 3-Fold CV)
    seed : int|None

    Retorna (yield)
    ----------------
    (X_train, X_val, y_train, y_val) — uno por cada fold

    Ejemplo
    -------
    for X_tr, X_val, y_tr, y_val in k_fold_split(X, y, k=3, seed=42):
        model = Network(...)
        model.train(X_tr, y_tr, ...)
        result = model.evaluate(X_val, y_val)
    """
    X, y = np.array(X), np.array(y)
    m = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(m)
    fold_sizes = np.full(k, m // k, dtype=int)
    fold_sizes[: m % k] += 1

    current = 0
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        val_idx = idx[start:end]
        train_idx = np.concatenate([idx[:start], idx[end:]])
        yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]
        current = end


# ─── DataFrame → numpy ───────────────────────────────────────────────────────

def to_numpy(df, cols: list = None) -> np.ndarray:
    """
    Extrae columnas de un DataFrame pandas y retorna un array numpy float64.
    Si cols es None, extrae todas las columnas.

    Parámetros
    ----------
    df   : pandas.DataFrame
    cols : list de str | None

    Retorna
    -------
    np.ndarray, shape (m, len(cols))
    """
    if cols is not None:
        df = df[cols]
    return df.to_numpy(dtype=float)