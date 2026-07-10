import numpy as np

class Normalizer:

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray) -> 'Normalizer':
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1  
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Llama a fit() antes de transform().")
        return (np.array(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

def normalize(X: np.ndarray) -> np.ndarray:

    return Normalizer().fit_transform(X)

def one_hot(y: np.ndarray, n_classes: int = None) -> np.ndarray:

    y = np.array(y, dtype=int)
    if n_classes is None:
        n_classes = y.max() + 1
    return np.eye(n_classes)[y]

class LabelEncoder:

    def __init__(self):
        self.classes_  = None  
        self._mapping  = {}    
        self._inverse  = {}

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

def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.1,
                     seed: int = None):

    X, y = np.array(X), np.array(y)
    m = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(m)
    n_test = int(m * test_size)
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 3, seed: int = None):

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

def to_numpy(df, cols: list = None) -> np.ndarray:

    if cols is not None:
        df = df[cols]
    return df.to_numpy(dtype=float)
