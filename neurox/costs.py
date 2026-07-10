import numpy as np

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:

    return 2 * (y_pred - y_true) / y_true.shape[0]

def binary_crossentropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:

    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])

def categorical_crossentropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_crossentropy_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:

    return (y_pred - y_true) / y_true.shape[0]

def weighted_categorical_crossentropy(class_weights):

    cw = np.array(class_weights, dtype=float)

    def cost_fn(y_pred, y_true):
        eps = 1e-12
        y_pred_c = np.clip(y_pred, eps, 1 - eps)
        pesos_muestra = y_true @ cw
        perdidas = -np.sum(y_true * np.log(y_pred_c), axis=1)
        return np.mean(pesos_muestra * perdidas)

    def cost_grad(y_pred, y_true):
        pesos_muestra = (y_true @ cw).reshape(-1, 1)
        return pesos_muestra * (y_pred - y_true) / y_true.shape[0]

    return cost_fn, cost_grad

def get_focal_loss(gamma=2.0, alpha_vector=None):

    if alpha_vector is not None:
        av = np.array(alpha_vector, dtype=float)
    else:
        av = None

    def cost_fn(y_pred, y_true):

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0)
        pt = np.sum(y_true * y_pred, axis=1, keepdims=True)

        if av is not None:
            sample_alpha = np.sum(y_true * av, axis=1, keepdims=True)
        else:
            sample_alpha = 1.0

        loss_vector = -sample_alpha * ((1 - pt) ** gamma) * np.log(pt)
        return np.mean(loss_vector)

    def cost_grad(y_pred, y_true):

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0)
        pt = np.sum(y_true * y_pred, axis=1, keepdims=True)

        if av is not None:
            sample_alpha = np.sum(y_true * av, axis=1, keepdims=True)
        else:
            sample_alpha = 1.0

        D = gamma * pt * np.log(pt) - (1 - pt)
        term1 = (1 - pt) ** (gamma - 1)
        bracket = y_true * (1 - pt) - y_pred + y_true * y_pred

        return (sample_alpha * term1 * D * bracket) / y_true.shape[0]

    return cost_fn, cost_grad

_COSTS = {
    'mse':                      (mse,                    mse_grad),
    'binary_crossentropy':      (binary_crossentropy,    binary_crossentropy_grad),
    'categorical_crossentropy': (categorical_crossentropy, categorical_crossentropy_grad),
}

def get_cost(name: str):

    name = name.lower()
    if name not in _COSTS:
        raise ValueError(
            f"Costo '{name}' no reconocido. "
            f"Opciones disponibles: {list(_COSTS.keys())}"
        )
    return _COSTS[name]

def list_costs():

    return list(_COSTS.keys())
