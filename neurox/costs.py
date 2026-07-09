"""
neurox.costs
------------
Funciones de costo (pérdida) y sus gradientes respecto a la predicción.

Uso:
    from neurox.costs import get_cost

    cost_fn, cost_grad = get_cost('mse')
    loss = cost_fn(y_pred, y_true)      # valor escalar
    grad = cost_grad(y_pred, y_true)    # gradiente ∂L/∂y_pred
"""

import numpy as np


# ─── MSE ─────────────────────────────────────────────────────────────────────
# Error Cuadrático Medio. Bueno para regresión o clasificación binaria simple.

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]


# ─── Binary Cross-Entropy ─────────────────────────────────────────────────────
# Para clasificación binaria (salida sigmoide, y_true ∈ {0, 1}).

def binary_crossentropy(y_pred, y_true):
    eps = 1e-12  # evita log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_grad(y_pred, y_true):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])


# ─── Categorical Cross-Entropy ────────────────────────────────────────────────
# Para clasificación multiclase (salida softmax, y_true one-hot).
# El gradiente softmax + CCE se simplifica a (y_pred - y_true),
# por eso se devuelve directamente.

def categorical_crossentropy(y_pred, y_true):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_crossentropy_grad(y_pred, y_true):
    # Gradiente combinado softmax + CCE
    return (y_pred - y_true) / y_true.shape[0]


# ─── Categorical Cross-Entropy ponderada por clase ───────────────────────────
# Igual que arriba, pero cada muestra pesa distinto según su clase real.
# Útil cuando quieres que el modelo le preste más atención a ciertas salidas
# (ej. subir el peso de LESIONADO/FALLECIDO si el modelo las confunde entre sí),
# incluso si las clases ya están balanceadas en cantidad de filas.

def weighted_categorical_crossentropy(class_weights):
    """
    Retorna (cost_fn, cost_grad) para categorical cross-entropy ponderada.

    Parámetros
    ----------
    class_weights : list|np.ndarray, shape (n_clases,)
        Peso de cada clase, EN EL MISMO ORDEN que las columnas del one-hot
        de y_true. Un peso > 1 hace que los errores en esa clase pesen más
        en la pérdida (el modelo la prioriza); un peso < 1 la de-prioriza.
        Ej.: si CLASES = ['ILESO', 'LESIONADO', 'FALLECIDO'],
             class_weights=[1.0, 1.5, 1.3]
             sube la importancia de LESIONADO y FALLECIDO frente a ILESO.

    Retorna
    -------
    tuple (cost_fn, cost_grad)

    Ejemplo
    -------
    cost_fn, cost_grad = weighted_categorical_crossentropy([1.0, 1.5, 1.3])
    model = Network(layers=[...], cost='categorical_crossentropy')
    model.cost_fn, model.cost_grad = cost_fn, cost_grad
    # (o, más simple, usa el parámetro class_weights de Network directamente)
    """
    cw = np.array(class_weights, dtype=float)

    def cost_fn(y_pred, y_true):
        eps = 1e-12
        y_pred_c = np.clip(y_pred, eps, 1 - eps)
        pesos_muestra = y_true @ cw                              # (m,)
        perdidas = -np.sum(y_true * np.log(y_pred_c), axis=1)     # (m,)
        return np.mean(pesos_muestra * perdidas)

    def cost_grad(y_pred, y_true):
        pesos_muestra = (y_true @ cw).reshape(-1, 1)              # (m,1)
        return pesos_muestra * (y_pred - y_true) / y_true.shape[0]

    return cost_fn, cost_grad


def get_focal_loss(gamma=2.0, alpha_vector=None):
    """
    Retorna (cost_fn, cost_grad) para la pérdida Focal Loss multiclase.
    """
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


# ─── Registro ────────────────────────────────────────────────────────────────

_COSTS = {
    'mse':                      (mse,                    mse_grad),
    'binary_crossentropy':      (binary_crossentropy,    binary_crossentropy_grad),
    'categorical_crossentropy': (categorical_crossentropy, categorical_crossentropy_grad),
}

def get_cost(name: str):
    """
    Retorna (cost_fn, cost_grad) para la función de costo solicitada.

    Parámetros
    ----------
    name : str
        'mse', 'binary_crossentropy', 'categorical_crossentropy'

    Retorna
    -------
    tuple (cost_fn, cost_grad)

    Ejemplo
    -------
    cost_fn, cost_grad = get_cost('binary_crossentropy')
    loss = cost_fn(y_pred, y_true)
    grad = cost_grad(y_pred, y_true)
    """
    name = name.lower()
    if name not in _COSTS:
        raise ValueError(
            f"Costo '{name}' no reconocido. "
            f"Opciones disponibles: {list(_COSTS.keys())}"
        )
    return _COSTS[name]


def list_costs():
    """Retorna la lista de funciones de costo disponibles."""
    return list(_COSTS.keys())