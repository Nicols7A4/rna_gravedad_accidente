"""
neurox.activations
------------------
Funciones de activación y sus derivadas.
Cada función acepta un array numpy y retorna un array numpy.

Uso:
    from neurox.activations import get_activation

    fn, fn_prime = get_activation('relu')
    output = fn(z)          # forward
    grad   = fn_prime(z)    # backward (derivada)
"""

import numpy as np


# ─── Sigmoid ─────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip evita overflow

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


# ─── ReLU ────────────────────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)


# ─── Tanh ────────────────────────────────────────────────────────────────────

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.tanh(z) ** 2


# ─── Softmax ─────────────────────────────────────────────────────────────────
# Para clasificación multiclase (capa de salida).
# Se estabiliza restando el máximo antes de exp (evita overflow).

def softmax(z):
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def softmax_prime(z):
    # En la práctica, la derivada de softmax se combina con cross-entropy
    # y se simplifica a (y_ - y). Retornamos unos para no romper el flujo
    # cuando se usa fuera de ese contexto.
    return np.ones_like(z)


# ─── Lineal (identidad) ──────────────────────────────────────────────────────
# Útil para regresión o capas intermedias de prueba.

def linear(z):
    return z

def linear_prime(z):
    return np.ones_like(z)


# ─── Registro ────────────────────────────────────────────────────────────────

_ACTIVATIONS = {
    'sigmoid':  (sigmoid,  sigmoid_prime),
    'relu':     (relu,     relu_prime),
    'tanh':     (tanh,     tanh_prime),
    'softmax':  (softmax,  softmax_prime),
    'linear':   (linear,   linear_prime),
}

def get_activation(name: str):
    """
    Retorna (fn, fn_prime) para la activación solicitada.

    Parámetros
    ----------
    name : str
        Nombre de la activación: 'sigmoid', 'relu', 'tanh', 'softmax', 'linear'

    Retorna
    -------
    tuple (fn, fn_prime)

    Ejemplo
    -------
    fn, fn_prime = get_activation('relu')
    """
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Activación '{name}' no reconocida. "
            f"Opciones disponibles: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]


def list_activations():
    """Retorna la lista de activaciones disponibles."""
    return list(_ACTIVATIONS.keys())