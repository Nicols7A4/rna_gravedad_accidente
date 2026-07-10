import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def softmax_prime(z):
    return np.ones_like(z)

def linear(z):
    return z

def linear_prime(z):
    return np.ones_like(z)

_ACTIVATIONS = {
    'sigmoid':  (sigmoid,  sigmoid_prime),
    'relu':     (relu,     relu_prime),
    'tanh':     (tanh,     tanh_prime),
    'softmax':  (softmax,  softmax_prime),
    'linear':   (linear,   linear_prime),
}

def get_activation(name: str):

    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Activación '{name}' no reconocida. "
            f"Opciones disponibles: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]

def list_activations():

    return list(_ACTIVATIONS.keys())
