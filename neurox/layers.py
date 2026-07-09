"""
neurox.layers
-------------
Capa densa (fully connected) con forward y backward pass.

Uso:
    from neurox.layers import Dense

    capa = Dense(n_inputs=6, n_neurons=10, activation='relu')
    output = capa.forward(X)
    grad_input = capa.backward(grad_output, alpha)
"""

import numpy as np
from neurox.activations import get_activation


class Dense:
    """
    Capa totalmente conectada (fully connected / dense).

    Parámetros
    ----------
    n_inputs  : int   — número de entradas (features de la capa anterior)
    n_neurons : int   — número de neuronas en esta capa
    activation: str   — función de activación: 'relu', 'sigmoid', 'tanh',
                        'softmax', 'linear'
    seed      : int|None — semilla para reproducibilidad de los pesos iniciales
    """

    def __init__(self, n_inputs: int, n_neurons: int,
                 activation: str = 'relu', seed: int = None):

        self.n_inputs  = n_inputs
        self.n_neurons = n_neurons
        self.activation_name = activation

        self.fn, self.fn_prime = get_activation(activation)

        # ── Inicialización de pesos ──────────────────────────────────────────
        # He init para ReLU (escala √(2/n)), Xavier para el resto (escala √(1/n))
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2 / n_inputs) if activation == 'relu' else np.sqrt(1 / n_inputs)

        self.W = rng.normal(0, scale, size=(n_inputs, n_neurons))  # (n_inputs, n_neurons)
        self.b = np.zeros(n_neurons)                               # (n_neurons,)

        # ── Cache para backward ──────────────────────────────────────────────
        self._input = None   # X guardado en forward
        self._z     = None   # z = X @ W + b (pre-activación)

        # ── Buffers de momentum (velocity) ───────────────────────────────────
        # Se usan solo si se entrena con momentum > 0. Empiezan en cero.
        self._vW = np.zeros_like(self.W)
        self._vb = np.zeros_like(self.b)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Pasada hacia adelante.

        Parámetros
        ----------
        X : np.ndarray, shape (m, n_inputs)   [m = ejemplos del batch]

        Retorna
        -------
        np.ndarray, shape (m, n_neurons)
        """
        self._input = X
        self._z = X @ self.W + self.b      # (m, n_neurons)
        return self.fn(self._z)            # activación

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self, grad_output: np.ndarray, alpha: float,
                 momentum: float = 0.0, weight_decay: float = 0.0) -> np.ndarray:
        """
        Pasada hacia atrás: calcula gradientes, actualiza pesos y retorna
        el gradiente respecto a la entrada (para propagarlo a la capa anterior).

        Parámetros
        ----------
        grad_output : np.ndarray, shape (m, n_neurons)
            Gradiente que llega desde la capa siguiente (∂L/∂output).
        alpha : float
            Tasa de aprendizaje.
        momentum : float
            Coeficiente de momentum clásico (0 = SGD puro, sin momentum).
            v = momentum * v - alpha * grad ; W += v
        weight_decay : float
            Coeficiente de regularización L2 (solo se aplica a W, no a b).
            grad_W += weight_decay * W antes de la actualización.
            OJO: valores típicos son 1e-4 a 1e-2. Un valor como 0.9
            no es "weight decay" en el sentido usual (destruiría los pesos
            en cada paso); si se necesita ese comportamiento hay que
            pedirlo explícitamente.

        Retorna
        -------
        np.ndarray, shape (m, n_inputs)
            Gradiente respecto a la entrada de esta capa (∂L/∂X).
        """
        # Gradiente respecto a z (pre-activación)
        # Excepción: softmax+CCE ya viene simplificado desde el costo,
        # así que su fn_prime devuelve unos y no modifica grad_output.
        delta = grad_output * self.fn_prime(self._z)   # (m, n_neurons)

        # Gradientes de pesos y bias
        grad_W = self._input.T @ delta                 # (n_inputs, n_neurons)
        grad_b = delta.sum(axis=0)                     # (n_neurons,)

        # Regularización L2 (weight decay) sobre W
        if weight_decay:
            grad_W = grad_W + weight_decay * self.W

        # Gradiente hacia la capa anterior (se calcula ANTES de actualizar W,
        # usando los pesos "viejos", como corresponde en backprop)
        grad_input = delta @ self.W.T                  # (m, n_inputs)

        # Actualización de pesos
        if momentum:
            self._vW = momentum * self._vW - alpha * grad_W
            self._vb = momentum * self._vb - alpha * grad_b
            self.W += self._vW
            self.b += self._vb
        else:
            self.W -= alpha * grad_W
            self.b -= alpha * grad_b

        return grad_input

    # ── Info ─────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (f"Dense(n_inputs={self.n_inputs}, "
                f"n_neurons={self.n_neurons}, "
                f"activation='{self.activation_name}')")

    @property
    def n_params(self):
        """Total de parámetros entrenables en esta capa."""
        return self.W.size + self.b.size


class Dropout:
    """
    Capa de Dropout para regularización.
    Desactiva aleatoriamente una fracción de entradas durante el entrenamiento.
    """

    def __init__(self, rate: float, seed: int = None):
        self.rate = rate
        self.seed = seed
        self.mask = None
        self.n_inputs = 0   # Se autodefine o no aplica
        self.n_neurons = 0  # Se autodefine o no aplica
        self.n_params = 0   # No tiene parámetros entrenables
        self.activation_name = "dropout"
        self._rng = np.random.default_rng(seed)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        if training and self.rate > 0:
            # Generar máscara con escala 1 / (1 - rate) para mantener la esperanza matemática
            self.mask = (self._rng.random(X.shape) >= self.rate) / (1.0 - self.rate)
            return X * self.mask
        else:
            self.mask = None
            return X

    def backward(self, grad_output: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self.mask is not None:
            return grad_output * self.mask
        return grad_output

    def __repr__(self):
        return f"Dropout(rate={self.rate})"