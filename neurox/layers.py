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
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 optimizer: str = 'sgd') -> np.ndarray:
        """
        Pasada hacia atrás: calcula gradientes, actualiza pesos y retorna
        el gradiente respecto a la entrada (para propagarlo a la capa anterior).
        """
        # Gradiente respecto a z (pre-activación)
        delta = grad_output * self.fn_prime(self._z)   # (m, n_neurons)

        # Gradientes de pesos y bias
        grad_W = self._input.T @ delta                 # (n_inputs, n_neurons)
        grad_b = delta.sum(axis=0)                     # (n_neurons,)

        # Regularización L2 (weight decay) sobre W
        if weight_decay:
            grad_W = grad_W + weight_decay * self.W

        # Gradiente hacia la capa anterior (se calcula ANTES de actualizar W)
        grad_input = delta @ self.W.T                  # (m, n_inputs)

        # Actualización de pesos
        if optimizer == 'adam':
            # Inicializar acumuladores de Adam de forma perezosa
            if not hasattr(self, '_mW'):
                self._mW = np.zeros_like(self.W)
                self._vW_adam = np.zeros_like(self.W)
                self._mb = np.zeros_like(self.b)
                self._vb_adam = np.zeros_like(self.b)
                self._t = 0
            
            self._t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            # Adam para W
            self._mW = beta1 * self._mW + (1 - beta1) * grad_W
            self._vW_adam = beta2 * self._vW_adam + (1 - beta2) * (grad_W ** 2)
            m_hat_W = self._mW / (1 - beta1 ** self._t)
            v_hat_W = self._vW_adam / (1 - beta2 ** self._t)
            self.W -= alpha * m_hat_W / (np.sqrt(v_hat_W) + eps)
            
            # Adam para b
            self._mb = beta1 * self._mb + (1 - beta1) * grad_b
            self._vb_adam = beta2 * self._vb_adam + (1 - beta2) * (grad_b ** 2)
            m_hat_b = self._mb / (1 - beta1 ** self._t)
            v_hat_b = self._vb_adam / (1 - beta2 ** self._t)
            self.b -= alpha * m_hat_b / (np.sqrt(v_hat_b) + eps)
            
        elif momentum:
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