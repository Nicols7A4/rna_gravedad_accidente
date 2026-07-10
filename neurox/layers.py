import numpy as np
from neurox.activations import get_activation

class Dense:

    def __init__(self, n_inputs: int, n_neurons: int,
                 activation: str = 'relu', seed: int = None):

        self.n_inputs  = n_inputs
        self.n_neurons = n_neurons
        self.activation_name = activation

        self.fn, self.fn_prime = get_activation(activation)

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2 / n_inputs) if activation == 'relu' else np.sqrt(1 / n_inputs)

        self.W = rng.normal(0, scale, size=(n_inputs, n_neurons))
        self.b = np.zeros(n_neurons)

        self._input = None
        self._z     = None

        self._vW = np.zeros_like(self.W)
        self._vb = np.zeros_like(self.b)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:

        self._input = X
        self._z = X @ self.W + self.b
        return self.fn(self._z)

    def backward(self, grad_output: np.ndarray, alpha: float,
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 optimizer: str = 'sgd') -> np.ndarray:

        delta = grad_output * self.fn_prime(self._z)

        grad_W = self._input.T @ delta
        grad_b = delta.sum(axis=0)

        if weight_decay:
            grad_W = grad_W + weight_decay * self.W

        grad_input = delta @ self.W.T

        if optimizer == 'adam':
            if not hasattr(self, '_mW'):
                self._mW = np.zeros_like(self.W)
                self._vW_adam = np.zeros_like(self.W)
                self._mb = np.zeros_like(self.b)
                self._vb_adam = np.zeros_like(self.b)
                self._t = 0

            self._t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            self._mW = beta1 * self._mW + (1 - beta1) * grad_W
            self._vW_adam = beta2 * self._vW_adam + (1 - beta2) * (grad_W ** 2)
            m_hat_W = self._mW / (1 - beta1 ** self._t)
            v_hat_W = self._vW_adam / (1 - beta2 ** self._t)
            self.W -= alpha * m_hat_W / (np.sqrt(v_hat_W) + eps)

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

    def __repr__(self):
        return (f"Dense(n_inputs={self.n_inputs}, "
                f"n_neurons={self.n_neurons}, "
                f"activation='{self.activation_name}')")

    @property
    def n_params(self):

        return self.W.size + self.b.size

class Dropout:

    def __init__(self, rate: float, seed: int = None):

        self.rate = rate
        self.seed = seed
        self.mask = None
        self.n_inputs = 0
        self.n_neurons = 0
        self.n_params = 0
        self.activation_name = "dropout"
        self._rng = np.random.default_rng(seed)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:

        if training and self.rate > 0:
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
