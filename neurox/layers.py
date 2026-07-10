"""
neurox.layers
=============
Módulo que define las capas básicas para la construcción de la red neuronal feedforward.
Implementa:
1. Capa Densa (Dense): Conexiones fully-connected, inicialización Xavier/He y optimizador Adam/SGD.
2. Capa Dropout: Regularización de descarte de nodos en fase de entrenamiento.

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
    Capa Densa completamente conectada (Fully Connected Layer).

    Matemática:
    ----------
    - Paso Forward: A = activation(X @ W + b)
    - Paso Backward: grad_W = X.T @ delta, grad_b = sum(delta)
    """

    def __init__(self, n_inputs: int, n_neurons: int,
                 activation: str = 'relu', seed: int = None):
        """
        Inicializa pesos, sesgos y buffers de momentos.

        Parámetros
        ----------
        n_inputs : int — Número de dimensiones del vector de entrada.
        n_neurons : int — Número de neuronas de salida en esta capa.
        activation : str — Función de activación ('relu', 'sigmoid', 'tanh', 'softmax').
        seed : int — Semilla de generación aleatoria de pesos.
        """
        self.n_inputs  = n_inputs
        self.n_neurons = n_neurons
        self.activation_name = activation

        # Cargar la función de activación y su derivada analítica
        self.fn, self.fn_prime = get_activation(activation)

        # ── Inicialización de Pesos (Weight Initialization) ──────────────────
        # He Normal (para ReLU) o Xavier Normal (para el resto)
        # Escala y previene problemas de gradiente desvanecido/explosivo
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2 / n_inputs) if activation == 'relu' else np.sqrt(1 / n_inputs)

        self.W = rng.normal(0, scale, size=(n_inputs, n_neurons))  # Pesos [n_entradas, n_neuronas]
        self.b = np.zeros(n_neurons)                               # Sesgos [n_neuronas,]

        # ── Cache para Backward Pass ─────────────────────────────────────────
        self._input = None   # Guarda el vector X del paso forward
        self._z     = None   # Guarda la pre-activación: z = X @ W + b

        # ── Buffers de Momentum para SGD ─────────────────────────────────────
        self._vW = np.zeros_like(self.W)
        self._vb = np.zeros_like(self.b)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Paso hacia adelante (Forward propagation).

        Parámetros
        ----------
        X : np.ndarray — Matriz de entradas de forma (m, n_inputs).
        training : bool — Si es True, indica fase de entrenamiento (para Dropout).

        Retorna
        -------
        np.ndarray — Matriz de salidas de forma (m, n_neurons).
        """
        self._input = X
        self._z = X @ self.W + self.b      # Combinación lineal pre-activación
        return self.fn(self._z)            # Aplicación de función de activación

    def backward(self, grad_output: np.ndarray, alpha: float,
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 optimizer: str = 'sgd') -> np.ndarray:
        """
        Paso hacia atrás (Backpropagation).
        Calcula gradientes locales, actualiza pesos/sesgos y propaga el gradiente.

        Parámetros
        ----------
        grad_output : np.ndarray — Gradiente acumulado de la capa siguiente [m, n_neurons].
        alpha : float — Tasa de aprendizaje (learning rate).
        momentum : float — Factor de inercia para SGD (Ignorado en Adam).
        weight_decay : float — Constante de regularización L2.
        optimizer : str — Algoritmo de optimización ('sgd' o 'adam').

        Retorna
        -------
        np.ndarray — Gradiente respecto a la entrada de esta capa [m, n_inputs].
        """
        # delta = dLoss/dz = dLoss/da * da/dz
        delta = grad_output * self.fn_prime(self._z)   # [m, n_neurons]

        # Gradientes de los parámetros de esta capa
        grad_W = self._input.T @ delta                 # [n_inputs, n_neurons]
        grad_b = delta.sum(axis=0)                     # [n_neurons,]

        # Aplicación de Regularización L2 (Weight Decay)
        if weight_decay:
            grad_W = grad_W + weight_decay * self.W

        # Propagación del gradiente hacia atrás (retropropagación)
        grad_input = delta @ self.W.T                  # [m, n_inputs]

        # ── OPTIMIZADOR ADAM ─────────────────────────────────────────────────
        if optimizer == 'adam':
            # Inicialización perezosa de los buffers de momentos
            if not hasattr(self, '_mW'):
                self._mW = np.zeros_like(self.W)       # Primer momento para W (media)
                self._vW_adam = np.zeros_like(self.W)  # Segundo momento para W (varianza)
                self._mb = np.zeros_like(self.b)       # Primer momento para b
                self._vb_adam = np.zeros_like(self.b)  # Segundo momento para b
                self._t = 0                            # Contador de pasos de tiempo (time step)
            
            self._t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            # 1. Adam para Pesos W
            self._mW = beta1 * self._mW + (1 - beta1) * grad_W
            self._vW_adam = beta2 * self._vW_adam + (1 - beta2) * (grad_W ** 2)
            m_hat_W = self._mW / (1 - beta1 ** self._t)  # Corrección de sesgo primer momento
            v_hat_W = self._vW_adam / (1 - beta2 ** self._t)  # Corrección de sesgo segundo momento
            self.W -= alpha * m_hat_W / (np.sqrt(v_hat_W) + eps)
            
            # 2. Adam para Sesgos b
            self._mb = beta1 * self._mb + (1 - beta1) * grad_b
            self._vb_adam = beta2 * self._vb_adam + (1 - beta2) * (grad_b ** 2)
            m_hat_b = self._mb / (1 - beta1 ** self._t)
            v_hat_b = self._vb_adam / (1 - beta2 ** self._t)
            self.b -= alpha * m_hat_b / (np.sqrt(v_hat_b) + eps)
            
        # ── OPTIMIZADOR SGD CON MOMENTUM ─────────────────────────────────────
        elif momentum:
            self._vW = momentum * self._vW - alpha * grad_W
            self._vb = momentum * self._vb - alpha * grad_b
            self.W += self._vW
            self.b += self._vb
            
        # ── OPTIMIZADOR SGD ESTÁNDAR ─────────────────────────────────────────
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
        """Devuelve el total de parámetros entrenables en la capa (pesos + sesgos)."""
        return self.W.size + self.b.size


class Dropout:
    """
    Capa de Dropout para regularización.
    Desactiva aleatoriamente una fracción de activaciones durante el entrenamiento
    para romper la co-adaptación de neuronas y prevenir overfitting.
    """

    def __init__(self, rate: float, seed: int = None):
        """
        Parámetros
        ----------
        rate : float — Probabilidad de descarte (tasa de dropout entre 0 y 1).
        seed : int — Semilla aleatoria.
        """
        self.rate = rate
        self.seed = seed
        self.mask = None
        self.n_inputs = 0   # Nodos virtuales
        self.n_neurons = 0  # Nodos virtuales
        self.n_params = 0   # No tiene parámetros libres entrenables
        self.activation_name = "dropout"
        self._rng = np.random.default_rng(seed)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Paso hacia adelante de Dropout.
        Aplica escala invertida (Inverted Dropout) 1/(1-p) durante el entrenamiento
        para que en el paso de inferencia (test) no sea necesario re-escalar las activaciones.
        """
        if training and self.rate > 0:
            # Creamos la máscara binaria escalada por 1 / (1 - rate)
            self.mask = (self._rng.random(X.shape) >= self.rate) / (1.0 - self.rate)
            return X * self.mask
        else:
            self.mask = None
            return X

    def backward(self, grad_output: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Propaga el gradiente solo por las neuronas que se mantuvieron activas."""
        if self.mask is not None:
            return grad_output * self.mask
        return grad_output

    def __repr__(self):
        return f"Dropout(rate={self.rate})"