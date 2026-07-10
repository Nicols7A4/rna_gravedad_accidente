"""
neurox.costs
============
Módulo que contiene las funciones de costo (pérdida) y sus gradientes.
Implementa el soporte para aprendizaje sensible al costo (Cost-Sensitive Learning)
mediante entropía cruzada ponderada y Focal Loss multiclase.

Uso:
    from neurox.costs import get_cost

    cost_fn, cost_grad = get_cost('mse')
    loss = cost_fn(y_pred, y_true)      # valor escalar
    grad = cost_grad(y_pred, y_true)    # gradiente ∂L/∂y_pred
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. ERROR CUADRÁTICO MEDIO (MSE)
# ─────────────────────────────────────────────────────────────────────────────
# Error Cuadrático Medio. Bueno para regresión o clasificación binaria simple.


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calcula el Error Cuadrático Medio (Mean Squared Error)."""
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Derivada analítica del MSE respecto a las predicciones (y_pred)."""
    return 2 * (y_pred - y_true) / y_true.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENTROPÍA CRUZADA BINARIA (BINARY CROSS-ENTROPY)
# ─────────────────────────────────────────────────────────────────────────────
# Para clasificación binaria (salida sigmoide, y_true ∈ {0, 1}).


def binary_crossentropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calcula la pérdida de Entropía Cruzada Binaria (salida sigmoide)."""
    eps = 1e-12  # Previene indeterminación matemática de log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Derivada analítica de BCE respecto a y_pred."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENTROPÍA CRUZADA CATEGÓRICA (CATEGORICAL CROSS-ENTROPY)
# ─────────────────────────────────────────────────────────────────────────────
# Para clasificación multiclase (salida softmax, y_true one-hot).
# El gradiente softmax + CCE se simplifica a (y_pred - y_true),
# por eso se devuelve directamente.

def categorical_crossentropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calcula la pérdida de Entropía Cruzada Categórica para multiclase (salida Softmax)."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_crossentropy_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradiente combinado analítico de Softmax + Categorical Crossentropy.
    Matemáticamente, la derivada ∂L/∂z simplifica directamente a: (y_pred - y_true).
    """
    return (y_pred - y_true) / y_true.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTROPÍA CRUZADA CATEGÓRICA PONDERADA (WEIGHTED CCE)
# ─────────────────────────────────────────────────────────────────────────────

def weighted_categorical_crossentropy(class_weights):
    """
    Genera funciones de coste y gradiente para CCE ponderada por clase.
    Fórmula: L = -mean( sum( w_c * y_true_c * log(y_pred_c) ) )
    """
    cw = np.array(class_weights, dtype=float)

    def cost_fn(y_pred, y_true):
        eps = 1e-12
        y_pred_c = np.clip(y_pred, eps, 1 - eps)
        # Multiplicar los vectores target por los pesos de clase
        pesos_muestra = y_true @ cw                              # [m,]
        perdidas = -np.sum(y_true * np.log(y_pred_c), axis=1)     # [m,]
        return np.mean(pesos_muestra * perdidas)

    def cost_grad(y_pred, y_true):
        pesos_muestra = (y_true @ cw).reshape(-1, 1)              # [m, 1]
        return pesos_muestra * (y_pred - y_true) / y_true.shape[0]

    return cost_fn, cost_grad


# ─────────────────────────────────────────────────────────────────────────────
# 5. FOCAL LOSS MULTICLASE (Sustento Científico: Wang et al., 2026)
# ─────────────────────────────────────────────────────────────────────────────

def get_focal_loss(gamma=2.0, alpha_vector=None):
    """
    Construye las clausuras (closures) para Focal Loss con factor modulador.
    Fórmula matemática: L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Donde:
        - p_t es la probabilidad Softmax predicha para la clase real.
        - gamma es el parámetro de enfoque (reproduce pérdida estándar si gamma=0).
    """
    if alpha_vector is not None:
        av = np.array(alpha_vector, dtype=float)
    else:
        av = None

    def cost_fn(y_pred, y_true):
        """Cálculo escalar de la Focal Loss del Batch."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0)
        # Extraer la probabilidad de la clase correcta (p_t)
        pt = np.sum(y_true * y_pred, axis=1, keepdims=True)
        
        # Ponderación alfa sensible al costo o frecuencia
        if av is not None:
            sample_alpha = np.sum(y_true * av, axis=1, keepdims=True)
        else:
            sample_alpha = 1.0
            
        loss_vector = -sample_alpha * ((1 - pt) ** gamma) * np.log(pt)
        return np.mean(loss_vector)

    def cost_grad(y_pred, y_true):
        """
        Derivación analítica compleja de la Focal Loss respecto a las predicciones Softmax.
        Calcula dLoss/dy_pred para la propagación hacia atrás.
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0)
        pt = np.sum(y_true * y_pred, axis=1, keepdims=True)
        
        if av is not None:
            sample_alpha = np.sum(y_true * av, axis=1, keepdims=True)
        else:
            sample_alpha = 1.0
            
        D = gamma * pt * np.log(pt) - (1 - pt)
        term1 = (1 - pt) ** (gamma - 1)
        # Factorización de la matriz jacobiana Softmax combinada
        bracket = y_true * (1 - pt) - y_pred + y_true * y_pred
        
        return (sample_alpha * term1 * D * bracket) / y_true.shape[0]

    return cost_fn, cost_grad


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRO GLOBAL DE FUNCIONES DE COSTO
# ─────────────────────────────────────────────────────────────────────────────

_COSTS = {
    'mse':                      (mse,                    mse_grad),
    'binary_crossentropy':      (binary_crossentropy,    binary_crossentropy_grad),
    'categorical_crossentropy': (categorical_crossentropy, categorical_crossentropy_grad),
}

def get_cost(name: str):
    """
    Retorna la tupla (cost_fn, cost_grad) para instanciar en la red.
    
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
    """Retorna los identificadores de pérdidas registrados."""
    return list(_COSTS.keys())