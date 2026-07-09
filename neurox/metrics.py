"""
neurox.metrics
--------------
Métricas de evaluación para clasificación.

Funciones disponibles:
    - confusion_matrix(y_true, y_pred)
    - accuracy(y_true, y_pred)
    - precision_recall_f1(y_true, y_pred)
    - classification_report(y_true, y_pred, class_names)
    - plot_confusion_matrix(y_true, y_pred, class_names)

Ejemplo:
    from neurox.metrics import classification_report, plot_confusion_matrix

    y_pred = model.predict_classes(X_test)
    classification_report(y_test, y_pred, class_names=['ILESO','HERIDO','FALLECIDO'])
    plot_confusion_matrix(y_test, y_pred, class_names=['ILESO','HERIDO','FALLECIDO'])
"""

import numpy as np


# ─── Helpers internos ────────────────────────────────────────────────────────

def _to_int(y: np.ndarray) -> np.ndarray:
    """Convierte one-hot o float a enteros."""
    y = np.array(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    return y.flatten().astype(int)


# ─── Matriz de confusión ─────────────────────────────────────────────────────

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de confusión.

    Retorna
    -------
    np.ndarray, shape (n_classes, n_classes)
    Filas = clase real, columnas = clase predicha.
    """
    y_true = _to_int(y_true)
    y_pred = _to_int(y_pred)
    classes = np.union1d(y_true, y_pred)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ─── Accuracy ────────────────────────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exactitud global: fracción de predicciones correctas."""
    y_true = _to_int(y_true)
    y_pred = _to_int(y_pred)
    return np.mean(y_true == y_pred)


# ─── Precision, Recall, F1 por clase ────────────────────────────────────────

def precision_recall_f1(y_true: np.ndarray,
                        y_pred: np.ndarray) -> dict:
    """
    Calcula precisión, recall y F1-score por clase, más promedios
    macro y weighted.

    Retorna
    -------
    dict con:
        'per_class'  : dict {clase: {'precision', 'recall', 'f1', 'support'}}
        'macro'      : {'precision', 'recall', 'f1'}
        'weighted'   : {'precision', 'recall', 'f1'}
        'accuracy'   : float
    """
    y_true = _to_int(y_true)
    y_pred = _to_int(y_pred)
    classes = np.unique(y_true)

    per_class = {}
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        support = np.sum(y_true == cls)

        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[int(cls)] = {
            'precision': prec,
            'recall':    rec,
            'f1':        f1,
            'support':   int(support),
        }

    # Macro (promedio simple)
    macro = {
        'precision': np.mean([v['precision'] for v in per_class.values()]),
        'recall':    np.mean([v['recall']    for v in per_class.values()]),
        'f1':        np.mean([v['f1']        for v in per_class.values()]),
    }

    # Weighted (promedio ponderado por support)
    total = len(y_true)
    weighted = {
        'precision': sum(v['precision'] * v['support'] for v in per_class.values()) / total,
        'recall':    sum(v['recall']    * v['support'] for v in per_class.values()) / total,
        'f1':        sum(v['f1']        * v['support'] for v in per_class.values()) / total,
    }

    return {
        'per_class': per_class,
        'macro':     macro,
        'weighted':  weighted,
        'accuracy':  accuracy(y_true, y_pred),
    }


# ─── Reporte de clasificación ────────────────────────────────────────────────

def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list = None) -> None:
    """
    Imprime un reporte de clasificación por clase.

    Parámetros
    ----------
    class_names : list|None — nombres de las clases (ej. ['ILESO', 'HERIDO'])
    """
    metrics = precision_recall_f1(y_true, y_pred)
    classes = sorted(metrics['per_class'].keys())

    if class_names is None:
        class_names = [str(c) for c in classes]

    col_w = max(len(n) for n in class_names) + 2

    header = f"{'Clase':<{col_w}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cls, name in zip(classes, class_names):
        m = metrics['per_class'][cls]
        print(f"{name:<{col_w}} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10}")

    print("-" * len(header))
    m = metrics['macro']
    print(f"{'macro avg':<{col_w}} {m['precision']:>10.4f} {m['recall']:>10.4f} "
          f"{m['f1']:>10.4f} {len(_to_int(y_true)):>10}")
    m = metrics['weighted']
    print(f"{'weighted avg':<{col_w}} {m['precision']:>10.4f} {m['recall']:>10.4f} "
          f"{m['f1']:>10.4f} {len(_to_int(y_true)):>10}")
    print("=" * len(header))
    print(f"Accuracy: {metrics['accuracy']:.4f}")


# ─── Gráfica de matriz de confusión ──────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list = None,
                          title: str = 'Matriz de Confusión',
                          savepath: str = None) -> None:
    """
    Grafica la matriz de confusión con matplotlib.

    Parámetros
    ----------
    class_names : list|None — nombres de las clases
    savepath    : str|None  — si se provee, guarda la figura en esa ruta
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib no está instalado.")
        return

    cm = confusion_matrix(y_true, y_pred)
    n  = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n)]

    # Normalizar para mostrar porcentajes dentro de cada fila
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n * 1.0)))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proporción', fontsize=9)

    # Ejes
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicción', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    # Anotaciones: count + porcentaje
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i,
                    f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                    ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en '{savepath}'")

    plt.show()