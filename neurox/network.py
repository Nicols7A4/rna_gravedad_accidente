"""
neurox.network
--------------
Clase Network: encadena capas Dense, entrena con backpropagation
y evalúa el modelo.

Uso:
    from neurox import Network, Dense

    model = Network(
        layers=[
            Dense(6, 16, activation='relu'),
            Dense(16, 1, activation='sigmoid')
        ],
        cost='binary_crossentropy'
    )

    history = model.train(X_train, y_train, epochs=1000, alpha=0.05)
    model.plot_cost()
    precision = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
"""

import numpy as np
from neurox.costs import get_cost, weighted_categorical_crossentropy


class Network:
    """
    Red neuronal feedforward totalmente configurable.

    Parámetros
    ----------
    layers : list[Dense]
        Lista de capas Dense en orden (entrada → salida).
    cost   : str
        Función de costo: 'mse', 'binary_crossentropy',
        'categorical_crossentropy'.
    class_weights : list|None
        Peso por clase (mismo orden que las columnas del one-hot de y).
        Un peso > 1 hace que el modelo priorice esa clase (le pesa más el
        error), un peso < 1 la de-prioriza. Solo válido junto con
        cost='categorical_crossentropy'.
    """

    def __init__(self, layers: list, cost: str = 'mse', class_weights=None):
        self.layers = layers
        self.cost_name = cost
        self.class_weights = class_weights

        if class_weights is not None:
            if cost != 'categorical_crossentropy':
                raise ValueError(
                    "class_weights por ahora solo está soportado con "
                    "cost='categorical_crossentropy'."
                )
            n_out = layers[-1].n_neurons
            if len(class_weights) != n_out:
                raise ValueError(
                    f"class_weights tiene {len(class_weights)} valores, "
                    f"pero la capa de salida tiene {n_out} neuronas."
                )
            self.cost_fn, self.cost_grad = weighted_categorical_crossentropy(class_weights)
        else:
            self.cost_fn, self.cost_grad = get_cost(cost)

        self.history = []   # costo por época (para graficar)

    # ── Forward ──────────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    # ── Backward ─────────────────────────────────────────────────────────────

    def _backward(self, y_pred: np.ndarray, y_true: np.ndarray,
                  alpha: float, momentum: float = 0.0,
                  weight_decay: float = 0.0, clip_norm: float = None) -> None:
        grad = self.cost_grad(y_pred, y_true)
        for layer in reversed(self.layers):
            if clip_norm is not None:
                grad = self._clip_grad(grad, clip_norm)
            grad = layer.backward(grad, alpha, momentum=momentum,
                                   weight_decay=weight_decay)

    @staticmethod
    def _clip_grad(grad: np.ndarray, max_norm: float) -> np.ndarray:
        """
        Recorta la señal de gradiente que fluye entre capas para que su
        norma no supere max_norm (gradient clipping).

        Nota: esto recorta el gradiente que se propaga capa a capa
        (∂L/∂output de cada capa), no la norma global de todos los
        parámetros a la vez como en frameworks tipo PyTorch. En una red
        feedforward simple como esta, es la forma práctica de lograr el
        mismo efecto estabilizador sin rehacer todo el motor de backward
        en dos fases.
        """
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / (norm + 1e-12))
        return grad

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 1000,
              alpha: float = 0.01,
              batch_size: int = None,
              momentum: float = 0.0,
              weight_decay: float = 0.0,
              clip_norm: float = None,
              shuffle: bool = True,
              seed: int = None,
              verbose: int = 10) -> list:
        """
        Entrena la red con gradient descent (full-batch o mini-batch/SGD).

        Parámetros
        ----------
        X            : np.ndarray, shape (m, n_features)
        y            : np.ndarray, shape (m,) o (m, n_clases)
        epochs       : int   — número de épocas
        alpha        : float — tasa de aprendizaje
        batch_size   : int|None — tamaño de mini-batch. None = full-batch
                       (todo el dataset en cada paso, comportamiento original).
        momentum     : float — coeficiente de momentum clásico (0 = SGD puro)
        weight_decay : float — coeficiente L2 (ver advertencia en Dense.backward
                       sobre valores grandes como 0.9)
        clip_norm    : float|None — umbral de gradient clipping (None = sin clip)
        shuffle      : bool — baraja los datos en cada época (solo relevante
                       si batch_size no es None)
        seed         : int|None — semilla para el barajado
        verbose      : int   — imprime el costo cada N épocas (0 = silencioso)

        Retorna
        -------
        list — historial de costo por época (promedio ponderado si hay batches)
        """
        self.history = []
        X, y = np.array(X), np.array(y)

        # Asegurar que y tenga shape (m, ?) para operaciones matriciales
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m = X.shape[0]
        rng = np.random.default_rng(seed)
        bs = batch_size if batch_size is not None else m

        for e in range(epochs):
            if shuffle and batch_size is not None:
                idx = rng.permutation(m)
                X_epoch, y_epoch = X[idx], y[idx]
            else:
                X_epoch, y_epoch = X, y

            epoch_loss_sum = 0.0
            for start in range(0, m, bs):
                end = start + bs
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]

                y_pred = self._forward(X_batch, training=True)
                loss   = self.cost_fn(y_pred, y_batch)
                epoch_loss_sum += loss * len(X_batch)
                self._backward(y_pred, y_batch, alpha,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                clip_norm=clip_norm)

            epoch_loss = epoch_loss_sum / m
            self.history.append(epoch_loss)

            if verbose and e % max(1, epochs // verbose) == 0:
                print(f"(Época {e:>6}) Costo: {epoch_loss:.6f}")

        print(f"(Época {epochs - 1:>6}) Costo: {self.history[-1]:.6f}")
        return self.history

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass sin actualizar pesos.

        Retorna
        -------
        np.ndarray con las probabilidades crudas de la capa de salida.
        """
        return self._forward(np.array(X), training=False)

    def predict_classes(self, X: np.ndarray,
                        threshold: float = 0.5) -> np.ndarray:
        """
        Retorna clases predichas.
        - Binario   : 0 o 1 según threshold.
        - Multiclase: índice de la clase con mayor probabilidad (argmax).

        Parámetros
        ----------
        threshold : float — solo aplica para salida binaria (1 neurona).
        """
        probs = self.predict(X)
        if probs.shape[1] == 1:
            return (probs >= threshold).astype(int).flatten()
        else:
            return np.argmax(probs, axis=1)

    # ── Evaluación ────────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 threshold: float = 0.5) -> dict:
        """
        Calcula precisión (accuracy) y costo sobre el conjunto dado.

        Retorna
        -------
        dict con 'accuracy' y 'cost'.
        """
        X, y = np.array(X), np.array(y)
        y_pred = self.predict(X)

        if y.ndim == 1:
            y_mat = y.reshape(-1, 1)
        else:
            y_mat = y

        loss = self.cost_fn(y_pred, y_mat)

        clases_pred = self.predict_classes(X, threshold)

        # y_true como clases enteras para comparar
        if y.ndim == 2 and y.shape[1] > 1:   # one-hot
            y_true_cls = np.argmax(y, axis=1)
        else:
            y_true_cls = y.flatten().astype(int)

        accuracy = np.mean(clases_pred == y_true_cls)

        return {'accuracy': accuracy, 'cost': loss}

    # ── Gráfica ───────────────────────────────────────────────────────────────

    def plot_cost(self, title: str = 'Curva de Costo', savepath: str = None) -> None:
        """
        Grafica el historial de costo usando matplotlib.
        Requiere que train() haya sido llamado antes.

        Parámetros
        ----------
        savepath : str|None — si se provee, guarda la figura en esa ruta (PNG)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib no está instalado. Instálalo con: pip install matplotlib")
            return

        if not self.history:
            print("No hay historial de entrenamiento. Llama a train() primero.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(self.history, color='steelblue', linewidth=1.5)
        plt.title(title)
        plt.xlabel('Época')
        plt.ylabel(f'Costo ({self.cost_name})')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=150, bbox_inches='tight')
            print(f"Figura guardada en '{savepath}'")

        plt.show()

    # ── Info ─────────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Imprime un resumen de la arquitectura."""
        total = 0
        print("=" * 45)
        print(f"{'Capa':<20} {'Shape W':<15} {'Params':>8}")
        print("=" * 45)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                shape = f"({layer.n_inputs}, {layer.n_neurons})"
                params = layer.n_params
            else:
                shape = "N/A"
                params = 0
            name = f"{layer.__class__.__name__}_{i+1}"
            print(f"{name:<20} [{layer.activation_name:<8}] {shape:<15} {params:>8}")
            total += params
        print("=" * 45)
        print(f"{'Total parámetros':<35} {total:>8}")
        print(f"Costo: {self.cost_name}")
        print("=" * 45)

    def __repr__(self):
        return f"Network(capas={len(self.layers)}, cost='{self.cost_name}')"

    def plot_network(self, feature_names: list = None,
                     output_names: list = None,
                     title: str = 'Arquitectura de la Red',
                     show_weights: bool = True,
                     theme: str = 'light',
                     savepath: str = 'red_neuronal.png') -> None:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib no está instalado.")
            return

        sizes = [self.layers[0].n_inputs] + [l.n_neurons for l in self.layers]
        n_layers = len(sizes)
        max_neurons = max(sizes)

        # Si alguna capa tiene muchas neuronas (típico tras one-hot con
        # muchas entradas), comprimimos el espaciado vertical entre nodos
        # y desactivamos las etiquetas de peso para que el PNG no termine
        # siendo gigantesco o ilegible.
        y_scale = 1.0
        if max_neurons > 30:
            y_scale = 30 / max_neurons
            show_weights = False

        if theme == 'dark':
            COLOR_INPUT  = '#5B8DD9'
            COLOR_HIDDEN = '#F0A500'
            COLOR_OUTPUT = '#4CAF82'
            COLOR_EDGE   = '#4A4A5A' 
            COLOR_BG     = '#1E1E2E'
            COLOR_TEXT   = '#FFFFFF'
            COLOR_MUTED  = '#A6ADC8'
            BOX_BG       = '#181825E6'
            BOX_EDGE     = '#313244'
            LEGEND_BG    = '#2A2A35'
        else:
            COLOR_INPUT  = '#8DB3E2'
            COLOR_HIDDEN = '#F4B942'
            COLOR_OUTPUT = '#6CC3A0'
            COLOR_EDGE   = '#9AA4B2'
            COLOR_BG     = '#FAFAF7'
            COLOR_TEXT   = '#1F2937'
            COLOR_MUTED  = '#6B7280'
            BOX_BG       = '#FFFFFFE6'
            BOX_EDGE     = '#D1D5DB'
            LEGEND_BG    = '#EEF2F7'

        fig, ax = plt.subplots(figsize=(max(8, n_layers * 2.5),
                                        min(max(5, max_neurons * 0.9 * y_scale), 22)))
        ax.set_facecolor(COLOR_BG)
        fig.patch.set_facecolor(COLOR_BG)
        ax.axis('off')

        node_radius = 0.28 * min(1.0, y_scale + 0.3)
        x_gap = 1.0
        positions = []

        for i, n in enumerate(sizes):
            x = i * x_gap * 2
            ys = [((max_neurons - n) / 2 + j) * y_scale for j in range(n)]
            positions.append([(x, y) for y in ys])

        # Conexiones
        for i in range(n_layers - 1):
            if hasattr(self.layers[i], 'W'):
                layer_weights = self.layers[i].W
                for idx_from, (x0, y0) in enumerate(positions[i]):
                    for idx_to, (x1, y1) in enumerate(positions[i + 1]):
                        ax.plot([x0, x1], [y0, y1],
                                color=COLOR_EDGE, linewidth=0.4, alpha=0.4, zorder=1)

                        if show_weights:
                            weight = layer_weights[idx_from, idx_to]
                            x_mid = (x0 + x1) / 2
                            y_mid = (y0 + y1) / 2
                            ax.text(x_mid, y_mid, f'{weight:.2f}',
                                    ha='center', va='center', fontsize=5.5,
                                    color=COLOR_TEXT, zorder=4,
                                    bbox=dict(boxstyle='round,pad=0.12',
                                            facecolor=BOX_BG,
                                            edgecolor=BOX_EDGE))
            else:
                for idx, ((x0, y0), (x1, y1)) in enumerate(zip(positions[i], positions[i + 1])):
                    ax.plot([x0, x1], [y0, y1],
                            color=COLOR_EDGE, linewidth=0.4, alpha=0.4, zorder=1)

        # Nodos
        for i, layer_pos in enumerate(positions):
            for j, (x, y) in enumerate(layer_pos):
                if i == 0:
                    color = COLOR_INPUT
                elif i == n_layers - 1:
                    color = COLOR_OUTPUT
                else:
                    color = COLOR_HIDDEN

                circle = plt.Circle((x, y), node_radius,
                                    color=color, zorder=2,
                                    ec='white' if theme == 'light' else COLOR_BG, 
                                    lw=0.8)
                ax.add_patch(circle)

                node_text_color = '#FFFFFF' if theme == 'dark' else COLOR_TEXT
                ax.text(x, y, str(j + 1), ha='center', va='center',
                    fontsize=7, color=node_text_color, fontweight='bold', zorder=3)

                if i == 0 and feature_names and j < len(feature_names):
                    ax.text(x - node_radius - 0.1, y, feature_names[j],
                        ha='right', va='center', fontsize=7.5, color=COLOR_MUTED)

                if i == n_layers - 1 and output_names and j < len(output_names):
                    ax.text(x + node_radius + 0.1, y, output_names[j],
                        ha='left', va='center', fontsize=7.5, color=COLOR_MUTED)

        # Etiquetas de capa
        layer_labels = ['Entrada'] + \
                       [f'Oculta {i+1}\n({self.layers[i].activation_name})'
                        for i in range(len(self.layers) - 1)] + \
                       [f'Salida\n({self.layers[-1].activation_name})']

        for i, (layer_pos, label) in enumerate(zip(positions, layer_labels)):
            x = layer_pos[0][0]
            y_top = max(p[1] for p in layer_pos) + node_radius + 0.35
            ax.text(x, y_top, label, ha='center', va='bottom',
                    fontsize=8, color=COLOR_TEXT, fontweight='bold')
            ax.text(x, min(p[1] for p in layer_pos) - node_radius - 0.35,
                    f'n={sizes[i]}', ha='center', va='top',
                    fontsize=7, color=COLOR_MUTED)

        # Leyenda
        legend = [
            mpatches.Patch(color=COLOR_INPUT,  label='Entrada'),
            mpatches.Patch(color=COLOR_HIDDEN, label='Oculta'),
            mpatches.Patch(color=COLOR_OUTPUT, label='Salida'),
        ]
        ax.legend(handles=legend, loc='lower right',
                  facecolor=LEGEND_BG, edgecolor=BOX_EDGE,
                  labelcolor=COLOR_TEXT, fontsize=8)

        all_x = [p[0] for lp in positions for p in lp]
        all_y = [p[1] for lp in positions for p in lp]
        ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
        ax.set_ylim(min(all_y) - 1.2, max(all_y) + 1.2)

        ax.set_title(title, color=COLOR_TEXT, fontsize=11,
                     fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(savepath, dpi=150,
                    bbox_inches='tight', facecolor=COLOR_BG)
        plt.show()
        print(f"Diagrama guardado como '{savepath}'")