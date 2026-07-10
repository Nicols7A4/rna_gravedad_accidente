"""
neurox.network
=============
Clase Network: Contenedor principal de la red neuronal feedforward.
Encadena capas secuenciales, coordina el forward pass, ejecuta el backward pass
para retropropagar el gradiente, calcula costos y optimiza los parámetros de la red.


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
    Representa una Red Neuronal Artificial multicapa secuencial.
    """

    def __init__(self, layers: list, cost: str = 'mse', class_weights=None, gamma: float = 2.0):
        """
        Parámetros
        ----------
        layers : list — Lista secuencial de instancias de capas (Dense, Dropout).
        cost : str — Identificador de la función de coste ('mse', 'binary_crossentropy', 'categorical_crossentropy', 'focal_loss').
        class_weights : list|None — Vector de pesos por clase para el aprendizaje sensible al costo.
        gamma : float — Factor de modulación para la Focal Loss (por defecto 2.0).
        """
        self.layers = layers
        self.cost_name = cost
        self.class_weights = class_weights
        self.gamma = gamma

        # ── Configuración de Función de Costo ─────────────────────────────────
        if cost == 'focal_loss':
            from neurox.costs import get_focal_loss
            n_out = layers[-1].n_neurons
            if class_weights is not None and len(class_weights) != n_out:
                raise ValueError(
                    f"class_weights tiene {len(class_weights)} valores, "
                    f"pero la capa de salida tiene {n_out} neuronas."
                )
            self.cost_fn, self.cost_grad = get_focal_loss(gamma=gamma, alpha_vector=class_weights)
        elif class_weights is not None:
            if cost != 'categorical_crossentropy':
                raise ValueError(
                    "class_weights por ahora solo está soportado con cost='categorical_crossentropy'."
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

        self.history = []   # Historial de coste para graficar convergencia

    # ── MÓDULO DE SERIALIZACIÓN (PICKLE COMPATIBILITY) ───────────────────────
    # Debido a que las funciones de coste dinámicas (closures) no pueden ser
    # serializadas directamente por pickle, implementamos bypasses en el estado.

    def __getstate__(self):
        """Remueve los closures locales no serializables del estado."""
        state = self.__dict__.copy()
        if 'cost_fn' in state:
            del state['cost_fn']
        if 'cost_grad' in state:
            del state['cost_grad']
        return state

    def __setstate__(self, state):
        """Restaura los closures locales de pérdida al deserializar el modelo."""
        self.__dict__.update(state)
        if hasattr(self, 'cost_name'):
            if self.cost_name == 'focal_loss':
                from neurox.costs import get_focal_loss
                self.cost_fn, self.cost_grad = get_focal_loss(gamma=getattr(self, 'gamma', 2.0), alpha_vector=getattr(self, 'class_weights', None))
            elif getattr(self, 'class_weights', None) is not None:
                from neurox.costs import weighted_categorical_crossentropy
                self.cost_fn, self.cost_grad = weighted_categorical_crossentropy(self.class_weights)
            else:
                from neurox.costs import get_cost
                self.cost_fn, self.cost_grad = get_cost(self.cost_name)
        else:
            self.cost_fn, self.cost_grad = None, None

    # ── PROPAGACIÓN HACIA ADELANTE (FORWARD) ─────────────────────────────────

    def _forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Propaga la entrada secuencialmente a través de las capas de la red."""
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    # ── RETROPROPAGACIÓN DE GRADIENTES (BACKWARD) ─────────────────────────────

    def _backward(self, y_pred: np.ndarray, y_true: np.ndarray,
                  alpha: float, momentum: float = 0.0,
                  weight_decay: float = 0.0, clip_norm: float = None,
                  optimizer: str = 'sgd') -> None:
        """Propaga el error de salida hacia atrás y actualiza los parámetros."""
        grad = self.cost_grad(y_pred, y_true)
        for layer in reversed(self.layers):
            if clip_norm is not None:
                grad = self._clip_grad(grad, clip_norm)
            grad = layer.backward(grad, alpha, momentum=momentum,
                                   weight_decay=weight_decay,
                                   optimizer=optimizer)

    @staticmethod
    def _clip_grad(grad: np.ndarray, max_norm: float) -> np.ndarray:
        """
        Recorta el gradiente local si su norma L2 supera el umbral (Gradient Clipping).
        Estabiliza el entrenamiento y evita gradientes explosivos en mini-lotes.
        """
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / (norm + 1e-12))
        return grad

    # ── ENTRENAMIENTO (TRAINING LOOP) ────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 1000,
              alpha: float = 0.01,
              batch_size: int = None,
              momentum: float = 0.0,
              weight_decay: float = 0.0,
              clip_norm: float = None,
              shuffle: bool = True,
              seed: int = None,
              optimizer: str = 'sgd',
              verbose: int = 10) -> list:
        """
        Entrena la red mediante Gradiente Descendiente Estocástico (SGD) o Adam.
        """
        self.history = []
        X, y = np.array(X), np.array(y)

        # Forzar formato bidimensional para el vector objetivo
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m = X.shape[0]
        rng = np.random.default_rng(seed)
        bs = batch_size if batch_size is not None else m

        for e in range(epochs):
            # Barajado de lotes para mitigar correlaciones locales (Shuffle)
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

                # 1. Forward Pass
                y_pred = self._forward(X_batch, training=True)
                # 2. Cost
                loss = self.cost_fn(y_pred, y_batch)
                epoch_loss_sum += loss * len(X_batch)
                # 3. Backward Pass
                self._backward(y_pred, y_batch, alpha,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                clip_norm=clip_norm,
                                optimizer=optimizer)

            epoch_loss = epoch_loss_sum / m
            self.history.append(epoch_loss)

            if verbose and e % max(1, epochs // verbose) == 0:
                print(f"(Época {e:>6}) Costo: {epoch_loss:.6f}")

        print(f"(Época {epochs - 1:>6}) Costo: {self.history[-1]:.6f}")
        return self.history

    # ── PREDICCIÓN E INFERENCIA ──────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza el paso forward sin activar regularización (Dropout)."""
        return self._forward(np.array(X), training=False)

    def predict_classes(self, X: np.ndarray,
                        threshold: float = 0.5) -> np.ndarray:
        """
        Determina las etiquetas de salida en función de las probabilidades.
        """
        probs = self.predict(X)
        if probs.shape[1] == 1:
            return (probs >= threshold).astype(int).flatten()
        else:
            return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 threshold: float = 0.5) -> dict:
        """Calcula la precisión global (accuracy) y coste medio sobre un set dado."""
        X, y = np.array(X), np.array(y)
        y_pred = self.predict(X)

        if y.ndim == 1:
            y_mat = y.reshape(-1, 1)
        else:
            y_mat = y

        loss = self.cost_fn(y_pred, y_mat)
        clases_pred = self.predict_classes(X, threshold)

        if y.ndim == 2 and y.shape[1] > 1:
            y_true_cls = np.argmax(y, axis=1)
        else:
            y_true_cls = y.flatten().astype(int)

        accuracy = np.mean(clases_pred == y_true_cls)
        return {'accuracy': accuracy, 'cost': loss}

    # ── VISUALIZACIÓN Y DIAGNÓSTICOS ─────────────────────────────────────────

    def plot_cost(self, title: str = 'Curva de Costo', savepath: str = None) -> None:
        """Grafica y guarda la curva de convergencia de coste por época."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib no está instalado.")
            return

        if not self.history:
            print("No hay historial de costo registrado.")
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

    def summary(self) -> None:
        """Imprime por consola un resumen detallado de la topología de la red."""
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
        """
        Dibuja y guarda un diagrama esquemático de los nodos y conexiones de la red.
        Omite las capas de Dropout para evitar graficar nodos vacíos.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib no está instalado.")
            return

        visible_layers = [l for l in self.layers if l.__class__.__name__ != 'Dropout']
        sizes = [visible_layers[0].n_inputs] + [l.n_neurons for l in visible_layers]
        n_layers = len(sizes)
        max_neurons = max(sizes)

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

        # Dibujar líneas de conexión
        for i in range(n_layers - 1):
            if hasattr(visible_layers[i], 'W'):
                layer_weights = visible_layers[i].W
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

        # Dibujar círculos de nodos
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

        # Dibujar etiquetas superiores de capa
        layer_labels = ['Entrada'] + \
                       [f'Oculta {i+1}\n({visible_layers[i].activation_name})'
                        for i in range(len(visible_layers) - 1)] + \
                       [f'Salida\n({visible_layers[-1].activation_name})']

        for i, (layer_pos, label) in enumerate(zip(positions, layer_labels)):
            x = layer_pos[0][0]
            y_max = max(pos[1] for pos in layer_pos)
            ax.text(x, y_max + 1.2, label, ha='center', va='bottom',
                    fontsize=8.5, color=COLOR_TEXT, fontweight='bold')

        plt.title(title, fontsize=12, fontweight='bold', color=COLOR_TEXT, pad=20)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=200, bbox_inches='tight')
            print(f"Diagrama de la red guardado en '{savepath}'")

        plt.close()