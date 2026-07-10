"""
train_siniestros.py
====================
Entrena una red neuronal (con la librería neurox, hecha desde cero) para
predecir la GRAVEDAD de un siniestro de tránsito: ILESO / LESIONADO / FALLECIDO.

Método A: se descarta "NO SE CONOCE" y se hace UNDERSAMPLING de las 3 clases
restantes a la cantidad de la clase minoritaria (5284, que es ILESO), para
que el modelo deje de converger siempre a "FALLECIDO".

Iteración 2 (objetivo: subir de ~70% a ~80% accuracy):
    - Se agregaron features: HORA, CAUSA ESPECIFICA, RED VIAL, DEPARTAMENTO.
    - Categorías raras (< 30 apariciones) en columnas de alta cardinalidad
      (CAUSA, CAUSA ESPECIFICA, VEHÍCULO) se agrupan en "OTRO".
    - Arquitectura más grande: 3 capas ocultas (128 -> 64 -> 32) en vez de 2.
    - Se reporta accuracy en train vs test para diagnosticar
      underfitting vs overfitting.

Uso:
    python train_siniestros.py

Requiere: DATA/datos.csv (ruta relativa a este script) con el esquema de
columnas del dataset de siniestros (MTC/PNP).
"""

import numpy as np
import pandas as pd

from neurox import (
    Network, Dense, Dropout,
    Normalizer, one_hot, LabelEncoder,
    train_test_split, k_fold_split
)
from neurox.metrics import classification_report, plot_confusion_matrix

RUTA_CSV = "DATA/datos.csv"
SEMILLA = 42
CLASES = ["ILESO", "LESIONADO", "FALLECIDO"]  # orden fijo para one-hot

# Carpeta donde se guardan los PNG de salida
CARPETA_SALIDA = "outputs"
PNG_COSTO = f"{CARPETA_SALIDA}/curva_costo.png"
PNG_MATRIZ = f"{CARPETA_SALIDA}/matriz_confusion.png"
PNG_RED = f"{CARPETA_SALIDA}/arquitectura_red.png"

# Peso por clase en la pérdida (mismo orden que CLASES).
# 1.0 = neutral. > 1.0 = el modelo prioriza esa clase (le pesa más el error).
# Ya balanceaste las clases con undersampling, así que esto ya no es para
# compensar frecuencia -- es para forzar al modelo a prestarle más atención
# a las clases que confunde entre sí (LESIONADO <-> FALLECIDO).
# Ajusta estos 3 números y vuelve a correr para ver el efecto.
CLASS_WEIGHTS = None  
# ej.: [1.0, 1.4, 1.4]  (orden: ILESO, LESIONADO, FALLECIDO)
# CLASS_WEIGHTS = [1.0, 1.1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carga y método A (undersampling balanceado)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_y_balancear(ruta_csv: str, seed: int = SEMILLA, balancear: bool = True) -> pd.DataFrame:
    # Cargar el dataset
    df = pd.read_csv(ruta_csv, encoding="utf-8")

    # Descartamos "NO SE CONOCE": no es una clase real de gravedad
    df = df[df["GRAVEDAD"].isin(CLASES)].copy()

    if not balancear:
        print(f"Dataset completo (sin balancear): {len(df)} filas totales")
        print(df["GRAVEDAD"].value_counts())
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Undersampling: llevamos las 3 clases al tamaño de la clase minoritaria
    n_min = df["GRAVEDAD"].value_counts().min()
    rng = np.random.default_rng(seed)

    partes = []
    for clase in CLASES:
        sub = df[df["GRAVEDAD"] == clase]
        idx = rng.choice(sub.index, size=n_min, replace=False)
        partes.append(sub.loc[idx])

    df_bal = pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Dataset balanceado (método A): {n_min} casos x {len(CLASES)} clases "
          f"= {len(df_bal)} filas totales")
    print(df_bal["GRAVEDAD"].value_counts())
    return df_bal


# ─────────────────────────────────────────────────────────────────────────────
# 2. Selección de features y preprocesamiento
# ─────────────────────────────────────────────────────────────────────────────
# Se excluyen: códigos/IDs (no aportan patrón real, son identificadores únicos),
# LUGAR DE DEFUNCIÓN / LUGAR ATENCIÓN LESIONADO / SITUACIÓN DE PERSONA (son
# consecuencia de la gravedad, no causa -> usarlas sería "leakage": el modelo
# haría trampa prediciendo con información que solo existe DESPUÉS de saber
# el resultado).

FEATURES_NUMERICAS = ["EDAD"]

FEATURES_CATEGORICAS = [
    "TIPO PERSONA", "SEXO", "POSEE LICENCIA", "ESTADO LICENCIA",
    "CLASE_LICENCIA", "¿SE SOMETIÓ A DOSAJE ETÍLICO CUALITATIVO?",
    "RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO",
    "¿SE SOMETIÓ A DOSAJE ETÍLICO CUANTITATIVO?",
    "VEHÍCULO", "CLASE DE SINIESTRO", "CAUSA", "CAUSA ESPECIFICA",
    "TIPO DE VÍA", "RED VIAL", "DEPARTAMENTO",
    "MES", "DIA", "HORA",
]

# Columnas de alta cardinalidad: agrupamos las categorías poco frecuentes
# en "OTRO" para que no diluyan el gradiente (una categoría que aparece
# 5 veces en todo el dataset casi no aporta señal, pero sí infla el
# número de columnas one-hot).
COLUMNAS_ALTA_CARDINALIDAD = ["CAUSA", "CAUSA ESPECIFICA", "VEHÍCULO"]
MIN_FRECUENCIA = 30  # categorías con menos de esto -> "OTRO"


class SiniestrosPreprocessor:
    def __init__(self, min_frecuencia=MIN_FRECUENCIA):
        self.min_frecuencia = min_frecuencia
        self.frequent_categories_ = {}          # Guarda qué categorías superan el umbral 30
        self.median_edad_ = None                # Guarda la mediana de edad de entrenamiento
        self.columns_ohe_ = None                # Guarda la lista exacta de las 213 columnas 

    def fit(self, df: pd.DataFrame):
        # 1. Mediana de edad
        edad_num = pd.to_numeric(df["EDAD"], errors="coerce")
        self.median_edad_ = edad_num.median()
        if pd.isna(self.median_edad_):
            self.median_edad_ = 35.0

        # 2. Categorías frecuentes para columnas de alta cardinalidad
        for col in COLUMNAS_ALTA_CARDINALIDAD:
            if col in df.columns:
                series = df[col].fillna("DESCONOCIDO").astype(str)
                counts = series.value_counts()
                frequent = counts[counts >= self.min_frecuencia].index.tolist()
                self.frequent_categories_[col] = set(frequent)

        # 3. Determinar el esquema de columnas de one-hot encoding
        df_clean = self._preprocess_basic(df, fitting=True)
        df_ohe = pd.get_dummies(df_clean[FEATURES_CATEGORICAS], drop_first=False)
        X_temp = pd.concat([df_clean[FEATURES_NUMERICAS], df_ohe], axis=1)
        self.columns_ohe_ = X_temp.columns.tolist()
        return self

    def _preprocess_basic(self, df: pd.DataFrame, fitting=False) -> pd.DataFrame:
        df = df.copy()
        # Rellenar nulos categóricos
        for col in FEATURES_CATEGORICAS:
            if col in df.columns:
                df[col] = df[col].fillna("DESCONOCIDO").astype(str)
            else:
                df[col] = "DESCONOCIDO"

        # Normalizar HORA a enteros como strings
        if "HORA" in df.columns:
            def clean_hora(x):
                try:
                    if ':' in str(x):
                        return str(int(str(x).split(':')[0]))
                    else:
                        return str(int(float(x)))
                except Exception:
                    return str(x)
            df["HORA"] = df["HORA"].apply(clean_hora)

        # Agrupar categorías raras
        for col in COLUMNAS_ALTA_CARDINALIDAD:
            if col in df.columns:
                if col in self.frequent_categories_:
                    df[col] = df[col].apply(lambda x: x if x in self.frequent_categories_[col] else "OTRO")
                else:
                    # Durante el fit, aún no tenemos frequent_categories_, hacemos agrupación al vuelo
                    counts = df[col].value_counts()
                    raras = counts[counts < self.min_frecuencia].index
                    df[col] = df[col].where(~df[col].isin(raras), other="OTRO")

        # Rellenar EDAD
        if "EDAD" in df.columns:
            df["EDAD"] = pd.to_numeric(df["EDAD"], errors="coerce")
            df["EDAD"] = df["EDAD"].fillna(self.median_edad_ if self.median_edad_ is not None else 35.0)
        else:
            df["EDAD"] = self.median_edad_ if self.median_edad_ is not None else 35.0

        return df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df_clean = self._preprocess_basic(df)
        df_ohe = pd.get_dummies(df_clean[FEATURES_CATEGORICAS], drop_first=False)
        X = pd.concat([df_clean[FEATURES_NUMERICAS], df_ohe], axis=1)

        # Alinear columnas
        for col in self.columns_ohe_:
            if col not in X.columns:
                X[col] = 0

        X = X[self.columns_ohe_]
        return X.to_numpy(dtype=float)


def codificar_target(df: pd.DataFrame, le: LabelEncoder) -> np.ndarray:
    y_idx = le.transform(df["GRAVEDAD"].values)
    return one_hot(y_idx, n_classes=len(CLASES))


def preparar_features(df: pd.DataFrame):
    """
    Función de compatibilidad. Cuidado: puede tener data leakage si se usa directamente
    sobre todo el dataset antes del split. Se prefiere usar SiniestrosPreprocessor.
    """
    prep = SiniestrosPreprocessor()
    prep.fit(df)
    X = prep.transform(df)

    le = LabelEncoder()
    le.classes_ = np.array(CLASES)
    le._mapping = {c: i for i, c in enumerate(CLASES)}
    le._inverse = {i: c for c, i in le._mapping.items()}
    y_idx = le.transform(df["GRAVEDAD"].values)
    y_ohe = one_hot(y_idx, n_classes=len(CLASES))

    return X, y_ohe, prep.columns_ohe_, le


# ─────────────────────────────────────────────────────────────────────────────
# 3. Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Configuración de balanceo y pérdida (Propuesta B)
    balancear = False  # True: undersampling tradicional, False: dataset completo
    usar_focal_loss = True  # True: Focal Loss, False: Categorical Crossentropy

    df = cargar_y_balancear(RUTA_CSV, seed=SEMILLA, balancear=balancear)

    # Separación df_train / df_test para evitar data leakage (aleatoria sin semilla fija)
    df_train = df.sample(frac=0.8, random_state=None)
    df_test = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    preprocesador = SiniestrosPreprocessor()
    preprocesador.fit(df_train)

    X_train = preprocesador.transform(df_train)
    X_test = preprocesador.transform(df_test)

    le = LabelEncoder()
    le.classes_ = np.array(CLASES)
    le._mapping = {c: i for i, c in enumerate(CLASES)}
    le._inverse = {i: c for c, i in le._mapping.items()}

    y_train = codificar_target(df_train, le)
    y_test = codificar_target(df_test, le)

    # Configurar pesos y función de costo adaptativos
    if usar_focal_loss and not balancear:
        class_counts = np.sum(y_train, axis=0)
        total_samples = y_train.shape[0]
        k_classes = y_train.shape[1]
        class_counts[class_counts == 0] = 1
        adaptive_alpha = total_samples / (k_classes * class_counts)
        print(f"Pesos adaptativos de clase calculados: {adaptive_alpha}")
        class_weights = adaptive_alpha
        cost_function = "focal_loss"
    else:
        class_weights = CLASS_WEIGHTS
        cost_function = "categorical_crossentropy"

    columnas = preprocesador.columns_ohe_
    print(f"\nDimensión final de entrada (features tras one-hot): {X_train.shape[1]}")

    # Normalización (fit solo en train, transform en test -> evita fuga de info)
    norm = Normalizer()
    X_train = norm.fit_transform(X_train)
    X_test = norm.transform(X_test)

    print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    # ── Arquitectura ──────────────────────────────────────────────────────
    # Softmax + categorical_crossentropy / focal_loss para 3 clases
    # 3 capas ocultas más Dropout para regularización.
    model = Network(
        layers=[
            Dense(X_train.shape[1], 128, activation="relu", seed=SEMILLA),
            Dropout(0.3, seed=SEMILLA + 1),
            Dense(128, 64, activation="relu", seed=SEMILLA + 2),
            Dropout(0.3, seed=SEMILLA + 3),
            Dense(64, 32, activation="relu", seed=SEMILLA + 4),
            Dropout(0.3, seed=SEMILLA + 5),
            Dense(32, len(CLASES), activation="softmax", seed=SEMILLA + 6),
        ],
        cost=cost_function,
        class_weights=class_weights,
        gamma=2.0,
    )
    model.summary()

    # ── Hiperparámetros (los que la librería soporta de tu tabla) ─────────
    print("\n[ENTRENAMIENTO] Iniciando entrenamiento con optimizador ADAM...")
    historial_costo = model.train(
        X_train, y_train,
        epochs=100,
        alpha=0.001,
        batch_size=64,
        momentum=0.80,  # Ignorado por Adam
        weight_decay=1e-4,
        clip_norm=2.0,
        seed=SEMILLA,
        optimizer='adam',
        verbose=10,
    )

    model.plot_cost(title="Costo — Método A (Focal Loss)", savepath=PNG_COSTO)

    # ── Diagnóstico y Captura de Logs ──────────────────────────────────────
    import io
    import datetime
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        print("=" * 70)
        print("REGISTRO DE ENTRENAMIENTO - PREDICTOR DE SINIESTROS VIALES")
        print(f"Fecha de Ejecución: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"Dataset de Entrada: {df.shape[0]} registros totales")
        print(f"Clase de Costo Utilizada: {model.cost_name}")
        print(f"Parámetros de Entrenamiento:")
        print(f"  - Optimizador: ADAM")
        print(f"  - Épocas: 200")
        print(f"  - Tasa de Aprendizaje (alpha): 0.002")
        print(f"  - Tamaño de Batch (batch_size): 64")
        print(f"  - Weight Decay (L2): 1e-4")
        print(f"  - Clip Norm: 2.0")
        print(f"  - Semilla de pesos: {SEMILLA}")
        print("\nArquitectura de la Red Neuronal:")
        model.summary()
        
        print("\nHistorial del Costo por Época:")
        for idx, costo in enumerate(historial_costo):
            if idx % 10 == 0 or idx == len(historial_costo) - 1:
                print(f"  Época {idx:>3}: Costo = {costo:.6f}")
                
        print("\nEvaluación en Conjunto de Validación (20% Test):")
        res_train = model.evaluate(X_train, y_train)
        res_test = model.evaluate(X_test, y_test)
        print(f"Accuracy train: {res_train['accuracy']:.4f}  (costo: {res_train['cost']:.4f})")
        print(f"Accuracy test:  {res_test['accuracy']:.4f}  (costo: {res_test['cost']:.4f})")
        brecha = res_train['accuracy'] - res_test['accuracy']
        if brecha > 0.10:
            print(f"-> Diagnóstico: Brecha de {brecha:.1%} indica OVERFITTING")
        elif res_train['accuracy'] < 0.80:
            print(f"-> Diagnóstico: Train bajo indica UNDERFITTING")
        else:
            print(f"-> Diagnóstico: Modelo estable y con buena generalización")
            
        print("\nReporte de Clasificación Final por Clase:")
        y_pred_cls = model.predict_classes(X_test)
        y_true_cls = np.argmax(y_test, axis=1)
        classification_report(y_true_cls, y_pred_cls, class_names=CLASES)
        print("=" * 70)

    report_str = f.getvalue()
    # Mostrar el reporte en la consola de ejecución normal
    print(report_str)

    # Guardar el registro de entrenamiento en outputs/log_entrenamiento.txt
    path_log = os.path.join(CARPETA_SALIDA, "log_entrenamiento.txt")
    with open(path_log, "w", encoding="utf-8") as f_log:
        f_log.write(report_str)
    print(f"\n[REGISTRO] Log del entrenamiento guardado en: {path_log}")

    plot_confusion_matrix(y_true_cls, y_pred_cls, class_names=CLASES,
                           title="Matriz de Confusión — Método A",
                           savepath=PNG_MATRIZ)

    # ── Diagrama de la arquitectura ───────────────────────────────────────
    # model.plot_network(
    #     feature_names=None,   # con one-hot son demasiadas para mostrar una por una
    #     output_names=CLASES,
    #     title="Arquitectura — Clasificador de Gravedad",
    #     show_weights=False,   # con 43+ entradas, mostrar cada peso satura el diagrama
    #     savepath=PNG_RED,
    # )

    print(f"\nPNGs guardados en '{CARPETA_SALIDA}/':")
    print(f"  - {PNG_COSTO}")
    print(f"  - {PNG_MATRIZ}")
    print(f"  - {PNG_RED}")

    # ── Guardar Respaldo del Modelo ─────────────────────────────────────────
    import pickle
    path_respaldo = os.path.join(CARPETA_SALIDA, "modelo_guardado.pkl")
    respaldo = {
        "model": model,
        "norm": norm,
        "le": le,
        "columnas": columnas,
        "accuracy": res_test['accuracy'],
        "preprocesador": preprocesador
    }
    with open(path_respaldo, "wb") as f:
        pickle.dump(respaldo, f)
    print(f"\n[RESPALDO] Modelo guardado con éxito en: {path_respaldo}")

    return model, norm, le, columnas, preprocesador


def busqueda_hiperparametros_3fold(X, y):
    """
    Grid search simple + 3-Fold CV sobre unos pocos hiperparámetros clave.
    Deliberadamente pequeño (para no disparar el tiempo de cómputo);
    amplía las listas si quieres una búsqueda más fina.
    """
    grid = {
        "alpha": [0.1, 0.01, 0.001],
        "batch_size": [16, 32],
        "momentum": [0.0, 0.8],
    }

    mejores = None
    mejor_score = -np.inf

    for alpha in grid["alpha"]:
        for bs in grid["batch_size"]:
            for mom in grid["momentum"]:
                scores = []
                for X_tr, X_val, y_tr, y_val in k_fold_split(X, y, k=3, seed=SEMILLA):
                    norm = Normalizer()
                    X_tr_n = norm.fit_transform(X_tr)
                    X_val_n = norm.transform(X_val)

                    m = Network(
                        layers=[
                            Dense(X_tr_n.shape[1], 128, "relu", seed=SEMILLA),
                            Dense(128, 64, "relu", seed=SEMILLA),
                            Dense(64, 32, "relu", seed=SEMILLA),
                            Dense(32, len(CLASES), "softmax", seed=SEMILLA),
                        ],
                        cost="categorical_crossentropy",
                    )
                    m.train(X_tr_n, y_tr, epochs=30, alpha=alpha,
                            batch_size=bs, momentum=mom, clip_norm=2.0,
                            verbose=0, seed=SEMILLA)
                    res = m.evaluate(X_val_n, y_val)
                    scores.append(res["accuracy"])

                score_prom = float(np.mean(scores))
                print(f"alpha={alpha}, batch_size={bs}, momentum={mom} "
                      f"-> acc_val_prom={score_prom:.4f}")

                if score_prom > mejor_score:
                    mejor_score = score_prom
                    mejores = {"alpha": alpha, "batch_size": bs, "momentum": mom}

    print(f"\nMejores hiperparámetros: {mejores} (acc={mejor_score:.4f})")
    return mejores


if __name__ == "__main__":
    main()
    # Para correr la búsqueda de hiperparámetros (3-Fold CV + grid search),
    # descomenta estas líneas (usa el dataset ya balanceado):
    #
    # df = cargar_y_balancear(RUTA_CSV, seed=SEMILLA)
    # X, y, columnas, le = preparar_features(df)
    # busqueda_hiperparametros_3fold(X, y)