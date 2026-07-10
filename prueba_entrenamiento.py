import os
import datetime
import io
import pickle
from contextlib import redirect_stdout
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
CLASES = ["ILESO", "LESIONADO", "FALLECIDO"]

CARPETA_SALIDA = "outputs"
PNG_COSTO = f"{CARPETA_SALIDA}/curva_costo.png"
PNG_MATRIZ = f"{CARPETA_SALIDA}/matriz_confusion.png"
PNG_RED = f"{CARPETA_SALIDA}/arquitectura_red.png"

CLASS_WEIGHTS = [1.4, 1.0, 0.9]

def cargar_y_balancear(ruta_csv: str, seed: int = SEMILLA, balancear: bool = True) -> pd.DataFrame:

    df = pd.read_csv(ruta_csv, encoding="utf-8")

    df = df[df["GRAVEDAD"].isin(CLASES)].copy()

    if not balancear:
        print(f"Dataset completo (sin balancear): {len(df)} filas totales")
        print(df["GRAVEDAD"].value_counts())
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_min = df["GRAVEDAD"].value_counts().min()
    rng = np.random.default_rng(seed)

    partes = []
    for clase in CLASES:
        sub = df[df["GRAVEDAD"] == clase]
        idx = rng.choice(sub.index, size=n_min, replace=False)
        partes.append(sub.loc[idx])

    df_bal = pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Dataset balanceado (método A): {n_min} casos x {len(CLASES)} clases = {len(df_bal)} filas totales")
    print(df_bal["GRAVEDAD"].value_counts())
    return df_bal

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

COLUMNAS_ALTA_CARDINALIDAD = ["CAUSA", "CAUSA ESPECIFICA", "VEHÍCULO"]
MIN_FRECUENCIA = 30

class SiniestrosPreprocessor:

    def __init__(self, min_frecuencia=MIN_FRECUENCIA):
        self.min_frecuencia = min_frecuencia
        self.frequent_categories_ = {}
        self.median_edad_ = None
        self.columns_ohe_ = None

    def fit(self, df: pd.DataFrame):

        edad_num = pd.to_numeric(df["EDAD"], errors="coerce")
        self.median_edad_ = edad_num.median()
        if pd.isna(self.median_edad_):
            self.median_edad_ = 35.0

        for col in COLUMNAS_ALTA_CARDINALIDAD:
            if col in df.columns:
                series = df[col].fillna("DESCONOCIDO").astype(str)
                counts = series.value_counts()
                frequent = counts[counts >= self.min_frecuencia].index.tolist()
                self.frequent_categories_[col] = set(frequent)

        df_clean = self._preprocess_basic(df, fitting=True)
        df_ohe = pd.get_dummies(df_clean[FEATURES_CATEGORICAS], drop_first=False)
        X_temp = pd.concat([df_clean[FEATURES_NUMERICAS], df_ohe], axis=1)
        self.columns_ohe_ = X_temp.columns.tolist()
        return self

    def _preprocess_basic(self, df: pd.DataFrame, fitting=False) -> pd.DataFrame:

        df = df.copy()

        for col in FEATURES_CATEGORICAS:
            if col in df.columns:
                df[col] = df[col].fillna("DESCONOCIDO").astype(str)
            else:
                df[col] = "DESCONOCIDO"

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

        for col in COLUMNAS_ALTA_CARDINALIDAD:
            if col in df.columns:
                if col in self.frequent_categories_:
                    df[col] = df[col].apply(lambda x: x if x in self.frequent_categories_[col] else "OTRO")
                else:
                    counts = df[col].value_counts()
                    raras = counts[counts < self.min_frecuencia].index
                    df[col] = df[col].where(~df[col].isin(raras), other="OTRO")

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

        for col in self.columns_ohe_:
            if col not in X.columns:
                X[col] = 0

        X = X[self.columns_ohe_]
        return X.to_numpy(dtype=float)

def codificar_target(df: pd.DataFrame, le: LabelEncoder) -> np.ndarray:

    y_idx = le.transform(df["GRAVEDAD"].values)
    return one_hot(y_idx, n_classes=len(CLASES))

def preparar_features(df: pd.DataFrame):

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

def main():
    import os
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    balancear = False
    usar_focal_loss = True

    df = cargar_y_balancear(RUTA_CSV, seed=SEMILLA, balancear=balancear)

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

    if usar_focal_loss and not balancear:
        class_counts = np.sum(y_train, axis=0)
        total_samples = y_train.shape[0]
        k_classes = y_train.shape[1]
        class_counts[class_counts == 0] = 1
        adaptive_alpha = total_samples / (k_classes * class_counts)
        print(f"Pesos adaptativos por frecuencia: {adaptive_alpha}")
        print(f"Pesos manuales (sensibles al costo): {CLASS_WEIGHTS}")

        class_weights = CLASS_WEIGHTS
        cost_function = "focal_loss"
    else:
        class_weights = CLASS_WEIGHTS
        cost_function = "categorical_crossentropy"

    columnas = preprocesador.columns_ohe_
    print(f"\nDimensión final de entrada (features tras one-hot): {X_train.shape[1]}")

    norm = Normalizer()
    X_train = norm.fit_transform(X_train)
    X_test = norm.transform(X_test)

    print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

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

    print("\n[ENTRENAMIENTO] Iniciando entrenamiento con optimizador ADAM...")
    historial_costo = model.train(
        X_train, y_train,
        epochs=100,
        alpha=0.001,
        batch_size=64,
        momentum=0.80,
        weight_decay=1e-4,
        clip_norm=2.0,
        seed=SEMILLA,
        optimizer='adam',
        verbose=10,
    )

    model.plot_cost(title="Costo — Método A (Focal Loss)", savepath=PNG_COSTO)

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
        print(f"  - Épocas: 100")
        print(f"  - Tasa de Aprendizaje (alpha): 0.001")
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
    print(report_str)

    path_log = os.path.join(CARPETA_SALIDA, "log_entrenamiento.txt")
    with open(path_log, "w", encoding="utf-8") as f_log:
        f_log.write(report_str)
    print(f"\n[REGISTRO] Log del entrenamiento guardado en: {path_log}")

    plot_confusion_matrix(y_true_cls, y_pred_cls, class_names=CLASES,
                           title="Matriz de Confusión — Método A",
                           savepath=PNG_MATRIZ)

    print(f"\nPNGs guardados en '{CARPETA_SALIDA}/':")
    print(f"  - {PNG_COSTO}")
    print(f"  - {PNG_MATRIZ}")
    print(f"  - {PNG_RED}")

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
                print(f"alpha={alpha}, batch_size={bs}, momentum={mom} -> acc_val_prom={score_prom:.4f}")

                if score_prom > mejor_score:
                    mejor_score = score_prom
                    mejores = {"alpha": alpha, "batch_size": bs, "momentum": mom}

    print(f"\nMejores hiperparámetros: {mejores} (acc={mejor_score:.4f})")
    return mejores

if __name__ == "__main__":
    main()
