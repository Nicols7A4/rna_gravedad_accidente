"""
app.py
======
Servidor web interactivo desarrollado en Flask.
Proporciona la interfaz del Simulador Vial y expone endpoints de predicción dinámica API JSON.
Carga eficientemente el pipeline pre-entrenado guardado en un archivo Pickle (.pkl) en menos de 1 segundo.

Flujo de Operación:
------------------
1. Inicialización: Carga del modelo persistido o re-entrenamiento en caso de ausencia.
2. Ruteo Flask:
   - Redirección por defecto '/' a '/simulator'.
   - '/simulator': Renderizado del Simulador Básico (parámetros preestablecidos).
   - '/simulator-complete': Renderizado del Simulador Completo (los 16 selectores interactivos).
   - '/api/predict' [POST]: Endpoint REST para predicciones en tiempo real mediante el preprocesador memorizado.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import pandas as pd
import json
import os
import pickle
from neurox import Network, Dense, Dropout, Normalizer, LabelEncoder, train_test_split
from prueba_entrenamiento import (
    CLASES, FEATURES_NUMERICAS, FEATURES_CATEGORICAS,
    cargar_y_balancear, RUTA_CSV, SEMILLA,
    COLUMNAS_ALTA_CARDINALIDAD, MIN_FRECUENCIA, CLASS_WEIGHTS,
    SiniestrosPreprocessor, codificar_target
)

app = Flask(__name__)

# Diccionario estructurado en memoria para contener los componentes del pipeline de inferencia
modelo_datos = {
    'model': None,          # Instancia de neurox.Network
    'norm': None,           # Instancia de neurox.Normalizer
    'le': None,             # LabelEncoder para decodificar las clases de salida
    'columnas': None,       # Estructura e indexado One-Hot
    'accuracy': None,       # Precisión histórica de prueba obtenida
    'preprocesador': None   # Instancia de SiniestrosPreprocessor ajustada en entrenamiento
}


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y SERIALIZACIÓN DE PIPELINE DE INFERENCIA
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo():
    """
    Carga de forma perezosa (lazy load) el modelo y preprocesadores.
    Si existe outputs/modelo_guardado.pkl, carga el archivo binario instantáneamente.
    Si no existe, ejecuta el entrenamiento desde cero antes de arrancar Flask.
    """
    try:
        if modelo_datos['model'] is not None:
            return modelo_datos
        
        path_respaldo = os.path.join("outputs", "modelo_guardado.pkl")
        
        # 1. Intentar cargar respaldo Pickle
        if os.path.exists(path_respaldo):
            print(f"[RESPALDO] Cargando modelo guardado de: {path_respaldo}...")
            with open(path_respaldo, "rb") as f:
                respaldo = pickle.load(f)
            modelo_datos['model'] = respaldo['model']
            modelo_datos['norm'] = respaldo['norm']
            modelo_datos['le'] = respaldo['le']
            modelo_datos['columnas'] = respaldo['columnas']
            modelo_datos['accuracy'] = respaldo['accuracy']
            modelo_datos['preprocesador'] = respaldo['preprocesador']
            print("[RESPALDO] ¡Modelo cargado exitosamente en menos de 1 segundo!")
            return modelo_datos
        
        # 2. Re-entrenamiento automático si no existe el respaldo
        print("[RESPALDO] No se encontró modelo guardado. Entrenando uno nuevo...")
        balancear = False
        usar_focal_loss = True
        
        df = cargar_y_balancear(RUTA_CSV, seed=SEMILLA, balancear=balancear)
        
        # Split Train/Test aleatorio para validación
        df_train = df.sample(frac=0.8, random_state=None)
        df_test = df.drop(df_train.index).reset_index(drop=True)
        df_train = df_train.reset_index(drop=True)
        
        # Preprocesador
        preprocesador = SiniestrosPreprocessor()
        preprocesador.fit(df_train)
        
        X_train = preprocesador.transform(df_train)
        X_test = preprocesador.transform(df_test)
        
        # Target
        le = LabelEncoder()
        le.classes = np.array(CLASES)
        le._mapping = {c: i for i, c in enumerate(CLASES)}
        le._inverse = {i: c for c, i in le._mapping.items()}
        
        y_train = codificar_target(df_train, le)
        y_test = codificar_target(df_test, le)
        
        # Coste y pesos sensibles al costo
        if usar_focal_loss and not balancear:
            class_counts = np.sum(y_train, axis=0)
            total_samples = y_train.shape[0]
            k_classes = y_train.shape[1]
            class_counts[class_counts == 0] = 1
            adaptive_alpha = total_samples / (k_classes * class_counts)
            print(f"Pesos adaptativos de clase calculados: {adaptive_alpha}")
            print(f"Pesos ajustados de clase calculados: {CLASS_WEIGHTS}")
            class_weights = adaptive_alpha
            class_weights = CLASS_WEIGHTS
            cost_function = "focal_loss"
        else:
            class_weights = None
            cost_function = "categorical_crossentropy"
            
        columnas = preprocesador.columns_ohe_
        
        # Normalización
        norm = Normalizer()
        X_train = norm.fit_transform(X_train)
        X_test = norm.transform(X_test)
        
        # Red neuronal
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
        
        # Entrenamiento corto de respaldo
        print("Entrenando modelo...")
        model.train(
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
        
        # Evaluación
        res_test = model.evaluate(X_test, y_test)
        accuracy = res_test['accuracy']
        
        # Almacenar en diccionario global
        modelo_datos['model'] = model
        modelo_datos['norm'] = norm
        modelo_datos['le'] = le
        modelo_datos['columnas'] = columnas
        modelo_datos['accuracy'] = accuracy
        modelo_datos['preprocesador'] = preprocesador
        
        # Guardar en outputs
        try:
            os.makedirs("outputs", exist_ok=True)
            with open(path_respaldo, "wb") as f:
                pickle.dump(modelo_datos, f)
            print(f"[RESPALDO] Nuevo modelo guardado con éxito en: {path_respaldo}")
        except Exception as e:
            print(f"[RESPALDO] No se pudo guardar el modelo: {e}")
            
        return modelo_datos
    
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS FLASK (CONTROLADORES)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Redirecciona al simulador dinámico por defecto."""
    return redirect(url_for('simulator'))


@app.route('/api/info')
def api_info():
    """Retorna metadatos técnicos en JSON sobre el modelo cargado."""
    if modelo_datos['model'] is None:
        cargar_modelo()
    
    return jsonify({
        'accuracy': float(modelo_datos['accuracy']) if modelo_datos['accuracy'] else None,
        'num_features': len(modelo_datos['columnas']) if modelo_datos['columnas'] else None,
        'features_numericas': FEATURES_NUMERICAS,
        'features_categoricas': FEATURES_CATEGORICAS,
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Inferencia Individual en tiempo real.
    Recibe un JSON con los valores del formulario, aplica el preprocesamiento
    alineado (one-hot) y normalización del set de entrenamiento, y calcula
    las probabilidades Softmax de cada clase.
    """
    try:
        datos = request.get_json()
        
        if modelo_datos['model'] is None:
            cargar_modelo()
        
        # Crear DataFrame temporal a partir de los datos JSON del formulario
        df_input = pd.DataFrame([datos])
        
        # Transformar alineando One-Hot sin Data Leakage
        X_input = modelo_datos['preprocesador'].transform(df_input)
        X_input = modelo_datos['norm'].transform(X_input)
        
        # Ejecutar propagación hacia adelante (forward pass) de la red
        y_pred = modelo_datos['model'].predict(X_input)
        pred_clase = int(np.argmax(y_pred[0]))
        pred_label = CLASES[pred_clase]
        pred_confianza = float(y_pred[0][pred_clase])
        
        # Mapeo de probabilidades
        probabilidades = {CLASES[i]: float(y_pred[0][i]) for i in range(len(CLASES))}
        
        return jsonify({
            'success': True,
            'prediccion': pred_label,
            'confianza': pred_confianza,
            'probabilidades': probabilidades,
            'clase_idx': pred_clase
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    """Inferencia por lotes (Batch prediction)."""
    try:
        datos = request.get_json()
        registros = datos.get('registros', [])
        
        if not registros:
            return jsonify({'success': False, 'error': 'No hay registros'}), 400
        
        if modelo_datos['model'] is None:
            cargar_modelo()
        
        df_input = pd.DataFrame(registros)
        X = modelo_datos['preprocesador'].transform(df_input)
        X = modelo_datos['norm'].transform(X)
        
        # Batch Predict
        y_pred = modelo_datos['model'].predict(X)
        
        resultados = []
        for i, pred in enumerate(y_pred):
            pred_clase = int(np.argmax(pred))
            pred_label = CLASES[pred_clase]
            pred_confianza = float(pred[pred_clase])
            
            resultados.append({
                'indice': i,
                'prediccion': pred_label,
                'confianza': pred_confianza,
                'probabilidades': {CLASES[j]: float(pred[j]) for j in range(len(CLASES))}
            })
        
        return jsonify({
            'success': True,
            'total': len(resultados),
            'resultados': resultados
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/simulator')
def simulator():
    """Renderiza el Simulador de Riesgos Básico (controles sliders simplificados)."""
    return render_template('simulator.html', clases=CLASES)


@app.route('/simulator-complete')
def simulator_complete():
    """Renderiza el Simulador de Riesgos Completo (con 16 selectores interactivos)."""
    return render_template('simulator_complete.html', clases=CLASES)


# ─────────────────────────────────────────────────────────────────────────────
# EXECUCIÓN DEL SERVIDOR LOCAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("[CAR] PREDICTOR DE GRAVEDAD DE SINIESTROS - Interfaz Web")
    print("=" * 70)
    print("\n[LOADING] Cargando modelo de red neuronal...")
    cargar_modelo()
    print("[OK] Modelo cargado correctamente\n")
    print("[WEB] Abre tu navegador en: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, port=5000, use_reloader=False)
