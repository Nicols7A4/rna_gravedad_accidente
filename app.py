"""
app.py
======
Aplicación Flask para predicción de gravedad de siniestros.

Uso:
    python app.py
    Luego abre: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
from neurox import Network, Dense, Dropout, Normalizer, LabelEncoder, train_test_split
from prueba_entrenamiento import (
    CLASES, FEATURES_NUMERICAS, FEATURES_CATEGORICAS,
    cargar_y_balancear, RUTA_CSV, SEMILLA,
    COLUMNAS_ALTA_CARDINALIDAD, MIN_FRECUENCIA,
    SiniestrosPreprocessor, codificar_target
)

app = Flask(__name__)

# Variables globales para almacenar el modelo
modelo_datos = {
    'model': None,
    'norm': None,
    'le': None,
    'columnas': None,
    'accuracy': None,
    'preprocesador': None
}

# ─────────────────────────────────────────────────────────────────────────────
# Funciones para cargar el modelo
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo():
    """Carga o entrena el modelo de red neuronal."""
    try:
        if modelo_datos['model'] is not None:
            return modelo_datos
        
        # Cargar y balancear datos
        df = cargar_y_balancear(RUTA_CSV, seed=SEMILLA)
        
        # Separar datos para evitar data leakage
        df_train = df.sample(frac=0.8, random_state=SEMILLA)
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
        
        columnas = preprocesador.columns_ohe_
        
        # Normalización
        norm = Normalizer()
        X_train = norm.fit_transform(X_train)
        X_test = norm.transform(X_test)
        
        # Construir modelo con Dropout y semillas diferentes
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
            cost="categorical_crossentropy",
            class_weights=None,
        )
        
        print("Entrenando modelo...")
        # Entrenar
        model.train(
            X_train, y_train,
            epochs=100,
            alpha=0.01,
            batch_size=32,
            momentum=0.80,
            weight_decay=1e-4,
            clip_norm=2.0,
            seed=SEMILLA,
            verbose=10,
        )
        
        # Evaluación
        res_test = model.evaluate(X_test, y_test)
        accuracy = res_test['accuracy']
        
        # Guardar en variables globales
        modelo_datos['model'] = model
        modelo_datos['norm'] = norm
        modelo_datos['le'] = le
        modelo_datos['columnas'] = columnas
        modelo_datos['accuracy'] = accuracy
        modelo_datos['preprocesador'] = preprocesador
        
        return modelo_datos
    
    except FileNotFoundError:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Rutas de la aplicación
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', clases=CLASES)

@app.route('/api/info')
def api_info():
    """Retorna información del modelo"""
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
    """Realiza predicción individual"""
    try:
        datos = request.get_json()
        
        # Asegurarse de que el modelo está cargado
        if modelo_datos['model'] is None:
            cargar_modelo()
        
        # Crear DataFrame con los datos ingresados
        df_input = pd.DataFrame([datos])
        
        # Usar el preprocesador entrenado sin data leakage ni bug de categorías raras
        X_input = modelo_datos['preprocesador'].transform(df_input)
        
        # Normalizar
        X_input = modelo_datos['norm'].transform(X_input)
        
        # Predicción
        y_pred = modelo_datos['model'].predict(X_input)
        pred_clase = int(np.argmax(y_pred[0]))
        pred_label = CLASES[pred_clase]
        pred_confianza = float(y_pred[0][pred_clase])
        
        # Probabilidades de todas las clases
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
    """Realiza predicciones en lote"""
    try:
        datos = request.get_json()
        registros = datos.get('registros', [])
        
        if not registros:
            return jsonify({'success': False, 'error': 'No hay registros'}), 400
        
        # Asegurarse de que el modelo está cargado
        if modelo_datos['model'] is None:
            cargar_modelo()
        
        # Crear DataFrame
        df_input = pd.DataFrame(registros)
        
        # Usar el preprocesador entrenado sin data leakage ni bug de categorías raras
        X = modelo_datos['preprocesador'].transform(df_input)
        
        # Normalizar
        X = modelo_datos['norm'].transform(X)
        
        # Predicciones
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

# ─────────────────────────────────────────────────────────────────────────────
# Iniciar servidor
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
