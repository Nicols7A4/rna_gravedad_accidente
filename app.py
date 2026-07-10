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
        
        # 1. Intentar cargar respaldo pre-entrenado
        import os
        import pickle
        path_respaldo = os.path.join("outputs", "modelo_guardado.pkl")
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
        
        print("[RESPALDO] No se encontró modelo guardado. Entrenando uno nuevo...")
        # Configuración de balanceo y pérdida (Propuesta B)
        balancear = False
        usar_focal_loss = True
        
        # Cargar y balancear datos
        df = cargar_y_balancear(RUTA_CSV, seed=SEMILLA, balancear=balancear)
        
        # Separar datos para evitar data leakage (aleatorio sin semilla fija)
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
            class_weights = None
            cost_function = "categorical_crossentropy"
            
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
            cost=cost_function,
            class_weights=class_weights,
            gamma=2.0,
        )
        
        print("Entrenando modelo...")
        # Entrenar
        model.train(
            X_train, y_train,
            epochs=200,
            alpha=0.01,
            batch_size=32,
            momentum=0.80, # Ignorado por Adam
            weight_decay=1e-4,
            clip_norm=2.0,
            seed=SEMILLA,
            optimizer='adam',
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
        
        # Guardar respaldo para evitar re-entrenamiento posterior
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

@app.route('/simulator')
def simulator():
    """Página del simulador interactivo de riesgo vial"""
    return render_template('simulator.html', clases=CLASES)


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
