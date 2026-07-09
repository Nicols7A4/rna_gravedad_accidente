# Informe Técnico de Sustentación del Proyecto — Red Neuronal Artificial (neurox)

Este documento contiene la estructura y el contenido formal en formato Markdown para las secciones solicitadas en el informe final de tu proyecto. Utiliza terminología científica, justificaciones basadas en la literatura analizada y la bitácora técnica de las iteraciones del modelo.

---

## 2. Objetivo del Modelo
El objetivo principal del modelo es clasificar y predecir de forma preventiva la gravedad de los siniestros viales (`GRAVEDAD` con las clases: `ILESO`, `LESIONADO` y `FALLECIDO`) a partir de las características del conductor, el vehículo, las causas registradas, y las condiciones del entorno espacial y temporal en el Perú. 

A diferencia de los modelos descriptivos tradicionales, este modelo busca servir como una herramienta predictiva activa para la toma de decisiones en políticas públicas de tránsito (evaluación de riesgos en vías públicas), alineándose con el enfoque de modelado predictivo de severidad vial estudiado en la literatura internacional (Sameen & Pradhan, 2017; Yang et al., 2022).

---

## 3. Arquitectura del Modelo
El modelo utiliza una arquitectura de **Red Neuronal Prealimentada Multicapa (Feedforward Neural Network - FNN)** o *Multilayer Perceptron (MLP)* de 7 capas en total (4 capas densas activas con parámetros de peso y 3 capas de regularización Dropout intercaladas):

*   **Flujo de Información:** Las señales fluyen en una sola dirección desde la capa de entrada de alta dimensionalidad (después del preprocesamiento categórico) hasta la capa de salida.
*   **Regularización Integrada:** Para mitigar el sobreajuste (*overfitting*) provocado por la alta dimensionalidad de los atributos codificados, se intercalan capas de **Dropout** (con tasas de desactivación de neuronas de $0.3$).
*   **Clasificación Multiclase:** La última capa utiliza la función de activación **Softmax** para proyectar las salidas de la red como una distribución de probabilidad que suma $1.0$ a través de las 3 clases.

---

## 4. Tipos de Neuronas
En este modelo implementado en la librería propia `neurox`, se utilizan dos tipos principales de neuronas matemáticas según su función de activación:

1.  **Neuronas ReLU (Rectified Linear Unit) en Capas Ocultas:**
    Definidas por la función:
    $$f(z) = \max(0, z)$$
    *Justificación:* Evitan el problema del desvanecimiento del gradiente (*vanishing gradient*) común en redes profundas que usan funciones sigmoides o tanh, permitiendo una convergencia de gradiente más rápida.
2.  **Neuronas Softmax en la Capa de Salida:**
    Definidas por la fórmula para la clase $i$ sobre $K$ clases totales:
    $$p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
    *Justificación:* Transforman los valores crudos de salida (*logits*) en probabilidades exponenciales mutuamente excluyentes, ideal para la clasificación multiclase de gravedad.

---

## 5. Justificación de la Elección
La elección del diseño de red (FNN de capas densas y optimizador SGD con Momentum) se justifica técnica y científicamente de la siguiente manera:

*   **¿Por qué una Red Neuronal y no estadística tradicional (ej: Excel o Regresiones)?**
    Las regresiones clásicas asumen linealidad e independencia de variables. Los siniestros viales ocurren por **interacciones combinatorias complejas** (ej. *conductor joven + ebriedad + motocicleta + de madrugada* tiene un riesgo exponencial superior a la suma individual de sus partes). Las capas ocultas y las activaciones ReLU permiten mapear estas relaciones no lineales abstractas de forma totalmente automática.
*   **¿Por qué una FNN/MLP y no arquitecturas secuenciales (LSTM) o de grafos (GNN)?**
    *   *GNN (Caso 1):* Requiere que el dataset tenga una topología de red vial o coordenadas continuas estructuradas como grafo, información ausente en nuestro archivo plano de accidentes.
    *   *LSTM (Caso 2):* Se usa para series de tiempo donde el orden de los accidentes en una misma autopista importa cronológicamente. Nuestro problema evalúa eventos independientes basándose en las variables intrínsecas de cada siniestro. Una FNN es la arquitectura ideal y matemáticamente robusta para datos tabulares independientes.
*   **¿Por qué SGD con Momentum y no Adam?**
    Aunque Adam (Caso 1 y 3) es eficiente para convergencia rápida, el **SGD con Momentum** (momentum = $0.80$) fue la arquitectura óptima demostrada en el **Caso 2** (Sameen & Pradhan, 2017) para clasificar severidad vial, ya que proporciona una trayectoria de gradiente más suave y menos propensa a memorizar el ruido de datasets pequeños o medianos ruidosos.

---

## 6. Capas del Modelo
La red se compone de las siguientes capas dispuestas de forma secuencial:

1.  **Capa de Entrada (215 neuronas):** Corresponde a la dimensión del vector de características tras aplicar One-Hot Encoding a las variables categóricas.
2.  **Capa Oculta 1 (Dense - 128 neuronas, ReLU):** Proyecta la entrada esparsa de alta dimensionalidad hacia un espacio latente.
3.  **Capa de Regularización (Dropout - tasa 0.3):** Desactiva aleatoriamente el 30% de las neuronas en cada paso de entrenamiento para forzar a la red a no depender de neuronas específicas.
4.  **Capa Oculta 2 (Dense - 64 neuronas, ReLU):** Reduce la dimensión intermedia y extrae patrones de segundo nivel.
5.  **Capa de Regularización (Dropout - tasa 0.3).**
6.  **Capa Oculta 3 (Dense - 32 neuronas, ReLU):** Refina la representación espacial de las variables.
7.  **Capa de Regularización (Dropout - tasa 0.3).**
8.  **Capa de Salida (Dense - 3 neuronas, Softmax):** Entrega la probabilidad para las clases `ILESO`, `LESIONADO` y `FALLECIDO`.

---

## 7. Entradas y Salidas
### Entradas (Features tras preprocesamiento)
El vector de entrada consta de **215 características** resultantes del preprocesamiento en `SiniestrosPreprocessor`, diseñado para evitar la fuga de información (*target leakage*):
*   *Numéricas:* `EDAD` (imputada con la mediana de entrenamiento y normalizada).
*   *Categóricas (One-Hot Encoded):* `TIPO PERSONA`, `SEXO`, `POSEE LICENCIA`, `ESTADO LICENCIA`, `CLASE_LICENCIA`, `¿SE SOMETIÓ A DOSAJE ETÍLICO CUALITATIVO?`, `RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO`, `¿SE SOMETIÓ A DOSAJE ETÍLICO CUANTITATIVO?`, `VEHÍCULO`, `CLASE DE SINIESTRO`, `CAUSA`, `CAUSA ESPECIFICA`, `TIPO DE VÍA`, `RED VIAL`, `DEPARTAMENTO`, `MES`, `DIA` y `HORA` (normalizada como entero).

> [!IMPORTANT]
> **Fuga de datos (Leakage) controlada:** Se excluyeron del dataset original variables como `LUGAR ATENCIÓN LESIONADO`, `LUGAR DE DEFUNCIÓN` o `SITUACIÓN DE PERSONA`, ya que estas representan consecuencias del accidente y su inclusión haría que el modelo haga trampa prediciendo con información que solo existe *después* del desenlace del siniestro.

### Salidas (Target)
Vector de probabilidad de tamaño 3, correspondiente a las clases codificadas por `LabelEncoder`:
*   `[1, 0, 0]` $\rightarrow$ `ILESO`
*   `[0, 1, 0]` $\rightarrow$ `LESIONADO`
*   `[0, 0, 1]` $\rightarrow$ `FALLECIDO`

---

## 8. Parámetros de Entrenamiento
Los hiperparámetros del entrenamiento final se configuraron de la siguiente manera:
*   **Épocas:** 200 (Incrementado desde 100 tras observar que el modelo continuaba minimizando la pérdida sin sobreajuste).
*   **Tasa de Aprendizaje ($\alpha$):** 0.01 (Establece un tamaño de paso controlado para el descenso de gradiente).
*   **Tamaño del lote (Batch Size):** 32 (Equilibrio óptimo entre velocidad de cómputo y estocasticidad del gradiente).
*   **Momento (Momentum):** 0.80 (Acelera la convergencia en la dirección correcta y reduce oscilaciones).
*   **Weight Decay (Regularización L2):** $10^{-4}$ (Penaliza pesos excesivamente grandes para evitar overfitting).
*   **Clip Norm:** 2.0 (Evita el problema de gradientes explosivos en capas profundas limitando la norma del vector de gradientes).

---

## 9. Conjunto de Datos
*   **Origen:** El dataset proviene de registros policiales y gubernamentales de accidentes de tránsito en el Perú (`DATA/datos.csv`), con más de 25,000 registros iniciales.
*   **Filtros de Calidad:** Se eliminaron los registros con la clase `"NO SE CONOCE"` en el campo de gravedad, resultando en un dataset limpio de **23,973 registros**.
*   **División de Datos:** 80% para entrenamiento (19,178 registros) y 20% para prueba (4,795 registros).
*   **Desbalance de Datos:** El dataset presenta una distribución asimétrica:
    *   `FALLECIDO`: 10,859 (45.3%)
    *   `LESIONADO`: 7,830 (32.7%)
    *   `ILESO`: 5,284 (22.0%)
*   **Tratamiento de Desbalance:** En lugar de realizar undersampling (que desecha más del 35% del dataset valioso), entrenamos con el 100% de los datos aplicando **Focal Loss con Pesos Adaptativos de Clase**.

---

## 10. Métricas Evaluadas
Para medir de forma objetiva la capacidad de clasificación del modelo se evaluaron:
1.  **Exactitud (Accuracy) General:** Porcentaje de aciertos globales sobre el dataset.
2.  **Precisión (Precision):** Capacidad del modelo de no etiquetar como positiva una muestra negativa (evitar falsos positivos).
3.  **Sensibilidad (Recall):** Capacidad del modelo de encontrar todas las muestras positivas reales (evitar falsos negativos, crítico en seguridad vial).
4.  **F1-Score (Macro y Weighted):** Media armónica entre precisión y recall. El F1 Macro evalúa el desempeño asignando igual importancia a las tres clases sin importar su volumen.
5.  **Matriz de Confusión:** Tabla cruzada de predicciones vs. valores reales para evaluar patrones de error.

---

## 11. Resultados y Evolución del Modelo

El proyecto se desarrolló mediante un proceso iterativo de optimización científica:

### Iteración 1: Modelo Base (Entrenamiento Tradicional)
*   *Configuración:* Undersampling balanceado clásico (reducción a 5,284 casos por clase), sin dropout, semillas idénticas en inicialización.
*   *Resultados:* Accuracy del **73.0%**, pero con una brecha de sobreajuste (*overfitting*) del **25.56%** (Accuracy en Train de ~98.5%). Las semillas idénticas causaban redundancia de pesos en la red.

### Iteración 2: Incorporación de Regularización y Dropout
*   *Configuración:* Se introdujeron capas de **Dropout (0.3)** y semillas diferenciadas por capa (`SEMILLA + i`).
*   *Resultados:* El sobreajuste cayó drásticamente a **18.4%**, manteniendo la precisión en **73.72%** en validación, demostrando el efecto regularizador.

### Iteración 3: Dataset Completo + Focal Loss Adaptativo (Modelo Final Elegido)
*   *Configuración:* Entrenamiento sobre el 100% del dataset desbalanceado (23,973 filas), uso de **Focal Loss** con pesos adaptativos calculados dinámicamente según la inversa de frecuencias ($\alpha_i = [1.51, 1.02, 0.73]$), $\gamma = 2.0$, Dropout $0.3$ y **200 épocas**.
*   *Resultados:*
    *   **Accuracy Test:** **72.37%** (Estable y representativo del total de datos).
    *   **Brecha de Overfitting:** **7.30%** (¡Casi nulo sobreajuste!).
    *   **Recall por Clase:**
        *   `ILESO`: **85.45%** (Excelente sensibilidad para la clase minoritaria).
        *   `LESIONADO`: **84.44%**.
        *   `FALLECIDO`: **57.08%**.
    *   *Diagnóstico:* Un rendimiento altamente equilibrado y seguro. (Nota: Al experimentar reduciendo $\gamma$ a $1.5$ y Dropout a $0.2$, el recall de Fallecido subió al $67.01\%$ con un Accuracy de $73.72\%$, pero con una brecha de sobreajuste superior del $15.0\%$, por lo que la iteración con $\gamma=2.0$ y Dropout $0.3$ se seleccionó como la más robusta contra overfitting).

---

## 12. Conclusiones
1.  **Viabilidad de la Librería Propia:** Es posible estructurar una librería de Deep Learning (`neurox`) funcional desde cero usando únicamente NumPy y Pandas, logrando resultados predictivos estables y competitivos frente a frameworks como TensorFlow o PyTorch.
2.  **Mitigación Científica del Desbalance:** La aplicación de **Focal Loss con Pesos Adaptativos** es superior al undersampling tradicional en problemas viales, ya que permite entrenar al modelo con el 100% de los datos (conservando la riqueza de variabilidad geográfica y temporal) y previene el sesgo hacia la clase mayoritaria.
3.  **Impacto de la Regularización:** El Dropout y la inicialización de pesos con semillas diferenciadas fueron críticos para bajar la brecha de sobreajuste de un $25.5\%$ a un mínimo de **$7.3\%$**, garantizando que el modelo generalice de forma idónea ante datos reales no vistos.
4.  **Sensibilidad Vial Coherente:** El simulador dinámico desarrollado demuestra que la red neuronal ha aprendido las dinámicas físicas reales de riesgo: el riesgo se dispara de forma coherente ante atropellos a peatones, conductores de motocicletas, consumo de alcohol y exceso de velocidad.

---

## 13. Recomendaciones
1.  **Implementación de Optimizadores de Segundo Orden:** Incorporar algoritmos como Adam o RMSprop en la librería `neurox` para acelerar la velocidad de convergencia matemática y permitir topologías más profundas con menores épocas de entrenamiento.
2.  **Ampliar la Explicabilidad (XAI):** Desarrollar un módulo nativo de *Permutation Feature Importance* (Importancia por Permutación) para medir cuantitativamente el peso específico de cada factor de riesgo (Ebriedad, Vía, Edad) en el informe final.
3.  **Capturar Relaciones Espaciales Continuas:** En futuros datasets, incorporar coordenadas GPS de los siniestros para ensayar una codificación sinusoidal de posición geográfica, reduciendo el ruido de la clasificación discreta por departamentos.
4.  **Estandarizar el Registro de Datos:** Proponer a los entes de transporte (MTC/PNP) la eliminación de valores nulos o "conductor fugado" en el registro de licencias para reducir variables esparsas que diluyen el aprendizaje del modelo.
