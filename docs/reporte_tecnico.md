# Reporte Técnico  
## Implementación de Redes Neuronales
**Curso:** Inteligencia Artificial  
**Grupo:** 4  

---

## 1. Introducción

El presente reporte técnico documenta el desarrollo e implementación de una **red neuronal artificial feedforward** construida **desde cero utilizando únicamente Python y NumPy**

El objetivo principal del trabajo es demostrar la comprensión profunda de los **fundamentos matemáticos y computacionales** de las redes neuronales, evitando el uso de librerías de alto nivel como TensorFlow o PyTorch, y aplicando el modelo a un problema práctico definido por el grupo.

---

## 2. Planteamiento del problema

En muchos problemas reales de análisis de datos y toma de decisiones, los modelos lineales tradicionales resultan insuficientes para capturar relaciones no lineales entre variables.  
Las redes neuronales artificiales permiten modelar este Considerable grado de complejidad mediante capas ocultas y funciones de activación no lineales.

El problema abordado en este proyecto consiste en **predecir una variable objetivo continua** a partir de un conjunto de variables de entrada, utilizando una red neuronal entrenada mediante descenso del gradiente.

---

## 3. Dataset y preprocesamiento

El dataset utilizado fue cargado y analizado en los notebooks del proyecto.  
Las principales etapas de preprocesamiento incluyeron:

- Revisión de valores faltantes
- Normalización / estandarización de variables numéricas
- Separación del conjunto de datos en:
  - Conjunto de entrenamiento
  - Conjunto de prueba

---

## 4. Arquitectura de la red neuronal

La red neuronal implementada corresponde a un **Multilayer Perceptron (MLP)** con las siguientes características:

- **Capa de entrada:**  
  Número de neuronas igual al número de variables independientes.

- **Capas ocultas:**  
  Se implementaron **al menos dos capas ocultas**, cumpliendo con los requisitos de la rúbrica.

- **Capa de salida:**  
  Una neurona con activación lineal, adecuada para problemas de regresión.

### 4.1 Funciones de activación

Se implementaron y evaluaron las siguientes funciones:

- ReLU (Rectified Linear Unit)
- Sigmoid
- Tanh

La selección de la función de activación influye directamente en la convergencia y desempeño del modelo.

---

## 5. Implementación desde cero

La implementación fue realizada completamente desde cero e incluye:

### 5.1 Forward Propagation

- Cálculo del potencial neuronal:  
  `Z = XW + b`
- Aplicación de la función de activación
- Propagación de la información capa por capa

### 5.2 Función de pérdida

Se utilizó el **Error Cuadrático Medio (MSE)** como función de pérdida:

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Esta función es adecuada para problemas de regresión y penaliza fuertemente errores grandes.

### 5.3 Backpropagation

- Cálculo del gradiente del error respecto a pesos y sesgos
- Aplicación de la regla de la cadena
- Propagación del error desde la capa de salida hacia atrás

### 5.4 Optimización

Se utilizó **descenso del gradiente** con una tasa de aprendizaje definida experimentalmente.

---

## 6. Métricas de evaluación

Para evaluar el desempeño del modelo se utilizaron las siguientes métricas:

- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (%)**, cuando fue aplicable

Además, se comparó el modelo propuesto contra un **modelo baseline**, con el fin de validar que la red neuronal aporta una mejora real.

---

## 7. Resultados

Los resultados obtenidos muestran:

- Convergencia progresiva del error durante el entrenamiento
- Mejora del desempeño frente al baseline
- Capacidad del modelo para capturar relaciones no lineales

Se generaron gráficos de:
- Curva de pérdida (loss vs epochs)
- Comparación entre valores reales y valores predichos

Los resultados completos se encuentran en la carpeta `results/`.

---

## 8. Análisis y discusión

El modelo implementado demuestra que una red neuronal correctamente configurada puede mejorar significativamente el desempeño predictivo frente a métodos simples.

Sin embargo, se identifican posibles mejoras:
- Ajuste fino de hiperparámetros
- Mayor volumen de datos
- Regularización para evitar sobreajuste

---

## 9. Conclusiones

- Se logró implementar exitosamente una red neuronal desde cero.
- El modelo cumple con todos los requisitos técnicos solicitados.
- La experiencia permitió comprender a profundidad los fundamentos matemáticos de las redes neuronales.
- El enfoque “from scratch” fortalece el entendimiento más allá del uso de librerías de alto nivel.

---

## 10. Trabajo futuro

Como extensiones futuras se propone:

- Implementar optimizadores avanzados (Momentum, Adam)
- Incorporar regularización L1/L2
- Probar arquitecturas más profundas
- Integrar validación cruzada

---

## 11. Integrantes – Grupo 4

- Arnaldo Andrés Rojas Jupiter  
- Andres Asisclo Florencia Toala 
- Denisse Angie Flores Arellano  
- Boris Ricardo Tigre Loja

---

## 12. Observaciones finales

Este proyecto tiene fines estrictamente académicos y forma parte de la evaluación del curso de Inteligencia Artificial de la UEES.
