# Tarea – Redes Neuronales (Grupo 4)

Este repositorio contiene la **implementación desde cero de una red neuronal feedforward**, desarrollada en Python utilizando únicamente **NumPy**, como parte de la **tarea de Redes Neuronales** del curso de **Inteligencia Artificial** en la **UEES**.

## Integrantes – Grupo 4

- Arnaldo Andrés Rojas Jupiter  
- Andres Asisclo Florencia Toala 
- Denisse Angie Flores Arellano  
- Boris Ricardo Tigre Loja

---

## Objetivo del proyecto

- Implementar una red neuronal artificial **sin utilizar frameworks de deep learning** (TensorFlow, PyTorch, Keras).
- Programar manualmente:
  - Forward propagation
  - Backpropagation
  - Actualización de pesos con gradiente descendente
- Evaluar el desempeño del modelo usando métricas adecuadas.
- Comparar los resultados contra un **modelo baseline**.
- Analizar los resultados en el contexto del problema planteado.

---

## Estructura del repositorio

- **data/**  
  Contiene los datasets utilizados para entrenamiento y prueba.

- **notebooks/**  
  Notebooks de Jupyter con:
  - Exploración y preprocesamiento de datos
  - Entrenamiento de la red neuronal
  - Evaluación y visualización de resultados

- **src/**  
  Código fuente reutilizable:
  - Implementación de la red neuronal
  - Funciones de activación
  - Función de pérdida
  - Métricas y utilidades

- **results/**  
  Resultados del entrenamiento:
  - Métricas finales
  - Gráficos
  - Comparaciones con el baseline

- **docs/**  
  Documentación del proyecto y conclusiones.

- **requirements.txt**  
  Librerías necesarias para ejecutar el proyecto.

---

## Requisitos

- Python 3.9 o superior (recomendado 3.10+)
- pip o conda
- Jupyter Notebook

---

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/boristigre-uees/tarea-redes-neuronales-grupo4.git
cd tarea-redes-neuronales-grupo4

