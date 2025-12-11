## Conexión con Proyecto Final

Los experimentos realizados en este proyecto evaluaron distintas arquitecturas de redes neuronales para predecir la demanda de productos en base a múltiples variables (edad del cliente, historial de compras, promociones, condiciones climáticas, etc.).

A partir de los resultados obtenidos:

- La función de activación **Tanh** mostró el menor error promedio para redes con múltiples capas ocultas, siendo más estable para la predicción.
- Las redes con una sola capa oculta (8 o 16 neuronas) presentaron un desempeño aceptable y eficiente en tiempo de entrenamiento.
- Las redes con dos capas ocultas tienden a sobreajustarse, especialmente usando ReLU.

Por lo tanto, para el Proyecto Final se recomienda implementar una red con:
- **Arquitectura:** 1 capa oculta de 8 a 16 neuronas
- **Función de activación:** Tanh
- **Tasa de aprendizaje y épocas:** 0.01 y 300 épocas (según mejores resultados)

Esta elección optimiza la precisión de predicción y mantiene la eficiencia computacional.
