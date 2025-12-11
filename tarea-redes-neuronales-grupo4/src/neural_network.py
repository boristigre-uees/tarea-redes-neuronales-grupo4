# src/experiments.py
# src/neural_network.py
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation="relu"):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def _activate(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unknown activation")

    def forward(self, X):
        out = X
        for w, b in zip(self.weights, self.biases):
            out = self._activate(np.dot(out, w) + b)
        return out

    def fit(self, X, y, epochs=100, lr=0.01):
        # Entrenamiento ficticio para evitar errores
        for _ in range(epochs):
            pass  # Aquí podrías poner tu lógica de backpropagation
        print("Método fit ejecutado (sin entrenamiento real)")

    def predict(self, X):
        return self.forward(X)

