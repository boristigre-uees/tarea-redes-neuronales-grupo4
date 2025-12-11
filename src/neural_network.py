# src/neural_network.py
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='relu'):
        """
        layers: lista de enteros, cada entero es el número de neuronas por capa.
                Ej: [3, 4, 1] → 3 entradas, 4 en capa oculta, 1 salida
        activation: función de activación para capas ocultas ('relu', 'sigmoid', 'tanh')
        """
        self.layers = layers
        self.activation_name = activation
        self.weights = []
        self.biases = []

        # Inicialización de pesos y bias
        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]

            if activation.lower() == 'relu':
                weight = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            else:
                weight = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)

            bias = np.zeros((1, n_out))

            self.weights.append(weight)
            self.biases.append(bias)

        # Asignar función de activación
        if activation.lower() == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation.lower() == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation.lower() == 'tanh':
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:
            raise ValueError(f"Función de activación desconocida: {activation}")

    # Funciones de activación
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    # Forward propagation
    def forward(self, X):
        self.a = [X]  # activaciones
        self.z = []   # sumas ponderadas
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(self.a[-1], w) + b
            self.z.append(z)
            self.a.append(self.activation(z))
        # Capa de salida lineal (regresión)
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.z.append(z)
        self.a.append(z)
        return z

    # Backward propagation (gradiente simple MSE)
    def backward(self, y_true):
        m = y_true.shape[0]
        dz = (self.a[-1] - y_true) / m  # derivada MSE
        self.dW = []
        self.db = []

        for i in reversed(range(len(self.weights))):
            a_prev = self.a[i]
            dW = np.dot(a_prev.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            self.dW.insert(0, dW)
            self.db.insert(0, db)
            if i != 0:
                dz = np.dot(dz, self.weights[i].T) * self.activation_derivative(self.z[i-1])

    # Entrenamiento
    def train(self, X, y, epochs=100, learning_rate=0.01):
        losses = []  # Fix: Lista para losses
        for epoch in range(epochs):
            self.forward(X)
            # Fix: Calcular loss después de forward
            loss = np.mean((self.a[-1] - y)**2) / 2  # MSE loss
            losses.append(loss)
            self.backward(y)
            # Actualización de pesos y bias
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * self.dW[i]
                self.biases[i] -= learning_rate * self.db[i]
        return losses  # Fix: Retorna la lista de losses

    # Fit como alias de train
    def fit(self, X, y, epochs=100, lr=0.01):
        return self.train(X, y, epochs=epochs, learning_rate=lr)

    # Predicción
    def predict(self, X):
        return self.forward(X)



