# src/experiments.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.neural_network import NeuralNetwork

DATA_PATH = "data/Walmart.csv"


def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    print("Columnas en CSV:", df.columns.tolist())

    # Seleccionamos solo columnas numéricas para X
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Definimos target
    target_column = "actual_demand"
    if target_column not in numeric_cols:
        raise ValueError(f"{target_column} no es numérico")

    X = df[numeric_cols].drop(columns=[target_column])
    y = df[target_column].values.reshape(-1, 1)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test), scaler

def run_single_experiment(hidden_layers, activation, lr, epochs):
    (X_train, y_train), (X_test, y_test), scaler = load_and_preprocess_data()

    # Creamos la red neuronal
    nn = NeuralNetwork(layer_sizes=[X_train.shape[1]] + hidden_layers + [1], activation=activation)

    # Entrenamiento (fit es ficticio)
    nn.fit(X_train, y_train, epochs=epochs, lr=lr)

    # Predicción
    y_pred = nn.predict(X_test)

    # Métrica simple: diferencia media
    error = (y_test - y_pred).mean()
    print(f"Error promedio: {error:.4f}")

    return error

def compare_experiments():
    experiments = [
        ([8], 'relu', 0.01, 300),
        ([16], 'relu', 0.01, 300),
        ([8, 8], 'relu', 0.01, 300)
    ]
    results = []
    for cfg in experiments:
        print("Ejecutando experimento:", cfg)
        r = run_single_experiment(*cfg)
        results.append((cfg, r))
    return results

if __name__ == "__main__":
    results = compare_experiments()
    print("Resultados finales:", results)
