# src/experiments.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.neural_network import NeuralNetwork

DATA_PATH = "data/Walmart.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def business_metric(y_true, y_pred, base_price=100):
    """Métrica de dominio: % error en revenue (precio * demanda)."""
    revenue_true = y_true * base_price
    revenue_pred = y_pred * base_price
    return np.mean(np.abs(revenue_true - revenue_pred)) / np.mean(revenue_true) * 100

def load_and_preprocess_data(target_col='quantity_sold'):
    """Carga y preprocess: Features para Proyecto 6 (precios/productos)."""
    df = pd.read_csv(DATA_PATH)
    print("Columnas en CSV:", df.columns.tolist())

    # Features específicas: unit_price, holiday_indicator, store_id
    feature_cols = ['unit_price', 'holiday_indicator', 'store_id']
    if target_col not in df.columns:
        raise ValueError(f"{target_col} no en dataset")
    X = df[feature_cols].fillna(0).values  # Llenar nulos
    y = df[target_col].values.reshape(-1, 1)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test), scaler

def run_baseline(X_train, y_train, X_test, y_test):
    """Baseline: Regresión lineal."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rev_error = business_metric(y_test.flatten(), y_pred.flatten())
    return mse, y_pred.flatten()

def run_single_experiment(hidden_layers, activation, lr, epochs):
    """Ejecuta NN experimento."""
    (X_train, y_train), (X_test, y_test), scaler = load_and_preprocess_data()

    # Crea NN: input = 3 features + hidden + output=1
    nn = NeuralNetwork(layers=[X_train.shape[1]] + hidden_layers + [1], activation=activation)

    # Entrenamiento (train retorna losses o self; capturamos)
    train_result = nn.train(X_train, y_train, epochs, lr)
    losses = train_result if isinstance(train_result, list) else []  # Fix: Si no lista, vacío

    # Predicción
    y_pred = nn.predict(X_test).flatten()

    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    rev_error = business_metric(y_test.flatten(), y_pred)
    print(f"MSE: {mse:.4f}, Revenue Error: {rev_error:.2f}%")

    return mse, rev_error, losses, y_pred

def compare_experiments():
    """Parte 3: Baseline + 9 configs (3 arch x 3 act), variando LR/epochs."""
    (X_train, y_train), (X_test, y_test), _ = load_and_preprocess_data()

    # Baseline PRIMERO
    mse_bl, y_pred_bl = run_baseline(X_train, y_train, X_test, y_test)
    print(f"Baseline MSE: {mse_bl:.4f}, Revenue Error: {business_metric(y_test.flatten(), y_pred_bl):.2f}%")

    # Experimentos
    configs = [
        ([8], 'relu', 0.01, 300),
        ([16], 'relu', 0.01, 300),
        ([8, 8], 'relu', 0.01, 300),
        ([8], 'sigmoid', 0.01, 300),
        ([16], 'sigmoid', 0.001, 300),  # Varia LR
        ([8, 8], 'sigmoid', 0.01, 500),  # Varia epochs
        ([8], 'tanh', 0.01, 300),
        ([16], 'tanh', 0.01, 300),
        ([8, 8], 'tanh', 0.005, 300)  # Varia LR
    ]

    results = []
    losses_refer = None
    for i, cfg in enumerate(configs):
        print(f"Ejecutando experimento: {cfg}")
        hidden_layers, activation, lr, epochs = cfg
        mse, rev_error, losses, y_pred = run_single_experiment(hidden_layers, activation, lr, epochs)
        results.append({
            "capas": hidden_layers,
            "activacion": activation,
            "lr": lr,
            "epochs": epochs,
            "MSE": mse,
            "Revenue_Error_%": rev_error
        })
        if i == 0 and isinstance(losses, list) and len(losses) > 0:
            losses_refer = losses  # Solo si lista

    # Guardar CSV (incluye baseline)
    df_results = pd.DataFrame(results)
    baseline_row = pd.DataFrame([{
        "capas": "Baseline", "activacion": "LR", "lr": "N/A", "epochs": "N/A",
        "MSE": mse_bl, "Revenue_Error_%": business_metric(y_test.flatten(), y_pred_bl)
    }])
    df_results = pd.concat([baseline_row, df_results], ignore_index=True)
    df_results.to_csv(os.path.join(RESULTS_DIR, "performance_comparison.csv"), index=False)
    print(f"Resultados guardados en {os.path.join(RESULTS_DIR, 'performance_comparison.csv')}")

    # training_curves.png (ReLU primera, solo si lista)
    if isinstance(losses_refer, list) and len(losses_refer) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(losses_refer[:min(100, len(losses_refer))])  # Safe slice
        plt.title('Training Loss - ReLU [8]')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
        plt.close()
    else:
        print("Nota: Losses no es lista; skipping training_curves.png (verifica return en train())")

    # architecture_analysis.png (bar MSE)
    plt.figure(figsize=(10, 6))
    models = [f"{r['capas']}-{r['activacion']}" for r in results] + ["Baseline"]
    mses = [r['MSE'] for r in results] + [mse_bl]
    plt.bar(models, mses)
    plt.title('MSE Comparison por Arquitectura/Activación')
    plt.xlabel('Modelo')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "architecture_analysis.png"))
    plt.close()

    # Comparación de activaciones (tu plot original)
    df_results_no_bl = df_results[df_results['capas'] != 'Baseline'].copy()
    df_results_no_bl['capas_tuple'] = df_results_no_bl['capas'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    plt.figure(figsize=(8, 6))
    for capas in df_results_no_bl['capas_tuple'].unique():
        subset = df_results_no_bl[df_results_no_bl['capas_tuple'] == capas]
        plt.plot(subset['activacion'], subset['MSE'], marker='o', label=f"Capas {capas}")
    plt.xlabel("Función de activación")
    plt.ylabel("MSE")
    plt.title("Comparación de Activaciones")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "activation_comparison.png"))
    plt.close()

    return df_results

if __name__ == "__main__":
    results = compare_experiments()
    print("=== Resultados Finales ===")
    for _, r in results.iterrows():
        print(f"Capas: {r['capas']}, Activación: {r['activacion']}, MSE: {r['MSE']:.4f}, Rev Error: {r['Revenue_Error_%']:.2f}%")