# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_and_preprocess_data(path='data/Walmart.csv', test_size=0.2):
    """
    Carga el CSV, selecciona columnas numéricas, escala y divide en train/test.
    """
    df = pd.read_csv(path)
    print("Columnas en CSV:", df.columns.tolist())
    
    target_column = 'actual_demand'
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)  # eliminar columna objetivo de X
    
    X = df[numeric_cols]
    y = df[target_column]
    
    # Escalar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    return (X_train, y_train), (X_test, y_test), scaler

def business_metric(y_true, y_pred):
    """
    Métrica de negocio: RMSE (puedes personalizar según tu objetivo)
    """
    return mean_squared_error(y_true, y_pred, squared=False)
