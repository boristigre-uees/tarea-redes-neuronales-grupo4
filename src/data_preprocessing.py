import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    path = "data/Walmart.csv"
    df = pd.read_csv(path)
    print("Columnas en CSV:", df.columns.tolist())
    
    # Aquí definimos la variable objetivo (ejemplo: forecasted_demand)
    target_column = "forecasted_demand"
    
    # Seleccionar columnas numéricas como features
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    X = df[numeric_cols]
    y = df[target_column].values.reshape(-1, 1)
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
   return np.mean(np.abs(revenue_true - revenue_pred)) / np.mean(revenue_true) * 100  # Abs para % error positivo
