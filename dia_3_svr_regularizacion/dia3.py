import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

# 1. Cargar Datos Reales: California Housing
print("Cargando dataset California Housing...")
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target  # Precio de la casa en cientos de miles de USD

print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"\nPrimeras 5 filas:\n{X.head()}")



# 2. Dividir en train y test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📦 Train: {X_train.shape[0]} muestras")
print(f"📦 Test:  {X_test.shape[0]} muestras")


# 3. Escalar los datos (¡IMPORTANTE para SVR, Ridge y Lasso!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 4. Entrenar los 4 modelos y evaluar
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1),
    "SVR (C=1.0)": SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

resultados = []

print("\n" + "="*60)
print("ENTRENANDO MODELOS...")
print("="*60)

for nombre, modelo in modelos.items():
    print(f"\n📌 Entrenando: {nombre}")
    
    # SVR necesita datos escalados, los demás también ayudan
    if "SVR" in nombre:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    
    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MSE:  {mse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    resultados.append({
        "Modelo": nombre,
        "MSE": mse,
        "MAE": mae,
        "R²": r2
    })



# 5. Comparar resultados en la tabla
df_resultados = pd.DataFrame(resultados)
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS")
print("="*60)
print(df_resultados.to_string(index=False))



# 6. Visualizar: R² de cada modelo
plt.figure(figsize=(10, 6))
plt.bar(df_resultados["Modelo"], df_resultados["R²"], color=['blue', 'green', 'orange', 'red'])
plt.title("R² Score por Modelo (más alto = mejor)")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(df_resultados["R²"]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()