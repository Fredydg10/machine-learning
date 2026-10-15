# Importar librerias de pandas, numpy, matplotlib, seaborn y sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Cargar Dataset Titanic (Kaggle)
print("Cargando dataset Titanic de Kaggle...")
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("❌ Error: No encuentro 'train.csv'. Asegúrate de que esté en la misma carpeta que este script.")
    exit()

print(f"Shape original: {df.shape}")
print(f"\nPrimeras filas:\n{df.head()}")

# 2. Limpieza De Datos
# Eliminamos columnas que no aportan al modelo o tienen demasiados nulos
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Rellenar valores nulos de 'Age' con la mediana (más robusto que la media)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Rellenar valores nulos de 'Embarked' con la moda (el puerto más frecuente)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convertir variables categóricas a numéricas (One-Hot Encoding)
# drop_first=True evita la multicolinealidad (ej: si no es 'male', es 'female')
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print(f"\nShape después de limpieza: {df.shape}")
print(f"Columnas finales: {df.columns.tolist()}")

# 3. Preparar X e y
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']               # Target (0 = murió, 1 = sobrevivió)

# Dividir en train y test (80/20)
# stratify=y asegura que la proporción de sobrevivientes sea igual en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar los datos (CRUCIAL para KNN y Regresión Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n Train: {X_train.shape[0]} muestras")
print(f" Test:  {X_test.shape[0]} muestras")


# 4. Entrenar Modelos de Clasificación
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (K=15)": KNeighborsClassifier(n_neighbors=15)
}

resultados = []

print("\n" + "="*60)
print("ENTRENANDO MODELOS DE CLASIFICACIÓN...")
print("="*60)

for nombre, modelo in modelos.items():
    print(f"\n Entrenando: {nombre}")
    
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    resultados.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })


# 5. Tabla Comprativa de Resultados
df_resultados = pd.DataFrame(resultados)
print("\n" + "="*60)
print("Comparación De Modelos")
print("="*60)
print(df_resultados.to_string(index=False))


# 6. Matrices de Confusión (Visualización)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (nombre, modelo) in zip(axes, modelos.items()):
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Murió (0)', 'Sobrevivió (1)'],
                yticklabels=['Murió (0)', 'Sobrevivió (1)'])
    ax.set_title(f'{nombre}\nMatriz de Confusión')
    ax.set_xlabel('Predicción del Modelo')
    ax.set_ylabel('Valor Real')

plt.tight_layout()
plt.show()

# 7. Reporte Detallado del Mejor Modelo (Regresión Logística)
print("\n" + "="*60)
print("Reporte De Clasificación (Mejor modelo: Regresión Logística)")
print("="*60)
modelo_final = LogisticRegression(max_iter=1000)
modelo_final.fit(X_train_scaled, y_train)
y_pred_final = modelo_final.predict(X_test_scaled)
print(classification_report(y_test, y_pred_final, target_names=['Murió (0)', 'Sobrevivió (1)']))