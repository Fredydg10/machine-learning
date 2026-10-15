# Tener instaladas las librerias de scikit-learn, pandas, numpy y matplotlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report


# 1. Cargar el dataset Iris
print("Cargando dataset Iris...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Shape: {X.shape}")
print(f"Clases: {target_names}")
print(f"\nPrimeras 5 muestras:\n{pd.DataFrame(X[:5], columns=feature_names)}")


# 2. Usar Solo 2 Features Para Visualizar En 2D                     
# Usaremos: largo del sépalo (0) y ancho del sépalo (1)
X_2d = X[:, :2]

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.3, random_state=42, stratify=y
)

# Escalar (IMPORTANTE para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n Train: {X_train.shape[0]} muestras")
print(f" Test:  {X_test.shape[0]} muestras")

# 3. Entrenar Modelos 
modelos = {
    "SVM Lineal (C=1)": SVC(kernel='linear', C=1.0),
    "SVM RBF (C=1)": SVC(kernel='rbf', C=1.0),
    "SVM RBF (C=10)": SVC(kernel='rbf', C=10.0),
    "SVM Polinomial": SVC(kernel='poly', degree=3, C=1.0),
    "Árbol (max_depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Árbol (max_depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42)
}

resultados = {}

print("\n" + "="*60)
print(" Entrenando Modelos...")
print("="*60)

for nombre, modelo in modelos.items():
    # Escalar solo para SVM
    if "SVM" in nombre:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    resultados[nombre] = {
        'modelo': modelo,
        'accuracy': acc,
        'escalado': "SVM" in nombre
    }
    print(f"📌 {nombre}: Accuracy = {acc:.4f}")


# 4. Función Para Visualizar Límite de Decisión
def plot_decision_boundary(X, y, modelo, escalado, scaler_obj, titulo, ax):
    """Dibuja el límite de decisión de un modelo en 2D"""
    
    # Crear una malla de puntos
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    # Preparar los puntos de la malla
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Escalar si es SVM
    if escalado:
        mesh_points_scaled = scaler_obj.transform(mesh_points)
        Z = modelo.predict(mesh_points_scaled)
    else:
        Z = modelo.predict(mesh_points)
    
    Z = Z.reshape(xx.shape)
    
    # Dibujar el fondo de colores (límites de decisión)
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
    
    # Dibujar los puntos reales
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1', 
                         edgecolors='black', s=50, zorder=5)
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(titulo)
    
    # Leyenda de clases
    handles, _ = scatter.legend_elements()
    ax.legend(handles, target_names, loc='upper right')



# 5. Visualizar Todos los limites de decisión
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
axes = axes.flatten()

for idx, (nombre, info) in enumerate(resultados.items()):
    plot_decision_boundary(
        X_train, y_train, 
        info['modelo'], info['escalado'], scaler,
        "",  # ← Título vacío aquí, lo ponemos manual abajo
        axes[idx]
    )
    
    # Agregar título con MUCHO espacio abajo (pad=25)
    axes[idx].set_title(
        f"{nombre}\nAccuracy: {info['accuracy']:.3f}",
        fontsize=11,
        fontweight='bold',
        pad=8
    )

# Título principal de la figura
fig.suptitle(
    "Límites de Decisión: SVM vs Árboles de Decisión (Dataset Iris)",
    fontsize=17,
    fontweight='bold'
)

plt.show()

# ============================================
# 6. VISUALIZAR EL ÁRBOL DE DECISIÓN (¡INTERPRETABLE!)
# ============================================
fig, ax = plt.subplots(figsize=(16, 10))

# Usamos el árbol de profundidad 3 (más interpretable)
arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol.fit(X_train, y_train)

plot_tree(arbol, 
          feature_names=[feature_names[0], feature_names[1]],
          class_names=target_names,
          filled=True, 
          rounded=True,
          fontsize=10,
          ax=ax)

ax.set_title("🌳 Árbol de Decisión (max_depth=3) - ¡Totalmente interpretable!", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. Reporte De Clasificación Del Mejor Modelo
mejor_nombre = max(resultados, key=lambda k: resultados[k]['accuracy'])
mejor_info = resultados[mejor_nombre]

print("\n" + "="*60)
print(f"MEJOR MODELO: {mejor_nombre} (Accuracy: {mejor_info['accuracy']:.4f})")
print("="*60)

if mejor_info['escalado']:
    y_pred_mejor = mejor_info['modelo'].predict(X_test_scaled)
else:
    y_pred_mejor = mejor_info['modelo'].predict(X_test)

print(classification_report(y_test, y_pred_mejor, target_names=target_names))