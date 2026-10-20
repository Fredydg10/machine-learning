# Tener instalados las librerias de pandas, numpy, matplotlib, seaborn y scikit-learn.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Cargar dataset de dígitos
print(" Cargando dataset de dígitos...")
digits = load_digits()
X = digits.data  # 1797 muestras, 64 features (8x8 píxeles)
y = digits.target  # Etiquetas: 0-9

print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"Clases: {np.unique(y)}")


# 2. Visualizar algunos dígitos de ejemplos 
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(digits.images[idx], cmap='gray')
    ax.set_title(f'Dígito: {y[idx]}')
    ax.axis('off')
plt.suptitle(' Ejemplos del Dataset de Dígitos (8x8 píxeles)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 3. Escalar los datos (IMPORTANTE para PCA y t-SNE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA: Analizar varianza explicada
print("\n" + "="*60)
print(" Analisis varianza explicada (PCA)")
print("="*60)

pca_full = PCA()
pca_full.fit(X_scaled)

# Varianza explicada acumulada
varianza_explicada = pca_full.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

print(f"\nPrimeros 10 componentes:")
for i in range(10):
    print(f"  PC{i+1}: {varianza_explicada[i]*100:.2f}% (Acumulada: {varianza_acumulada[i]*100:.2f}%)")

print(f"\nCon 2 componentes: {varianza_acumulada[1]*100:.2f}% de varianza")
print(f"Con 10 componentes: {varianza_acumulada[9]*100:.2f}% de varianza")
print(f"Con 30 componentes: {varianza_acumulada[29]*100:.2f}% de varianza")

# Visualizar varianza explicada
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
axes[0].bar(range(1, 21), varianza_explicada[:20] * 100, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Varianza Explicada (%)')
axes[0].set_title('Varianza Explicada por Componente', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Gráfico acumulativo
axes[1].plot(range(1, 65), varianza_acumulada * 100, 'b-', linewidth=2)
axes[1].axhline(y=95, color='r', linestyle='--', label='95% varianza')
axes[1].axhline(y=99, color='g', linestyle='--', label='99% varianza')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Varianza Acumulada (%)')
axes[1].set_title('Varianza Acumulada', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 5. PCA: Proyectar a 2D
print("\n Aplicando PCA para reducir a 2 dimensiones...")
pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_scaled)

print(f"Shape original: {X.shape}")
print(f"Shape después de PCA: {X_pca.shape}")
print(f"Varianza explicada por PC1: {pca_2d.explained_variance_ratio_[0]*100:.2f}%")
print(f"Varianza explicada por PC2: {pca_2d.explained_variance_ratio_[1]*100:.2f}%")

# Visualizar PCA 2D
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=50, edgecolors='black')
plt.colorbar(scatter, label='Dígito')
plt.title(f' PCA 2D (Varianza explicada: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%)', 
          fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 6. t-SNE: Proyectar a 2D
print("\n Aplicando t-SNE para reducir a 2 dimensiones...")
print(" Esto puede tardar 10-30 segundos...")

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

print(f"Shape después de t-SNE: {X_tsne.shape}")

# Visualizar t-SNE 2D
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=50, edgecolors='black')
plt.colorbar(scatter, label='Dígito')
plt.title(' t-SNE 2D (perplexity=30)', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE dimensión 1')
plt.ylabel('t-SNE dimensión 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 7. Comparación de lado a lado 
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=50, edgecolors='black')
axes[0].set_title(f' PCA 2D\n(Varianza: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%)', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, alpha=0.3)

# t-SNE
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=50, edgecolors='black')
axes[1].set_title(' t-SNE 2D\n(perplexity=30)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)

plt.suptitle('🔢 PCA vs t-SNE: Visualización de Dígitos en 2D', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# 8. BONUS: PCA Como preprocesamiento para Random Forest
print("\n" + "="*60)
print(" BONUS: PCA + Random Forest para Clasificación")
print("="*60)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo 1: Sin PCA (64 features)
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)
acc_full = accuracy_score(y_test, rf_full.predict(X_test))
print(f"\nRandom Forest con 64 features: Accuracy = {acc_full:.4f}")

# Modelo 2: Con PCA (10 componentes = 95% varianza)
pca_10 = PCA(n_components=10, random_state=42)
X_train_pca = pca_10.fit_transform(X_train)
X_test_pca = pca_10.transform(X_test)

rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, rf_pca.predict(X_test_pca))
print(f"Random Forest con 10 features (PCA): Accuracy = {acc_pca:.4f}")

print(f"\n Reducción: 64 → 10 features ({(10/64)*100:.1f}% del original)")
print(f" Pérdida de accuracy: {(acc_full - acc_pca)*100:.2f}%")
print(f" Conclusión: PCA mantiene el rendimiento mientras reduce la complejidad")


# 9. Experimento: t-SNE con diferentes valores de perplexity
print("\n" + "="*60)
print(" EXPERIMENTO: t-SNE con diferente perplexity")
print("="*60)

perplexities = [5, 30, 50]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, perp in zip(axes, perplexities):
    print(f"  Ejecutando t-SNE con perplexity={perp}...")
    tsne_temp = TSNE(n_components=2, random_state=42, perplexity=perp, n_iter=1000)
    X_tsne_temp = tsne_temp.fit_transform(X_scaled)
    
    ax.scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1], c=y, cmap='tab10', alpha=0.6, s=30, edgecolors='black')
    ax.set_title(f't-SNE (perplexity={perp})', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)

plt.suptitle(' Efecto de Perplexity en t-SNE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n Día 10 completado!")