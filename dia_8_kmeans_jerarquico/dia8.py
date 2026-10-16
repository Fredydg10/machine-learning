# Tener las librerias descargadas de pandas, numpy, matplotlib, seaborn y scikit-learn(adicional adjuntar el archivo 'Mall_Customers.csv' en la misma carpeta que este script para que funcione correctamente)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch


# 1. Cargar Dataset
print("🛍️ Cargando Mall Customers...")
try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("❌ No encuentro 'Mall_Customers.csv'")
    exit()

print(f"Shape: {df.shape}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nInfo:\n{df.info()}")
print(f"\nEstadísticas:\n{df.describe()}")


# 2. Exploración de datos 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribución de género
df['Gender'].value_counts().plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#4ecdc4'])
axes[0].set_title('Distribución de Género')
axes[0].set_ylabel('Cantidad')

# Distribución de edad
sns.histplot(df['Age'], bins=20, ax=axes[1], color='#95e1d3')
axes[1].set_title('Distribución de Edad')

# Ingreso vs Spending Score
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Gender', ax=axes[2], palette=['#ff6b6b', '#4ecdc4'])
axes[2].set_title('Ingreso vs Spending Score')

plt.tight_layout()
plt.show()


# 3. Preparar Features para Clustering
# Incluimos edad, ingreso y gasto para comparar perfiles de clientes
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Escalar (importante para K-Means y Jerárquico)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures para clustering: {X.shape}")


# 4. Metodo del codo y Silhouette para determinar K óptimo
print("\n" + "="*60)
print(" MÉTODO DEL CODO")
print("="*60)

inercias = []
silhouette_scores = []
K_range = range(1, 11)
K_silhouette = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercias.append(kmeans.inertia_)

    if k in K_silhouette:
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k}: Inercia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")

mejor_k_silhouette = list(K_silhouette)[int(np.argmax(silhouette_scores))]
print(
    f"\nMayor Silhouette Score entre K=2 y K=8: K={mejor_k_silhouette} "
    f"({max(silhouette_scores):.4f})"
)
if mejor_k_silhouette == 5:
    print("El Silhouette Score respalda K=5 como la mejor opción dentro del rango probado.")
else:
    print(
        f"El Silhouette Score no confirma K=5; favorece K={mejor_k_silhouette} "
        "dentro del rango probado. Contrástalo con el método del codo."
    )

# Visualizar método del codo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inercias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de clusters (K)')
axes[0].set_ylabel('Inercia')
axes[0].set_title('Método del Codo', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_silhouette, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por K', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 5. K-Means con K óptimo
# Según el codo, K=5 suele ser óptimo para este dataset
K_OPTIMO = 5

print(f"\n Aplicando K-Means con K={K_OPTIMO}")
kmeans = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# Visualizar clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster_KMeans', palette='viridis', s=100, edgecolor='black')

# Dibujar centroides
centroides = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroides[:, 1], centroides[:, 2],
            c='red', marker='X', s=300, label='Centroides', 
            edgecolors='black', linewidths=2)

plt.title(f'🛍️  K-Means Clustering (K={K_OPTIMO})', fontsize=16, fontweight='bold')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 6. Cluster Jerárquico
print("\n" + "="*60)
print("CLUSTERING JERÁRQUICO")
print("="*60)

# Comparar dendrogramas con los tres criterios de enlace
linkage_methods = ['ward', 'complete', 'single']
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, method in zip(axes, linkage_methods):
    sch.dendrogram(
        sch.linkage(X_scaled, method=method),
        labels=df['CustomerID'].astype(str),
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax
    )
    ax.set_title(f"Linkage: {method}", fontsize=13, fontweight='bold')
    ax.set_xlabel('Clientes')
    ax.set_ylabel('Distancia')

plt.suptitle('Comparación de dendrogramas por linkage', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# Aplicar clustering jerárquico con K=5 para cada linkage
print(f"\nClustering jerárquico con K={K_OPTIMO} por linkage:")
for method in linkage_methods:
    hc = AgglomerativeClustering(n_clusters=K_OPTIMO, linkage=method)
    labels = hc.fit_predict(X_scaled)
    df[f'Cluster_{method}'] = labels
    score = silhouette_score(X_scaled, labels)
    print(f"Linkage {method}: Silhouette Score = {score:.4f}")

df['Cluster_Jerarquico'] = df['Cluster_ward']

# Visualizar
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster_Jerarquico', palette='plasma', s=100, edgecolor='black')

plt.title(f'Clustering Jerárquico (K={K_OPTIMO})', fontsize=16, fontweight='bold')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 7. Comparar K-Means vs Jerárquico
print("\n" + "="*60)
print("COMPARACIÓN DE MÉTODOS")
print("="*60)

sil_kmeans = silhouette_score(X_scaled, df['Cluster_KMeans'])
sil_hc = silhouette_score(X_scaled, df['Cluster_Jerarquico'])

print(f"K-Means:        Silhouette Score = {sil_kmeans:.4f}")
print(f"Jerárquico:     Silhouette Score = {sil_hc:.4f}")

# Visualizar ambos lado a lado
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster_KMeans', palette='viridis', s=100, edgecolor='black', ax=axes[0])
centroides = scaler.inverse_transform(kmeans.cluster_centers_)
axes[0].scatter(centroides[:, 1], centroides[:, 2], c='red', marker='X', s=200)
axes[0].set_title(f'K-Means (Silhouette: {sil_kmeans:.4f})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Ingreso Anual (k$)')
axes[0].set_ylabel('Spending Score')

sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster_Jerarquico', palette='plasma', s=100, edgecolor='black', ax=axes[1])
axes[1].set_title(f'Jerárquico (Silhouette: {sil_hc:.4f})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Ingreso Anual (k$)')
axes[1].set_ylabel('Spending Score')

plt.suptitle('🛍️  K-Means vs Clustering Jerárquico', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# 8. Analizar los clusters (La parte del negocio)
print("\n" + "="*60)
print("PERFIL DE CADA CLUSTER (K-Means)")
print("="*60)

cluster_stats = df.groupby('Cluster_KMeans').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Gender': lambda x: x.value_counts().idxmax()
}).round(2)

print(cluster_stats)

# Nombres interpretativos para cada cluster
nombres_clusters = {
    0: "Gastadores moderados",
    1: "Alto ingreso, bajo gasto",
    2: "Ingreso medio, gasto medio",
    3: "Bajo ingreso, alto gasto",
    4: "Alto ingreso, alto gasto"
}

print("\n Interpretación de clusters:")
for cluster, nombre in nombres_clusters.items():
    stats = cluster_stats.loc[cluster]
    print(f"\nCluster {cluster} - {nombre}:")
    print(f"   Edad promedio: {stats['Age']:.1f}")
    print(f"   Ingreso promedio: ${stats['Annual Income (k$)']:.1f}k")
    print(f"   Spending Score promedio: {stats['Spending Score (1-100)']:.1f}")
    print(f"   Género mayoritario: {stats['Gender']}")