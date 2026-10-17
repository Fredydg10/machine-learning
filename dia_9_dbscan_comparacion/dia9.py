#Tener instalado las librerias de pandas, numpy, matplotlib, seaborn y scikit-learn(adicional adjuntar el archivo Mall_Customers.csv en la misma carpeta que este script)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score


# 1. Demostración: Formas Raras (Lunas)
print("🌙 Generando datos en forma de lunas...")
X_moons, _ = make_moons(n_samples=400, noise=0.05, random_state=42)

# Escalar datos
scaler = StandardScaler()
X_moons_scaled = scaler.fit_transform(X_moons)

# --- K-Means (Fallará) ---
kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_kmeans = kmeans_moons.fit_predict(X_moons_scaled)

# --- Clustering Jerárquico (También fallará) ---
hc_moons = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels_hc = hc_moons.fit_predict(X_moons_scaled)

# --- DBSCAN (¡Ganará!) ---
# eps=0.3 y min_samples=5 funcionan perfecto para estas lunas escaladas
dbscan_moons = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan_moons.fit_predict(X_moons_scaled)

# Visualizar la comparación
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means
axes[0].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=labels_kmeans, cmap='viridis', s=30)
axes[0].set_title('K-Means (Falló: corta las lunas)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')

# Jerárquico
axes[1].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=labels_hc, cmap='plasma', s=30)
axes[1].set_title('Jerárquico (Falló: también corta)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')

# DBSCAN
# Los puntos de ruido (-1) los pintamos de negro
colors_dbscan = np.where(labels_dbscan == -1, 'black', labels_dbscan)
axes[2].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=colors_dbscan, cmap='viridis', s=30)
axes[2].set_title('DBSCAN (¡Éxito: separa las lunas perfecto!)', fontsize=12, fontweight='bold', color='green')
axes[2].set_xlabel('X'); axes[2].set_ylabel('Y')

plt.suptitle('🌙 DBSCAN vs Los Demás en Formas No Esféricas', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# 2. DBSCAN En datos reales (MALL CUSTOMERS)
print("\n Aplicando DBSCAN a Mall Customers...")
try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("❌ No encuentro 'Mall_Customers.csv'")
    exit()

X_real = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
X_real_scaled = StandardScaler().fit_transform(X_real)

# Ajustar DBSCAN
# Nota: DBSCAN es sensible a eps. Probamos 0.5
dbscan_real = DBSCAN(eps=0.5, min_samples=5)
df['Cluster_DBSAN'] = dbscan_real.fit_predict(X_real_scaled)

# Contar clusters y outliers
n_clusters = len(set(df['Cluster_DBSAN'])) - (1 if -1 in df['Cluster_DBSAN'] else 0)
n_noise = list(df['Cluster_DBSAN']).count(-1)

print(f" Clusters encontrados: {n_clusters}")
print(f"🔴 Outliers (ruido) detectados: {n_noise} ({n_noise/len(df)*100:.1f}%)")

# Visualizar
plt.figure(figsize=(10, 7))
# Puntos de clusters normales
mask_clusters = df['Cluster_DBSAN'] != -1
plt.scatter(df.loc[mask_clusters, 'Annual Income (k$)'], 
            df.loc[mask_clusters, 'Spending Score (1-100)'],
            c=df.loc[mask_clusters, 'Cluster_DBSAN'], cmap='viridis', s=60, edgecolor='k', label='Clientes normales')

# Puntos de ruido (outliers) en rojo grande
mask_noise = df['Cluster_DBSAN'] == -1
plt.scatter(df.loc[mask_noise, 'Annual Income (k$)'], 
            df.loc[mask_noise, 'Spending Score (1-100)'],
            c='red', marker='x', s=150, label=f'Outliers ({n_noise})')

plt.title(f'DBSCAN en Mall Customers (eps=0.5)\nDetectó {n_noise} clientes atípicos', 
          fontsize=14, fontweight='bold')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 3. Comparación Final: Silhouette Score 
print("\n" + "="*60)
print(" Comparación de los 3 Algoritmos (Mall Customers)")
print("="*60)

# K-Means K=5
kmeans_real = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_km = kmeans_real.fit_predict(X_real_scaled)
sil_km = silhouette_score(X_real_scaled, labels_km)

# Jerárquico K=5
hc_real = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels_hc_real = hc_real.fit_predict(X_real_scaled)
sil_hc = silhouette_score(X_real_scaled, labels_hc_real)

# DBSCAN (solo calculamos silhouette en los puntos que NO son ruido)
mask_no_noise = df['Cluster_DBSAN'] != -1
if n_clusters > 1 and mask_no_noise.sum() > 1:
    sil_db = silhouette_score(X_real_scaled[mask_no_noise], df['Cluster_DBSAN'][mask_no_noise])
else:
    sil_db = 0.0

print(f"K-Means (K=5):        Silhouette = {sil_km:.4f}")
print(f"Jerárquico (K=5):     Silhouette = {sil_hc:.4f}")
print(f"DBSCAN (eps=0.5):     Silhouette = {sil_db:.4f} (en {n_clusters} clusters)")

print("\n💡 NOTA: DBSCAN puede tener menor Silhouette Score porque")
print("   detecta outliers y clusters de formas complejas, pero")
print("   conceptualmente es más robusto para datos reales.")

# Gráfico final comparativo
fig, ax = plt.subplots(figsize=(10, 6))
modelos = ['K-Means', 'Jerárquico', 'DBSCAN']
scores = [sil_km, sil_hc, sil_db]
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

bars = ax.bar(modelos, scores, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Silhouette Score', fontsize=12)
ax.set_title('🏆 Comparación de Silhouette Score', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.6)
ax.grid(axis='y', alpha=0.3)

# Añadir valores encima de las barras
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{score:.4f}', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()