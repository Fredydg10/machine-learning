# Tener Instaladas las librerias de Python: numpy, pandas, matplotlib, seaborn, scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, LeaveOneOut, 
    cross_val_score, cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


# 1. Crear Dataset 
print(" Creando dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],  # 70% vs 30% (ligeramente desequilibrado)
    random_state=42
)

print(f"Shape: {X.shape}")
print(f"Clase 0: {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
print(f"Clase 1: {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")


# 2. Problema: UNA SOLA DIVISIÓN TRAIN/TEST
print("\n" + "="*70)
print("Problema: Una sola división train/test")
print("="*70)

modelo = LogisticRegression(max_iter=1000, random_state=42)

# Ejecutar 5 veces con diferentes random_state
resultados_unicos = []

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i
    )
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    resultados_unicos.append({
        'División': i+1,
        'Accuracy': acc,
        'F1-Score': f1
    })
    
    print(f"División {i+1}: Accuracy={acc:.4f}, F1={f1:.4f}")

df_unicos = pd.DataFrame(resultados_unicos)
print(f"\n Estadísticas:")
print(f"  Accuracy: {df_unicos['Accuracy'].mean():.4f} (+/- {df_unicos['Accuracy'].std():.4f})")
print(f"  F1-Score: {df_unicos['F1-Score'].mean():.4f} (+/- {df_unicos['F1-Score'].std():.4f})")

print("\n⚠️  PROBLEMA: La variación entre divisiones es alta.")
print("   ¿Cuál división es la 'correcta'? ¡Ninguna!")


# 3. Solución 1: K-Fold Cross-Validation
print("\n" + "="*70)
print(" Solución 1: K-Fold Cross-Validation (K=5)")
print("="*70)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_score hace el loop automáticamente
scores_accuracy = cross_val_score(
    modelo, X, y, cv=kfold, scoring='accuracy'
)
scores_f1 = cross_val_score(
    modelo, X, y, cv=kfold, scoring='f1'
)

print(f"\nAccuracy por fold:")
for i, score in enumerate(scores_accuracy, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nF1-Score por fold:")
for i, score in enumerate(scores_f1, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\n Resultados finales:")
print(f"  Accuracy: {scores_accuracy.mean():.4f} (+/- {scores_accuracy.std():.4f})")
print(f"  F1-Score: {scores_f1.mean():.4f} (+/- {scores_f1.std():.4f})")

print(f"\n Ventaja: Menos varianza, más confiable")
print(f"   Desviación de Accuracy: {scores_accuracy.std():.4f} (vs {df_unicos['Accuracy'].std():.4f} con train/test simple)")


# 4. Solución 2: Stratified K-Fold
print("\n" + "="*70)
print(" Solución 2: Stratified K-Fold (preserva clases)")
print("="*70)

# Crear dataset más desequilibrado para demostrar
X_imb, y_imb = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    weights=[0.9, 0.1],  # 90% vs 10%
    random_state=42
)

print(f"Distribución original:")
print(f"  Clase 0: {np.sum(y_imb==0)} ({np.mean(y_imb==0)*100:.1f}%)")
print(f"  Clase 1: {np.sum(y_imb==1)} ({np.mean(y_imb==1)*100:.1f}%)")

# K-Fold normal
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n K-Fold Normal (puede desbalancear folds):")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_imb, y_imb), 1):
    y_train_fold = y_imb[train_idx]
    y_test_fold = y_imb[test_idx]
    
    prop_clase1_train = np.mean(y_train_fold == 1) * 100
    prop_clase1_test = np.mean(y_test_fold == 1) * 100
    
    print(f"  Fold {fold}: Train clase 1={prop_clase1_train:.1f}%, "
          f"Test clase 1={prop_clase1_test:.1f}%")

print("\n Stratified K-Fold (mantiene proporción):")
for fold, (train_idx, test_idx) in enumerate(skf.split(X_imb, y_imb), 1):
    y_train_fold = y_imb[train_idx]
    y_test_fold = y_imb[test_idx]
    
    prop_clase1_train = np.mean(y_train_fold == 1) * 100
    prop_clase1_test = np.mean(y_test_fold == 1) * 100
    
    print(f"  Fold {fold}: Train clase 1={prop_clase1_train:.1f}%, "
          f"Test clase 1={prop_clase1_test:.1f}%")

print("\n Stratified K-Fold mantiene ~10% en cada fold")
print("   ¡Importante para datos desequilibrados!")


# 5. Solución 3: LOOCV (Leave One Out)
print("\n" + "="*70)
print("✅ Solución 3: LOOCV (Leave One Out)")
print("="*70)
print("️  Esto puede tardar (1000 entrenamientos)...")

# Usar solo 200 muestras para no tardar tanto
X_small = X[:200]
y_small = y[:200]

loocv = LeaveOneOut()

scores_loocv = cross_val_score(
    modelo, X_small, y_small, cv=loocv, scoring='accuracy', n_jobs=-1
)

print(f"\n LOOCV Results (200 muestras):")
print(f"  Accuracy: {scores_loocv.mean():.4f} (+/- {scores_loocv.std():.4f})")
print(f"  Número de folds: {len(scores_loocv)} (uno por muestra)")

print("\n✅ VENTAJA: Máximo uso de datos")
print("❌ DESVENTAJA: Muy lento para datasets grandes")


# 6. Compración de métodos de validación
print("\n" + "="*70)
print(" Comparación de métodos de validación")
print("="*70)

# K-Fold con diferentes K
k_values = [3, 5, 10]
results_comparison = []

for k in k_values:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X, y, cv=kf, scoring='accuracy')
    
    results_comparison.append({
        'Método': f'K-Fold (K={k})',
        'Accuracy': scores.mean(),
        'Std': scores.std(),
        'Folds': k
    })

# Añadir LOOCV (con dataset pequeño)
results_comparison.append({
    'Método': 'LOOCV',
    'Accuracy': scores_loocv.mean(),
    'Std': scores_loocv.std(),
    'Folds': 200
})

df_comparison = pd.DataFrame(results_comparison)
print(df_comparison.to_string(index=False))

# Gráfico comparativo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparado
bars1 = axes[0].bar(df_comparison['Método'], df_comparison['Accuracy'], 
                    color='skyblue', edgecolor='black', yerr=df_comparison['Std'])
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title(' Accuracy por Método', fontsize=14, fontweight='bold')
axes[0].set_ylim(0.8, 0.95)
axes[0].grid(axis='y', alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Desviación estándar
bars2 = axes[1].bar(df_comparison['Método'], df_comparison['Std'], 
                    color='coral', edgecolor='black')
axes[1].set_ylabel('Desviación Estándar', fontsize=12)
axes[1].set_title(' Varianza (menor = más estable)', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle('🏆 Comparación de Métodos de Validación', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# 7. cross_validate: Múltiples métricas a la vez
print("\n" + "="*70)
print(" BONUS: cross_validate (múltiples métricas)")
print("="*70)

scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

scores_multi = cross_validate(
    modelo, X, y, cv=5, scoring=scoring_metrics, n_jobs=-1
)

print("\n Métricas múltiples con 5-Fold CV:")
for metric, scores in scores_multi.items():
    print(f"  {metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")


# 8. Lecciónes Clave 
print("\n" + "="*70)
print("💡 Lecciónes Clave")
print("="*70)

print("""
1. Train/Test simple es INESTABLE:
   - Depende de la división aleatoria
   - Alta varianza entre ejecuciones
   - No confíes en un solo número

2. K-Fold CV es el estándar:
   - K=5 o K=10 son los valores típicos
   - Menos varianza, más confiable
   - Usa todos los datos para train y test

3. Stratified K-Fold para datos desequilibrados:
   - Mantiene proporción de clases en cada fold
   - Evita folds sin clase minoritaria

4. LOOCV es extremo:
   - Máximo uso de datos
   - Muy lento (N entrenamientos)
   - Úsalo solo con datasets pequeños

5. cross_validate es tu amigo:
   - Calcula múltiples métricas a la vez
   - Más información sobre el rendimiento

RECOMENDACIÓN: Usa Stratified K-Fold (K=5 o 10) 
para clasificación, especialmente con datos desequilibrados.
""")

print("\n✅ Día 13 completado!")