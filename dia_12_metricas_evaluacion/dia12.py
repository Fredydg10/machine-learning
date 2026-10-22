# Tener instaladas las librerías necesarias de Python de: Pandas, Numpy, Matplotlib, Seaborn y Scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)


# 1. Crear dataset desequilibrado (Imbalanced)
print(" Creando dataset desequilibrado (fraude)...")
np.random.seed(42)

# Dataset con 95% clase 0, 5% clase 1
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95, 0.05],  # 95% vs 5%
    flip_y=0,
    random_state=42
)

print(f"Shape: {X.shape}")
print(f"Distribución de clases:")
print(f"  Clase 0 (Normal):  {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
print(f"  Clase 1 (Fraude):  {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 2. Entrenar Modelo "INGENUO" (solo accuracy)
print("\n" + "="*70)
print("Modelo 1: Regresión Logística")
print("="*70)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_logreg = log_reg.predict(X_test)
y_proba_logreg = log_reg.predict_proba(X_test)[:, 1]

# Métricas básicas
acc = accuracy_score(y_test, y_pred_logreg)
prec = precision_score(y_test, y_pred_logreg)
rec = recall_score(y_test, y_pred_logreg)
f1 = f1_score(y_test, y_pred_logreg)

print(f"\nAccuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print("\n Reporte completo:")
print(classification_report(y_test, y_pred_logreg, 
                            target_names=['Normal (0)', 'Fraude (1)']))


# 3. Matriz de Confusión 
cm = confusion_matrix(y_test, y_pred_logreg)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (0)', 'Fraude (1)'],
            yticklabels=['Normal (0)', 'Fraude (1)'])
plt.title(' Matriz de Confusión - Regresión Logística', 
          fontsize=14, fontweight='bold')
plt.ylabel('Valor Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
plt.show()


# 4. Curva ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_proba_logreg)
roc_auc = auc(fpr, tpr)

print(f"\n ROC-AUC Score: {roc_auc:.4f}")
print("   (1.0 = perfecto, 0.5 = adivinar al azar)")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Modelo aleatorio (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity/Recall)')
plt.title('Curva ROC-AUC', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 5. Curva Precision-Recall
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba_logreg)
avg_precision = average_precision_score(y_test, y_proba_logreg)

print(f"\n Average Precision Score: {avg_precision:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=2,
         label=f'PR curve (AP = {avg_precision:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(' Curva Precision-Recall', fontsize=14, fontweight='bold')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 6. Modelo 2: Árbol de Decisión (overfitting)
print("\n" + "="*70)
print(" Modelo 2: Árbol de Decisión (sin límites)")
print("="*70)

tree = DecisionTreeClassifier(random_state=42)  # Sin max_depth → overfitting
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

acc_tree = accuracy_score(y_test, y_pred_tree)
prec_tree = precision_score(y_test, y_pred_tree)
rec_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)

print(f"Accuracy:  {acc_tree:.4f}")
print(f"Precision: {prec_tree:.4f}")
print(f"Recall:    {rec_tree:.4f}")
print(f"F1-Score:  {f1_tree:.4f}")


# 7. Comparación Modelos
print("\n" + "="*70)
print(" Comparacion Final de metricas")
print("="*70)

df_comparacion = pd.DataFrame({
    'Modelo': ['Regresión Logística', 'Árbol de Decisión'],
    'Accuracy': [acc, acc_tree],
    'Precision': [prec, prec_tree],
    'Recall': [rec, rec_tree],
    'F1-Score': [f1, f1_tree]
})

print(df_comparacion.to_string(index=False))

# Gráfico de barras comparativo
fig, axes = plt.subplots(2, 2, figsize=(9, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#ff6b6b', '#4ecdc4']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    values = df_comparacion[metric]
    bars = ax.bar(df_comparacion['Modelo'], values, color=colors, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)
    
    # Añadir valores encima de las barras
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', fontweight='bold')

plt.suptitle('Comparación de métricas por modelo',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()


# 8. Lección ¿Porque importa esto?
print("\n" + "="*70)
print(" Lecciones Clave")
print("="*70)

print(f"""
1. Accuracy engañoso:
   - Ambos modelos tienen ~95% de accuracy
   - ¡Pero eso es porque el 95% de los datos son normales!
   - El accuracy NO nos dice si detectan fraude

2. Precision vs Recall:
   - RegLog: Precision={prec:.2f}, Recall={rec:.2f}
   - Árbol:   Precision={prec_tree:.2f}, Recall={rec_tree:.2f}
   
   Si Recall es bajo → Muchos fraudes pasan desapercibidos (FN)
   Si Precision es baja → Muchas alertas falsas (FP)

3. F1-Score es el balance:
   - RegLog: F1={f1:.4f}
   - Árbol:  F1={f1_tree:.4f}
   - El mejor F1 indica mejor balance entre Precision y Recall

4. ROC-AUC mide capacidad de separación:
   - AUC=1.0 → Separa perfecto las clases
   - AUC=0.5 → No mejor que adivinar al azar
   - Nuestro modelo: AUC={roc_auc:.4f}

CONCLUSIÓN: En datos desequilibrados, NUNCA confíes solo en accuracy.
Usa siempre: Precision, Recall, F1, y ROC-AUC.
""")

print("\n✅ Día 12 completado!")