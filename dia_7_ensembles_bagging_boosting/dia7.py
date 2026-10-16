# Tener instalados las librerias de pandas, numpy, matplotlib, seaborn y scikit-learn( adicional el XGBoost si se desea usar).
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    BaggingClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Intentar importar XGBoost (si no está instalado, lo saltamos)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost no está instalado. Instálalo con: pip install xgboost")


# 1. Cargar y limpiar datos 
print(" Cargando dataset Titanic...")
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("❌ Error: No encuentro 'train.csv'")
    exit()

# Limpieza
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# 2. Ajustar hiperparámetros de XGBoost
xgb_grid_search = None
if XGB_AVAILABLE:
    print("\n" + "="*70)
    print("Mini-reto: búsqueda de hiperparámetros para XGBoost")
    print("="*70)

    xgb_grid_search = GridSearchCV(
        estimator=XGBClassifier(
            max_depth=3,
            random_state=42,
            eval_metric='logloss',
            n_jobs=1
        ),
        param_grid={
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'n_estimators': [50, 100, 200, 500]
        },
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        refit=True
    )
    xgb_grid_search.fit(X_train, y_train)

    resultados_xgb = pd.DataFrame(xgb_grid_search.cv_results_)
    resultados_xgb = resultados_xgb[
        ['param_learning_rate', 'param_n_estimators', 'mean_test_score', 'std_test_score']
    ].sort_values('mean_test_score', ascending=False)
    print("Accuracy de validación cruzada por combinación:")
    print(resultados_xgb.to_string(index=False, formatters={
        'mean_test_score': '{:.4f}'.format,
        'std_test_score': '{:.4f}'.format
    }))
    print(f"\nMejor combinación: {xgb_grid_search.best_params_}")
    print(f"Mejor accuracy CV: {xgb_grid_search.best_score_:.4f}")


# 3. Definir Modelos
modelos = {
    "Bagging (100 árboles)": BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=100,
        random_state=42
    ),
    "AdaBoost (100 estimadores)": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    "Random Forest (baseline)": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
}

# Agregar la mejor combinación de XGBoost si está disponible
if xgb_grid_search is not None:
    modelos[
        "XGBoost (mejor combinación)"
    ] = xgb_grid_search.best_estimator_

# Stacking: comparar las dos combinaciones base del mini-reto
stacking_rf_ada_knn = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('ada', AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            random_state=42
        )),
        ('knn', Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]))
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
modelos["Stacking (RF+AdaBoost+KNN)"] = stacking_rf_ada_knn

stacking_svm_gb_nb = StackingClassifier(
    estimators=[
        ('svm', Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=False, random_state=42))
        ])),
        ('gb', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )),
        ('nb', GaussianNB())
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
modelos["Stacking (SVM+Gradient Boosting+Naive Bayes)"] = stacking_svm_gb_nb


# 4. Entrenar y evaluar modelos
resultados = []

print("\n" + "="*70)
print("Entrenando Modelos Ensemble...")
print("="*70)

for nombre, modelo in modelos.items():
    print(f"\n Entrenando: {nombre}")
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    # Cross-validation para métrica más robusta
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"   Accuracy (test): {acc:.4f}")
    print(f"   Accuracy (CV 5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    resultados.append({
        "Modelo": nombre,
        "Accuracy_Test": acc,
        "Accuracy_CV": cv_scores.mean(),
        "CV_Std": cv_scores.std(),
        "CV_Scores": cv_scores,
        "modelo_obj": modelo,
        "y_pred": y_pred
    })


# 5. Tabla Comparativa
df_resultados = pd.DataFrame(resultados)
print("\n" + "="*70)
print("Comparación de Modelos")
print("="*70)
print(df_resultados[['Modelo', 'Accuracy_Test', 'Accuracy_CV', 'CV_Std']].to_string(index=False))


# 6. Visualización: Accuracy comparado
fig, axes = plt.subplots(
    1, 2, figsize=(18, 7),
    gridspec_kw={'width_ratios': [1, 1.5]},
    constrained_layout=True
)

# Gráfico de barras
df_plot = df_resultados.sort_values('Accuracy_Test', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_plot)))
axes[0].barh(df_plot['Modelo'], df_plot['Accuracy_Test'], color=colors)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Accuracy en Test Set', fontsize=14, fontweight='bold')
axes[0].set_xlim(0.7, 0.9)

for i, (idx, row) in enumerate(df_plot.iterrows()):
    axes[0].text(row['Accuracy_Test'] + 0.005, i, f"{row['Accuracy_Test']:.4f}", 
                 va='center', fontweight='bold')

# Gráfico de boxplot con CV scores
modelos_para_boxplot = []
scores_para_boxplot = []

for info in resultados:
    scores = info['CV_Scores']
    modelos_para_boxplot.extend([info['Modelo']] * len(scores))
    scores_para_boxplot.extend(scores)

df_box = pd.DataFrame({
    'Modelo': modelos_para_boxplot,
    'Score': scores_para_boxplot
})
etiquetas_boxplot = {
    'AdaBoost (100 estimadores)': 'AdaBoost',
    'Random Forest (baseline)': 'Random Forest',
    'XGBoost (mejor combinación)': 'XGBoost (mejor config.)',
    'Stacking (RF+AdaBoost+KNN)': 'Stacking (RF + AdaBoost + KNN)',
    'Stacking (SVM+Gradient Boosting+Naive Bayes)':
        'Stacking (SVM + GB + Naive Bayes)'
}
df_box['Modelo'] = df_box['Modelo'].replace(etiquetas_boxplot)

sns.boxplot(
    x='Score',
    y='Modelo',
    hue='Modelo',
    data=df_box,
    ax=axes[1],
    palette='viridis',
    legend=False
)
axes[1].set_title('Accuracy CV (5 folds)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Accuracy')
axes[1].set_xlim(0.7, 0.9)
plt.show()


# 7. Matrices de Confusión (top 3 modelos)
top3 = df_resultados.nlargest(3, 'Accuracy_Test')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (_, info) in zip(axes, top3.iterrows()):
    cm = confusion_matrix(y_test, info['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Murió (0)', 'Sobrevivió (1)'],
                yticklabels=['Murió (0)', 'Sobrevivió (1)'])
    ax.set_title(f"{info['Modelo']}\n(Accuracy: {info['Accuracy_Test']:.3f})",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')

plt.suptitle("Top 3 Modelos: Matrices de Confusión",
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


# 8. Reporte del Ganador y análisis de complejidad
ganador = df_resultados.loc[df_resultados['Accuracy_Test'].idxmax()]
print("\n" + "="*70)
print(f"GANADOR: {ganador['Modelo']}")
print(f"   Accuracy Test: {ganador['Accuracy_Test']:.4f}")
print(f"   Accuracy CV: {ganador['Accuracy_CV']:.4f}")
print("="*70)

ganador_modelo = ganador['modelo_obj']
y_pred_final = ganador['y_pred']
print("\n Reporte de clasificación:")
print(classification_report(y_test, y_pred_final, 
                            target_names=['Murió (0)', 'Sobrevivió (1)']))

mejores_stacking = df_resultados[
    df_resultados['Modelo'].str.startswith('Stacking')
].sort_values('Accuracy_Test', ascending=False)
print("\nComparación de las variantes de Stacking (accuracy de test):")
print(mejores_stacking[['Modelo', 'Accuracy_Test']].to_string(index=False))
print(f"Mejor Stacking: {mejores_stacking.iloc[0]['Modelo']} "
      f"({mejores_stacking.iloc[0]['Accuracy_Test']:.4f})")

if xgb_grid_search is not None:
    xgb_resultado = df_resultados[
        df_resultados['Modelo'] == "XGBoost (mejor combinación)"
    ].iloc[0]
    mejor_stacking_acc = mejores_stacking.iloc[0]['Accuracy_Test']
    diferencia = mejor_stacking_acc - xgb_resultado['Accuracy_Test']
    print("\nTrade-off: mejor Stacking vs. XGBoost")
    print(f"XGBoost: {xgb_resultado['Accuracy_Test']:.4f} "
          f"con {xgb_grid_search.best_params_}")
    print(f"Mejor Stacking: {mejor_stacking_acc:.4f}")
    print(f"Diferencia (Stacking - XGBoost): {diferencia:+.4f}")
    print("Bagging entrena modelos independientes y se puede paralelizar; "
          "Boosting los entrena secuencialmente; Stacking añade un meta-modelo "
          "y validación cruzada, por lo que suele ser el más costoso.")
    print("La complejidad extra del Stacking solo se justifica si la mejora "
          "observada compensa el coste y la dificultad de mantenimiento.")
