# Tener instaladas las librerias de Optuna, Scikit-learn, Pandas, Numpy, Matplotlib y Seaborn(el proceso del script demora 1-2 minutos tener paciencia)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import randint, uniform


# OPTUNA (si está instalado)
try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna no está instalado. Instálalo con: pip install optuna")
    print("   (El código funcionará igual, solo saltará esa parte)\n")


# 1. Crear Dataset
print(" Creando dataset de clasificación...")
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# 2. Modelo Base (parámetros por defecto)
print("\n" + "="*70)
print(" Modelo Base (Random Forest por defecto)")
print("="*70)

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

acc_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base)

print(f"Accuracy: {acc_base:.4f}")
print(f"F1-Score: {f1_base:.4f}")
print("\nHiperparámetros por defecto:")
print(f"  n_estimators: {rf_base.n_estimators}")
print(f"  max_depth: {rf_base.max_depth}")
print(f"  min_samples_split: {rf_base.min_samples_split}")


# 3. GRIDSEARCHCV (Búsqueda Exhaustiva)
print("\n" + "="*70)
print(" GRIDSEARCHCV - Búsqueda en Cuadrícula")
print("="*70)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print(f"\n Parámetros a buscar:")
print(f"  n_estimators: {param_grid['n_estimators']}")
print(f"  max_depth: {param_grid['max_depth']}")
print(f"  min_samples_split: {param_grid['min_samples_split']}")
print(f"  min_samples_leaf: {param_grid['min_samples_leaf']}")

# Calcular combinaciones totales
total_combinations = (
    len(param_grid['n_estimators']) *
    len(param_grid['max_depth']) *
    len(param_grid['min_samples_split']) *
    len(param_grid['min_samples_leaf'])
)
print(f"\n Total de combinaciones: {total_combinations}")

print("\n Ejecutando GridSearchCV (esto puede tardar 1-2 minutos)...")
start_time = time.time()

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

grid_time = time.time() - start_time

print(f"\n✅ GridSearchCV completado en {grid_time:.2f} segundos")
print(f"\n🏆 MEJORES HIPERPARÁMETROS (GridSearch):")
print(f"  {grid_search.best_params_}")
print(f"\n📊 Mejor F1-Score (CV): {grid_search.best_score_:.4f}")

# Evaluar en test set
y_pred_grid = grid_search.predict(X_test)
acc_grid = accuracy_score(y_test, y_pred_grid)
f1_grid = f1_score(y_test, y_pred_grid)

print(f"Accuracy en test: {acc_grid:.4f}")
print(f"F1-Score en test: {f1_grid:.4f}")


# 4. RANDOMIZEDSEARCHCV (Búsqueda Aleatoria)
print("\n" + "="*70)
print(" RANDOMIZEDSEARCHCV - Búsqueda Aleatoria")
print("="*70)

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2']
}

print(f"\n Distribuciones de parámetros:")
print(f"  n_estimators: randint(50, 300)")
print(f"  max_depth: [10, 20, 30, None]")
print(f"  min_samples_split: randint(2, 10)")
print(f"  min_samples_leaf: randint(1, 5)")
print(f"  max_features: ['sqrt', 'log2']")

n_iter = 20
print(f"\n Iteraciones: {n_iter} (combinaciones aleatorias)")

print("\n Ejecutando RandomizedSearchCV (30-60 segundos)...")
start_time = time.time()

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

random_time = time.time() - start_time

print(f"\n✅ RandomizedSearchCV completado en {random_time:.2f} segundos")
print(f"\n🏆 MEJORES HIPERPARÁMETROS (RandomSearch):")
print(f"  {random_search.best_params_}")
print(f"\n📊 Mejor F1-Score (CV): {random_search.best_score_:.4f}")

# Evaluar en test set
y_pred_random = random_search.predict(X_test)
acc_random = accuracy_score(y_test, y_pred_random)
f1_random = f1_score(y_test, y_pred_random)

print(f"Accuracy en test: {acc_random:.4f}")
print(f"F1-Score en test: {f1_random:.4f}")


# 5. Optuna (Optimización Bayesiana)
optuna_time = 0.0
acc_optuna = None
f1_optuna = None

if OPTUNA_AVAILABLE:
    print("\n" + "="*70)
    print("🧠 OPTUNA - Optimización Bayesiana")
    print("="*70)
    
    def objective(trial):
        """Función objetivo para Optuna"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42,
            'n_jobs': -1
        }
        
        # max_depth: None o un valor entre 10 y 50
        max_depth_choice = trial.suggest_categorical('max_depth_choice', [None, 'value'])
        if max_depth_choice == 'value':
            params['max_depth'] = trial.suggest_int('max_depth', 10, 50)
        
        model = RandomForestClassifier(**params)
        
        # Cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        
        return scores.mean()
    
    print("\n Ejecutando Optuna (50 trials, ~1-2 minutos)...")
    start_time = time.time()
    
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='random_forest_tuning'
    )
    
    # Optimizar
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    optuna_time = time.time() - start_time
    
    print(f"\n✅ Optuna completado en {optuna_time:.2f} segundos")
    print(f"\n🏆 MEJORES HIPERPARÁMETROS (Optuna):")
    print(f"  {study.best_params}")
    print(f"\n📊 Mejor F1-Score (CV): {study.best_value:.4f}")
    
    # Entrenar modelo con mejores parámetros
    best_params_optuna = study.best_params.copy()
    best_params_optuna.pop('max_depth_choice')
    best_params_optuna['random_state'] = 42
    best_params_optuna['n_jobs'] = -1
    
    rf_optuna = RandomForestClassifier(**best_params_optuna)
    rf_optuna.fit(X_train, y_train)
    
    y_pred_optuna = rf_optuna.predict(X_test)
    acc_optuna = accuracy_score(y_test, y_pred_optuna)
    f1_optuna = f1_score(y_test, y_pred_optuna)
    
    print(f"Accuracy en test: {acc_optuna:.4f}")
    print(f"F1-Score en test: {f1_optuna:.4f}")
    
    # Visualizaciones de Optuna
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image("optuna_history.png")
        print("\n Gráfico de optimización guardado: optuna_history.png")
        
        fig2 = plot_param_importances(study)
        fig2.write_image("optuna_importance.png")
        print(" Importancia de parámetros guardada: optuna_importance.png")
    except Exception as e:
        print(f"\n No se pudieron guardar las visualizaciones: {e}")
else:
    print("\n Saltando Optuna (no instalado)")
    acc_optuna, f1_optuna = None, None


# 6. Comparación Final
print("\n" + "="*70)
print(" Comparación Final")
print("="*70)

comparacion = {
    'Método': ['Base (default)', 'GridSearchCV', 'RandomizedSearchCV'],
    'Accuracy': [acc_base, acc_grid, acc_random],
    'F1-Score': [f1_base, f1_grid, f1_random],
    'Tiempo (s)': [0.0, grid_time, random_time]
}

if OPTUNA_AVAILABLE:
    comparacion['Método'].append('Optuna')
    comparacion['Accuracy'].append(acc_optuna)
    comparacion['F1-Score'].append(f1_optuna)
    comparacion['Tiempo (s)'].append(optuna_time)

df_comparacion = pd.DataFrame(comparacion)
print(df_comparacion.to_string(index=False))

# Calcular mejoras
mejora_f1_grid = ((f1_grid - f1_base) / f1_base) * 100
mejora_f1_random = ((f1_random - f1_base) / f1_base) * 100
mejora_f1_optuna = ((f1_optuna - f1_base) / f1_base) * 100 if OPTUNA_AVAILABLE else 0

print(f"\n Mejora en F1-Score vs Base:")
print(f"  GridSearchCV:     +{mejora_f1_grid:.2f}%")
print(f"  RandomizedSearch: +{mejora_f1_random:.2f}%")
if OPTUNA_AVAILABLE:
    print(f"  Optuna:           +{mejora_f1_optuna:.2f}%")


# 7. Visualización De Resultados
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Accuracy comparado
methods = df_comparacion['Método']
acc_values = df_comparacion['Accuracy']
colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71'][:len(methods)]

axes[0, 0].bar(methods, acc_values, color=colors, edgecolor='black')
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title(' Accuracy por Método', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(min(acc_values) * 0.99, max(acc_values) * 1.01)
axes[0, 0].grid(axis='y', alpha=0.3)

for i, v in enumerate(acc_values):
    axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

# 2. F1-Score comparado
f1_values = df_comparacion['F1-Score']
axes[0, 1].bar(methods, f1_values, color=colors, edgecolor='black')
axes[0, 1].set_ylabel('F1-Score', fontsize=12)
axes[0, 1].set_title('📊 F1-Score por Método', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(min(f1_values) * 0.99, max(f1_values) * 1.01)
axes[0, 1].grid(axis='y', alpha=0.3)
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

for i, v in enumerate(f1_values):
    axes[0, 1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

# 3. Tiempo de ejecución
times = df_comparacion['Tiempo (s)']
axes[1, 0].bar(methods, times, color=colors, edgecolor='black')
axes[1, 0].set_ylabel('Tiempo (segundos)', fontsize=12)
axes[1, 0].set_title('⏱️ Tiempo de Búsqueda', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

for i, v in enumerate(times):
    axes[1, 0].text(i, v + max(times)*0.01, f'{v:.1f}s', ha='center', fontweight='bold')

# 4. Trade-off: Tiempo vs Calidad
if OPTUNA_AVAILABLE:
    axes[1, 1].scatter(times[1:], f1_values[1:], s=200, c=colors[1:], alpha=0.6, edgecolors='black')
    axes[1, 1].set_xlabel('Tiempo (segundos)', fontsize=12)
    axes[1, 1].set_ylabel('F1-Score', fontsize=12)
    axes[1, 1].set_title('⚖️ Trade-off: Tiempo vs Calidad', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    for i, method in enumerate(methods[1:], 1):
        axes[1, 1].annotate(method, (times[i], f1_values[i]), 
                           fontsize=9, ha='center', fontweight='bold')

plt.suptitle('🏆 Comparación de Métodos de Hyperparameter Tuning', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# 8. Lecciones Clave
print("\n" + "="*70)
print("💡 Lecciones Clave")
print("="*70)

print(f"""
1. GridSearchCV vs RandomizedSearchCV:
   - GridSearch: Exhaustivo pero LENTO ({grid_time:.1f}s)
   - RandomSearch: Rápido ({random_time:.1f}s) y encuentra buenos resultados
   - RandomSearch es 5-10x más rápido con resultados similares

2. Optuna (si lo instalaste):
   - El más inteligente (aprende de trials anteriores)
   - Encuentra óptimos en menos iteraciones
   - Recomendado para proyectos serios

3. Mejora obtenida:
   - Base:        F1={f1_base:.4f}
   - Optimizado:  F1={max(f1_grid, f1_random, f1_optuna if OPTUNA_AVAILABLE else 0):.4f}
   - Mejora:      +{max(mejora_f1_grid, mejora_f1_random, mejora_f1_optuna if OPTUNA_AVAILABLE else 0):.1f}%

4. ¿Cuándo usar cada uno?
   - GridSearch:     Cuando tienes POCOS parámetros y tiempo
   - RandomSearch:   Cuando tienes MUCHOS parámetros (default recomendado)
   - Optuna:         Cuando quieres el MEJOR resultado posible

5. Consejo profesional:
   - Empieza con RandomizedSearchCV (n_iter=20-50)
   - Si necesitas más precisión, usa Optuna
   - GridSearch solo para grids pequeños (<100 combinaciones)

RECOMENDACIÓN: Para producción, usa RandomizedSearchCV o Optuna.
GridSearch es bueno para aprender, pero poco práctico en la vida real.
""")

print("\n✅ Día 14 completado!")