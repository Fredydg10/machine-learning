#Tener instaladas las librerias de pandas,sklearn,seaborn y matplotlib. 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Cargar Dataset Titanic
print("Cargando dataset Titanic...")
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("❌ Error: No encuentro 'train.csv'. Ponlo en la misma carpeta.")
    exit()

print(f"Shape original: {df.shape}")


# 2. Limpieza de datos
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print(f"Shape después de limpieza: {df.shape}")


# 3. Preparar X e y
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar (Naive Bayes no lo necesita, pero Random Forest tampoco)
# Lo hacemos por consistencia
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n Train: {X_train.shape[0]} muestras")
print(f" Test:  {X_test.shape[0]} muestras")


# 4. Experimento: efecto de n_estimators en Random Forest
valores_arboles = [10, 50, 100, 500, 1000]
resultados_arboles = []

print("\n" + "="*60)
print("Experimento de n_estimators (Random Forest)")
print("="*60)

for n_arboles in valores_arboles:
    modelo_rf = RandomForestClassifier(
        n_estimators=n_arboles,
        max_depth=10,
        random_state=42
    )
    modelo_rf.fit(X_train, y_train)
    predicciones = modelo_rf.predict(X_test)
    resultados_arboles.append({
        "Árboles": n_arboles,
        "Accuracy": accuracy_score(y_test, predicciones)
    })

df_arboles = pd.DataFrame(resultados_arboles)
accuracies_arboles = [resultado["Accuracy"] for resultado in resultados_arboles]
mejoras_arboles = [
    accuracies_arboles[indice] - accuracies_arboles[indice - 1]
    for indice in range(1, len(accuracies_arboles))
]
df_arboles["Mejora vs. anterior"] = [
    "—",
    *(f"{mejora:+.4f}" for mejora in mejoras_arboles)
]
print(df_arboles.to_string(index=False, formatters={"Accuracy": "{:.4f}".format}))

# Tomamos 0.01 (un punto porcentual) como umbral de mejora significativa.
umbral_mejora = 0.01
punto_estable = next(
    (
        indice
        for indice in range(len(valores_arboles))
        if max(accuracies_arboles[indice:]) - min(accuracies_arboles[indice:])
        < umbral_mejora
    ),
    None
)
if punto_estable is None:
    print(
        "El accuracy varía más del "
        f"{umbral_mejora:.0%} en todo el rango probado."
    )
else:
    rango_accuracy = (
        max(accuracies_arboles[punto_estable:])
        - min(accuracies_arboles[punto_estable:])
    )
    print(
        f"Diminishing returns: desde {valores_arboles[punto_estable]} árboles, "
        f"el accuracy se mantiene en un rango de "
        f"{rango_accuracy * 100:.2f} puntos porcentuales "
        f"(menos de {umbral_mejora * 100:.0f} pp)."
    )


# 5. Entrenar modelos
modelos = {
    "Random Forest (100 árboles)": RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    ),
    "Random Forest (500 árboles)": RandomForestClassifier(
        n_estimators=500, 
        max_depth=10, 
        random_state=42
    ),
    "Naive Bayes (Gaussian)": GaussianNB()
}

# Un bosque promedia muchos árboles entrenados con muestras y variables aleatorias.
# Esa diversidad reduce la varianza y el sobreajuste frente a un solo árbol,
# por lo que suele generalizar mejor (aunque no garantiza mayor accuracy en todo conjunto).
resultados = []

print("\n" + "="*60)
print("Entrenando Modelos...")
print("="*60)

for nombre, modelo in modelos.items():
    print(f"\n Entrenando: {nombre}")
    
    # Random Forest no necesita escalado, pero Naive Bayes tampoco
    # Usamos los datos sin escalar para ambos
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    
    print(f"   Accuracy: {acc:.4f}")
    
    resultados.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "modelo_obj": modelo,
        "y_pred": y_pred
    })

# Naive Bayes con escalado: comparamos la misma partición, cambiando solo X.
modelo_nb_escalado = GaussianNB()
modelo_nb_escalado.fit(X_train_scaled, y_train)
y_pred_nb_escalado = modelo_nb_escalado.predict(X_test_scaled)
acc_nb_escalado = accuracy_score(y_test, y_pred_nb_escalado)
print("\nComparación Naive Bayes:")
print(f"   Sin escalado: {resultados[2]['Accuracy']:.4f}")
print(f"   Con escalado: {acc_nb_escalado:.4f}")
print(f"   Diferencia:   {acc_nb_escalado - resultados[2]['Accuracy']:+.4f}")
resultados.append({
    "Modelo": "Naive Bayes (Gaussian, escalado)",
    "Accuracy": acc_nb_escalado,
    "modelo_obj": modelo_nb_escalado,
    "y_pred": y_pred_nb_escalado
})


# 6. Tabla Comparativa
df_resultados = pd.DataFrame(resultados)
print("\n" + "="*60)
print("Comparación De Modelos")
print("="*60)
print(df_resultados[['Modelo', 'Accuracy']].to_string(index=False))


# 7. Matrices de Confusión
for inicio in range(0, len(resultados), 2):
    grupo = resultados[inicio:inicio + 2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, info in zip(axes, grupo):
        cm = confusion_matrix(y_test, info['y_pred'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Murió (0)', 'Sobrevivió (1)'],
                    yticklabels=['Murió (0)', 'Sobrevivió (1)'])
        ax.set_title(f"{info['Modelo']}\n(Accuracy: {info['Accuracy']:.3f})")
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.tick_params(axis='x', labelrotation=0)
        ax.tick_params(axis='y', labelrotation=0)

    fig.suptitle(
        f"Titanic: matrices de confusión ({inicio + 1}-{inicio + len(grupo)} de {len(resultados)})",
        fontsize=14,
        fontweight='bold'
    )
    plt.show()


# 8. Importancia de Features (solo para Random Forest)
print("\n" + "="*60)
print("🔍 IMPORTANCIA DE FEATURES (Random Forest con 500 árboles)")
print("="*60)

rf_model = resultados[1]['modelo_obj']  # El de 500 árboles
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))
print(
    f"\nFeature más importante: {feature_importance.iloc[0]['Feature']} "
    f"({feature_importance.iloc[0]['Importance']:.3f})."
)
print(
    "Sex_male es coherente con la prioridad de mujeres y niños, "
    "aunque la importancia no indica por sí sola la dirección del efecto."
)

# Visualizar importancia
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Importancia de Features en Random Forest')
plt.xlabel('Importancia')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# 9. Reporte detallado del mejor modelo
mejor = max(resultados, key=lambda x: x['Accuracy'])
print("\n" + "="*60)
print(f"MEJOR MODELO: {mejor['Modelo']}")
print("="*60)
print(classification_report(y_test, mejor['y_pred'], 
                            target_names=['Murió (0)', 'Sobrevivió (1)']))
