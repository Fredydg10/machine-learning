# Tener instaladas las librerias de: Pandas, Numpy, Matplotlib y Seaborn( adicional descargar la libreria de tensorflow)
# type: ignore (Este script para eliminar de las advertencias de tipo en el código, ya que algunas funciones de Keras pueden generar advertencias de tipo que no afectan la ejecución del script.) 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.activations import relu, sigmoid, tanh, softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd
import time


# 1. Cargar dataset MNIST
print("Cargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Train: {x_train.shape}, Test: {x_test.shape}")
print(f"Labels train: {y_train.shape}, Labels test: {y_test.shape}")


# 2. Visualizar datos ejemplo
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[idx], cmap='gray')
    ax.set_title(f'Dígito: {y_train[idx]}')
    ax.axis('off')
plt.suptitle(' Dataset MNIST (28x28 píxeles)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 3. Preparar datos
print("\n Preparando datos...")

# Aplanar imágenes: (28, 28) → (784,)
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

# Normalizar a [0, 1]
x_train_norm = x_train_flat.astype('float32') / 255.0
x_test_norm = x_test_flat.astype('float32') / 255.0

# One-hot encoding para las etiquetas
# Ej: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

print(f"Shape después de aplanar: {x_train_norm.shape}")
print(f"Shape después de one-hot: {y_train_cat.shape}")


# 4. Explicar arquitectura de la red 
print("\n" + "="*70)
print("Arquitectura de la red neuronal")
print("="*70)

print("""
Input Layer (784 neuronas)
    ↓
Dense 128 + ReLU (128 neuronas)
    ↓
Dropout 0.2 (apaga 20% de neuronas aleatoriamente)
    ↓
Dense 64 + ReLU (64 neuronas)
    ↓
Dropout 0.2
    ↓
Dense 10 + Softmax (10 neuronas, una por dígito)
    ↓
Output: Probabilidades [p(0), p(1), ..., p(9)]
""")


# 5. Crear Modelo 1: FNN Simple
print("\n" + "="*70)
print("Modelo 1: FNN Simple (2 capas)")
print("="*70)

model_simple = Sequential([
    Flatten(input_shape=(28, 28)),  # Aplanar 28x28 → 784
    Dense(128, activation='relu'),   # Capa oculta 1
    Dense(10, activation='softmax')  # Capa de salida
])

model_simple.summary()

# Compilar
model_simple.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
print("\n Entrenando Modelo Simple...")
history_simple = model_simple.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluar
test_loss_simple, test_acc_simple = model_simple.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n Accuracy en test: {test_acc_simple:.4f} ({test_acc_simple*100:.2f}%)")


# 6. Crear Modelo 2: FNN Profundo ( Con dropout) 
print("\n" + "="*70)
print("🏗️  MODELO 2: FNN Profundo (3 capas + Dropout)")
print("="*70)

model_deep = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.2),  # Regularización: apaga 20% de neuronas
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model_deep.summary()

# Compilar
model_deep.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
print("\n Entrenando Modelo Profundo...")
history_deep = model_deep.fit(
    x_train, y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluar
test_loss_deep, test_acc_deep = model_deep.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n Accuracy en test: {test_acc_deep:.4f} ({test_acc_deep*100:.2f}%)")


# 7. Comparar Funciones de activación
print("\n" + "="*70)
print("Experimento: Funciones de Activación")
print("="*70)

activations = ['relu', 'tanh', 'sigmoid']
results_activation = []

for act in activations:
    print(f"\n Probando {act.upper()}...")
    
    model_act = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=act),
        Dense(10, activation='softmax')
    ])
    
    model_act.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history = model_act.fit(
        x_train, y_train_cat,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )
    
    train_time = time.time() - start_time
    _, acc = model_act.evaluate(x_test, y_test_cat, verbose=0)
    
    results_activation.append({
        'Activación': act.upper(),
        'Accuracy': acc,
        'Tiempo (5 epochs)': f"{train_time:.2f}s"
    })
    
    print(f"  Accuracy: {acc:.4f}, Tiempo: {train_time:.2f}s")

df_activation = pd.DataFrame(results_activation)
print("\n" + "="*70)
print("📊 Comparación de Funciones de Activación")
print("="*70)
print(df_activation.to_string(index=False))


# 8. Visualizar resultados
print("\n" + "="*70)
print(" Visualización de resultados")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Accuracy por época (Modelo Simple)
axes[0, 0].plot(history_simple.history['accuracy'], label='Train Acc', color='blue')
axes[0, 0].plot(history_simple.history['val_accuracy'], label='Val Acc', color='orange')
axes[0, 0].set_title('📊 Modelo Simple: Accuracy por Época', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Época')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Accuracy por época (Modelo Profundo)
axes[0, 1].plot(history_deep.history['accuracy'], label='Train Acc', color='green')
axes[0, 1].plot(history_deep.history['val_accuracy'], label='Val Acc', color='red')
axes[0, 1].set_title(' Modelo Profundo: Accuracy por Época', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Comparación de activaciones
act_names = df_activation['Activación']
acc_values = df_activation['Accuracy']
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

axes[1, 0].bar(act_names, acc_values, color=colors, edgecolor='black')
axes[1, 0].set_ylabel('Accuracy', fontsize=12)
axes[1, 0].set_title('🔬 Funciones de Activación', fontsize=12, fontweight='bold')
axes[1, 0].set_ylim(0.9, 0.99)
axes[1, 0].grid(axis='y', alpha=0.3)

for i, v in enumerate(acc_values):
    axes[1, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

# 4. Predicciones del modelo profundo
y_pred = model_deep.predict(x_test[:10])
y_pred_classes = np.argmax(y_pred, axis=1)

for i in range(10):
    ax = fig.add_subplot(2, 10, i+11)
    ax.imshow(x_test[i], cmap='gray')
    ax.set_title(f'Pred: {y_pred_classes[i]}\nReal: {y_test[i]}', 
                 fontsize=8, color='green' if y_pred_classes[i] == y_test[i] else 'red')
    ax.axis('off')

plt.suptitle('Predicciones del Modelo Profundo', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 9. Mostrar arquitectura detallada
print("\n" + "="*70)
print("Arquitectura final del modelo profundo")
print("="*70)

print("""
┌─────────────────────────────────────────────────┐
│            INPUT LAYER (784)                    │
│  28×28 píxeles aplanados                        │
└─────────────────────┬───────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────┐
│         DENSE 512 + ReLU                        │
│  784 → 512 neuronas                             │
│  Parámetros: 401,920                            │
└─────────────────────┬───────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────┐
│         DROPOUT (0.2)                           │
│  Apaga 20% de neuronas aleatoriamente           │
└─────────────────────┬───────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────┐
│         DENSE 256 + ReLU                        │
│  512 → 256 neuronas                             │
│  Parámetros: 131,328                            │
└─────────────────────┬───────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────┐
│         DROPOUT (0.2)                           │
└─────────────────────┬───────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────┐
│         DENSE 128 + ReLU                        │
│  256 → 128 neuronas                             │
│  Parámetros: 32,896                             │
└─────────────────────┬───────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────┐
│         DENSE 10 + Softmax                      │
│  128 → 10 neuronas (una por dígito)            │
│  Parámetros: 1,290                              │
└─────────────────────────────────────────────────┘
                      │
                      ↓
         OUTPUT: [p(0), p(1), ..., p(9)]
         Suma de probabilidades = 1
""")

print(f"\nTotal de parámetros entrenables: {model_deep.count_params():,}")


# 10. Lecciónes Clave 
print("\n" + "="*70)
print("Lecciónes Clave")
print("="*70)

print(f"""
1. Perceptrón = Neurona artificial:
   - Recibe entradas
   - Multiplica por pesos
   - Suma + bias
   - Aplica activación
   - Produce salida

2. FNN = Múltiples perceptrones en capas:
   - Información fluye en una dirección (feedforward)
   - Cada capa transforma los datos
   - Última capa da la predicción

3. Funciones de activación (no linealidad):
   - ReLU: La más usada en hidden layers (rápida, efectiva)
   - Sigmoid: Para clasificación binaria (0-1)
   - Tanh: Como sigmoid pero centrada en 0 (-1 a 1)
   - Softmax: Para multi-clase (probabilidades que suman 1)

4. Dropout = Regularización:
   - Apaga neuronas aleatoriamente durante entrenamiento
   - Previene overfitting
   - Fuerza a la red a ser robusta

5. Comparación de modelos:
   - Modelo Simple:   {test_acc_simple*100:.2f}% accuracy
   - Modelo Profundo: {test_acc_deep*100:.2f}% accuracy
   - Mejora: +{(test_acc_deep - test_acc_simple)*100:.2f}%

6. Forward pass:
   - Es el proceso de predicción
   - Datos → Capa 1 → Capa 2 → ... → Output
   - Cada capa aplica: z = W·x + b, a = f(z)

CONCLUSIÓN: Las FNN son poderosas pero tienen millones de parámetros.
Para imágenes, las CNN (Día 16+) son más eficientes.
""")

print("\n✅ Día 15 completado!")