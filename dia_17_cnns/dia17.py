import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import time

print(f"TensorFlow versión: {tf.__version__}")


# 1. Cargar y preparar los datos
print("\n Cargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# CNN necesita datos en formato (samples, height, width, channels)
# MNIST es en escala de grises, así que channels=1
x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Shape para CNN: {x_train_cnn.shape}")


# 2. Visualizar filtros de convulcion
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[idx], cmap='gray')
    ax.set_title(f'Dígito: {y_train[idx]}')
    ax.axis('off')
plt.suptitle('Dataset MNIST (28x28 píxeles)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 3. Crear modelo CNN
print("\n" + "="*70)
print("Arquitectura CNN")
print("="*70)

print("""
Input (28, 28, 1) - Imagen en escala de grises
    ↓
Conv2D(32 filtros, 3x3) + ReLU - Detecta bordes y texturas básicas
    ↓
MaxPooling2D(2x2) - Reduce dimensionalidad a la mitad
    ↓
Conv2D(64 filtros, 3x3) + ReLU - Detecta formas más complejas
    ↓
MaxPooling2D(2x2) - Reduce más
    ↓
Flatten - Aplanar para capas densas
    ↓
Dense(128) + ReLU + Dropout(0.5) - Clasificación
    ↓
Dense(10) + Softmax - 10 clases (dígitos 0-9)
""")

# Modelo CNN
# Modelo CNN (Formato moderno Keras 3)
cnn_model = Sequential([
    # 1. Capa de entrada explícita (esto elimina el error de Pylance)
    Input(shape=(28, 28, 1)),
    
    # 2. Primera capa convolucional (ya no necesita input_shape)
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 3. Segunda capa convolucional
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 4. Capas densas para clasificación
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

cnn_model.summary()


# 4. Compilar y entrenar
print("\n Compilando Modelo...")
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early Stopping para evitar overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print("\n Entrenando CNN (esto puede tardar 1-2 minutos)...")
start_time = time.time()

history_cnn = cnn_model.fit(
    x_train_cnn, y_train_cat,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

train_time = time.time() - start_time
print(f"\n Entrenamiento completado en {train_time:.2f} segundos")


# 5. Evaluar Modelo
print("\n" + "="*70)
print("Evaluación del modelo")
print("="*70)

test_loss, test_acc = cnn_model.evaluate(x_test_cnn, y_test_cat, verbose=0)
print(f"\n🏆 Accuracy en test: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"📉 Loss en test: {test_loss:.4f}")


# 6. Visualizar curvas de aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history_cnn.history['accuracy'], label='Train Acc', color='blue', linewidth=2)
axes[0].plot(history_cnn.history['val_accuracy'], label='Val Acc', color='orange', linewidth=2)
axes[0].set_title('📈 Accuracy por Época', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history_cnn.history['loss'], label='Train Loss', color='blue', linewidth=2)
axes[1].plot(history_cnn.history['val_loss'], label='Val Loss', color='orange', linewidth=2)
axes[1].set_title('📉 Loss por Época', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 7. Visualizar predicciones
print("\n Visualizando predicciones...")
y_pred = cnn_model.predict(x_test_cnn[:10])
y_pred_classes = np.argmax(y_pred, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(x_test[idx], cmap='gray')
    pred_class = y_pred_classes[idx]
    real_class = y_test[idx]
    color = 'green' if pred_class == real_class else 'red'
    ax.set_title(f'Pred: {pred_class}\nReal: {real_class}', 
                 fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle('🎯 Predicciones de la CNN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 8. Comparar CNN VS FNN
print("\n" + "="*70)
print("CNN vs FNN (Feedforward Neural Network)")
print("="*70)

print("""
─────────────────────────────────────────────────────────┐
│  FNN (Día 15): ~98% accuracy                            │
│  - Aplana la imagen (pierde estructura espacial)        │
│  - 400,000+ parámetros                                  │
│  - No detecta patrones locales                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  CNN (Hoy): ~99%+ accuracy                              │
│  - Mantiene estructura 2D de la imagen                  │
│  - ~100,000 parámetros (4x menos que FNN)               │
│  - Detecta bordes, texturas, formas automáticamente     │
└─────────────────────────────────────────────────────────┘
""")

print(f"✅ CNN Accuracy: {test_acc*100:.2f}%")
print("💡 La CNN es más eficiente y precisa que la FNN para imágenes")


# 9. Lecciones Clave
print("\n" + "="*70)
print(" Lecciónes claves del día 17")
print("="*70)

print("""
1. Convolución (Conv2D):
   - Aplica filtros (kernels) que detectan patrones locales
   - Cada filtro aprende a detectar algo diferente (bordes, texturas)
   - Los filtros se mueven por toda la imagen (sliding window)

2. Pooling (MaxPooling2D):
   - Reduce la dimensionalidad (hace la imagen más pequeña)
   - Hace el modelo más rápido y resistente a pequeñas variaciones
   - MaxPooling toma el valor máximo de cada región

3. Arquitectura típica de CNN:
   - Bloques de [Conv2D + ReLU + MaxPooling] repetidos
   - Al final: Flatten + Dense para clasificación
   - Dropout para evitar overfitting

4. Ventajas de CNN sobre FNN para imágenes:
   - Menos parámetros (comparte pesos entre posiciones)
   - Detecta patrones locales (bordes, formas)
   - Invariante a traslaciones (reconoce objetos en cualquier posición)

5. Early Stopping:
   - Detiene el entrenamiento cuando val_loss deja de mejorar
   - Previene overfitting automáticamente
   - Restaura los mejores pesos

CONCLUSIÓN: Las CNN son el estándar para visión por computadora.
Usa FNN para datos tabulares, CNN para imágenes.
""")

print("\n✅ Día 17 completado!")