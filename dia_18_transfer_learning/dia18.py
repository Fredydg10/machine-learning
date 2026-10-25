import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import time

print(f"TensorFlow versión: {tf.__version__}")


# 1. Cargar dataset CIFRAR-10
print("\n Cargando dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Nombres de las clases
class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# Normalizar a [0, 1]
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0


# 2. Visualizar datos de ejemplo
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[idx])
    ax.set_title(class_names[y_train[idx][0]], fontsize=10, fontweight='bold')
    ax.axis('off')
plt.suptitle('Dataset CIFAR-10 (Imágenes reales a color)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 3. Data Augmentation
print("\n" + "="*70)
print("Data Augmentation")
print("="*70)

# Creamos un modelo secuencial SOLO para aumento de datos
data_augmentation = Sequential([
    Input(shape=(32, 32, 3)),
    RandomFlip("horizontal"),          # Volteo horizontal aleatorio
    RandomRotation(0.1),               # Rotación de hasta 10% (36 grados)
    RandomZoom(0.1),                   # Zoom aleatorio de hasta 10%
], name="data_augmentation")

print("✅ Pipeline de aumento de datos creado.")
print("   (Se aplica en tiempo real durante el entrenamiento)")


# 4. Crear modelo con transfer learning
print("\n" + "="*70)
print(" Transfer Learning con MobileNetV2")
print("="*70)

# Descargamos MobileNetV2 pre-entrenado en ImageNet
# include_top=False: quitamos la capa de clasificación original
# weights='imagenet': usamos los pesos aprendidos de millones de imágenes
print(" Descargando pesos pre-entrenados de MobileNetV2 (primera vez tarda unos segundos)...")
base_model = MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights='imagenet'
)

# CONGELAMOS la base para no destruir los pesos pre-entrenados
base_model.trainable = False
print(f"✅ Base model cargado. Capas congeladas: {base_model.trainable == False}")

# Construimos nuestro modelo final
print("\n Construyendo cabeza clasificadora personalizada...")
model = Sequential([
    Input(shape=(32, 32, 3)),
    data_augmentation,          # 1. Aumento de datos
    base_model,                 # 2. Extracción de características (congelado)
    GlobalAveragePooling2D(),   # 3. Promedio espacial (mejor que Flatten para Transfer Learning)
    Dropout(0.2),               # 4. Regularización
    Dense(10, activation='softmax') # 5. Clasificación (10 clases de CIFAR-10)
])

model.summary()


# 5. Compilar y entrenar 
print("\n Compilando modelo...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', # Usamos sparse porque y_train no está en one-hot
    metrics=['accuracy']
)

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

print("\n Entrenando (usando Transfer Learning + Augmentation)...")
print("   (Nota: Usamos un subset de 10,000 imágenes para que sea rápido en CPU)")
start_time = time.time()

# Subset para entrenamiento rápido en local
x_train_sub = x_train_norm[:10000]
y_train_sub = y_train[:10000]
x_test_sub = x_test_norm[:2000]
y_test_sub = y_test[:2000]

history = model.fit(
    x_train_sub, y_train_sub,
    epochs=10, # En un proyecto real serían 30-50
    batch_size=64,
    validation_data=(x_test_sub, y_test_sub),
    callbacks=[early_stopping],
    verbose=1
)

train_time = time.time() - start_time
print(f"\n Entrenamiento completado en {train_time:.2f} segundos")


# 6. Evaluar y visualizar
print("\n" + "="*70)
print("Evaluación del modelo")
print("="*70)

test_loss, test_acc = model.evaluate(x_test_sub, y_test_sub, verbose=0)
print(f"\n Accuracy en test: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Acc', color='blue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='orange', linewidth=2)
axes[0].set_title('📈 Accuracy por Época', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', color='orange', linewidth=2)
axes[1].set_title('📉 Loss por Época', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 7. Visualizar predicciones con aumento 
print("\n Visualizando predicciones en datos de prueba...")
y_pred_probs = model.predict(x_test_sub[:10])
y_pred_classes = np.argmax(y_pred_probs, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(x_test_sub[idx])
    pred_name = class_names[y_pred_classes[idx]]
    real_name = class_names[y_test_sub[idx][0]]
    color = 'green' if pred_name == real_name else 'red'
    ax.set_title(f'Pred: {pred_name}\nReal: {real_name}', 
                 fontsize=9, color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle('Predicciones del Modelo con Transfer Learning', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 8. Lecciónes Clave
print("\n" + "="*70)
print("Lecciones clave del día 18")
print("="*70)

print("""
1. Transfer Learning es el estándar de la industria:
   - Ahorra días/semanas de entrenamiento.
   - Funciona increíblemente bien con pocos datos.
   - MobileNetV2 es ligero; ResNet50 o EfficientNet son más potentes.

2. Data Augmentation en Keras 3:
   - Se usa como capas del modelo (RandomFlip, RandomZoom).
   - Se ejecuta en la GPU, mucho más rápido que el antiguo ImageDataGenerator.
   - Solo se aplica durante el entrenamiento (no en la inferencia).

3. GlobalAveragePooling2D vs Flatten:
   - En Transfer Learning, GlobalAveragePooling2D es casi siempre mejor.
   - Promedia las características espaciales, reduciendo drásticamente los parámetros y el overfitting.

4. Fine-Tuning (Próximo paso avanzado):
   - Después de entrenar la cabeza, puedes descongelar las últimas capas 
     de la base_model (base_model.trainable = True) y entrenar con un 
     learning rate MUY bajo (ej: 1e-5) para ajustar los detalles.

CONCLUSIÓN: Con solo 10,000 imágenes y un modelo pre-entrenado, 
logramos un rendimiento que a una red desde cero le tomaría 100,000 imágenes.
""")

print("\n✅ Día 18 completado!")