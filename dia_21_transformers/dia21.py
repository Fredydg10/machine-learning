import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Add,
)
from keras.callbacks import EarlyStopping
import time

print(f"TensorFlow versión: {tf.__version__}")


# 1. Crear dataset de secuencias sintéticas
print("\n Creando dataset de clasificación de secuencias...")
np.random.seed(42)

N_SAMPLES = 800
SEQ_LEN = 40
N_FEATURES = 8
N_CLASSES = 2

X = np.random.randn(N_SAMPLES, SEQ_LEN, N_FEATURES).astype("float32")
# Señal útil: la clase depende de la media de la primera feature
y = (X[:, :, 0].mean(axis=1) > 0).astype("int32")

split = int(0.8 * N_SAMPLES)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Clases train: {np.bincount(y_train)}")


# 2. Bloque Transformer (atención + residual + FFN)
print("\n" + "=" * 70)
print("Arquitectura del bloque Transformer")
print("=" * 70)
print("""
Input
  ↓
MultiHeadAttention
  ↓
Dropout
  ↓
Add (residual: Input + Attention) + LayerNorm
  ↓
Dense (FFN) → Dropout → Dense
  ↓
Add (residual) + LayerNorm
  ↓
Output
""")


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate):
    attention_output = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout_rate,
    )(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Add()([attention_output, ff_output])
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)

    return ff_output


def build_transformer(
    input_shape,
    head_size=16,
    num_heads=4,
    ff_dim=32,
    num_blocks=2,
    dropout_rate=0.1,
    mlp_units=32,
    n_classes=2,
):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(mlp_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(n_classes, activation="softmax")(x)
    return Model(inputs, outputs)


# 3. Crear y entrenar el modelo
print("\n Construyendo Transformer...")
model = build_transformer(input_shape=(SEQ_LEN, N_FEATURES))
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

print("\n Entrenando...")
start = time.time()
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=8,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)
elapsed = time.time() - start
print(f"\n Tiempo de entrenamiento: {elapsed:.1f}s")


# 4. Evaluar
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")


# 5. Curvas de aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_title("Loss", fontweight="bold")
axes[0].set_xlabel("Época")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Train")
axes[1].plot(history.history["val_accuracy"], label="Val")
axes[1].set_title("Accuracy", fontweight="bold")
axes[1].set_xlabel("Época")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Transformer — clasificación de secuencias", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()


# 6. Lecciones clave
print("\n" + "=" * 70)
print("Lecciones clave del día 21")
print("=" * 70)
print("""
1. Un Transformer no recorre la secuencia paso a paso (como RNN/LSTM).
   Self-attention mira todas las posiciones a la vez.

2. MultiHeadAttention: varias "cabezas" aprenden relaciones distintas
   (por ejemplo corto vs largo alcance).

3. Conexiones residuales (Add): suman la entrada con la salida del bloque
   para que el gradiente no se pierda en redes profundas.

4. LayerNormalization estabiliza las activaciones en cada bloque.

5. El FFN (capas Dense) transforma cada posición después de la atención.

6. GlobalAveragePooling1D resume la secuencia para clasificar.

CONCLUSIÓN: el bloque atención + residual + FFN es la pieza base
de BERT, GPT y los transformers de visión.
""")
