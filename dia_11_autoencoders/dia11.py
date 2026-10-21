# Tener instalados las librerias de numpy, matplotlib, tensorflow y sklearn (el tensorflow puede instalarse con "pip install tensorflow" en la terminal de vscode)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA


# 1. Cargar Dataset MNIST
print(" Cargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar a [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Aplanar imágenes: (28, 28) → (784,)
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

print(f"Train: {x_train_flat.shape}")
print(f"Test: {x_test_flat.shape}")


# 2. Visualizar digitos ejemplos 
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[idx], cmap='gray')
    ax.set_title(f'Dígito {idx+1}')
    ax.axis('off')
plt.suptitle('Ejemplos de MNIST (28x28 píxeles)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 3. Construir el autoencoder
print("\n" + "="*60)
print(" Contruyendo el Autoencoder...")
print("="*60)

# Dimensión del bottleneck
BOTTLENECK_DIM = 32

# Encoder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
bottleneck = Dense(BOTTLENECK_DIM, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(bottleneck)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# Modelo completo
autoencoder = Model(input_img, decoded)

# Modelo encoder (para visualizar el bottleneck)
encoder = Model(input_img, bottleneck)

# Modelo decoder (para reconstruir desde bottleneck)
decoder_input = Input(shape=(BOTTLENECK_DIM,))
decoder_layer = autoencoder.layers[-3](decoder_input)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_output = autoencoder.layers[-1](decoder_layer)
decoder = Model(decoder_input, decoder_output)

autoencoder.summary()


# 4. Entrenar el autoencoder
print("\n🚀 Entrenando autoencoder...")
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(
    x_train_flat, x_train_flat,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat),
    verbose=1
)

# Visualizar pérdida durante entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(' Pérdida durante el entrenamiento', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss (Binary Crossentropy)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 5. Visualizar reconstrucciones
print("\n Visualizando reconstrucciones...")
decoded_imgs = autoencoder.predict(x_test_flat)

fig, axes = plt.subplots(3, 10, figsize=(15, 5))

for i in range(10):
    # Imagen original
    axes[0, i].imshow(x_test[i], cmap='gray')
    axes[0, i].set_title(f'Original {i}', fontsize=8)
    axes[0, i].axis('off')
    
    # Imagen reconstruida
    axes[1, i].imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f'Reconstruida {i}', fontsize=8)
    axes[1, i].axis('off')
    
    # Diferencia (error)
    diff = np.abs(x_test[i] - decoded_imgs[i].reshape(28, 28))
    axes[2, i].imshow(diff, cmap='hot')
    axes[2, i].set_title(f'Error {i}', fontsize=8)
    axes[2, i].axis('off')

plt.suptitle(' Original vs Reconstruida vs Error', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 6. Comparación con PCA
print("\n" + "="*60)
print(" Comparacón: Autoencoder vs PCA")
print("="*60)

# PCA con 32 componentes (misma dimensión que el bottleneck)
pca = PCA(n_components=BOTTLENECK_DIM)
x_test_pca = pca.fit_transform(x_test_flat)
x_test_pca_reconstructed = pca.inverse_transform(x_test_pca)

fig, axes = plt.subplots(3, 10, figsize=(15, 5))

for i in range(10):
    # Original
    axes[0, i].imshow(x_test[i], cmap='gray')
    axes[0, i].set_title(f'Original', fontsize=8)
    axes[0, i].axis('off')
    
    # Autoencoder
    axes[1, i].imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f'Autoencoder', fontsize=8)
    axes[1, i].axis('off')
    
    # PCA
    axes[2, i].imshow(x_test_pca_reconstructed[i].reshape(28, 28), cmap='gray')
    axes[2, i].set_title(f'PCA (32 comp)', fontsize=8)
    axes[2, i].axis('off')

plt.suptitle('Autoencoder (32 dim) vs PCA (32 componentes)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Calcular error de reconstrucción
ae_error = np.mean(np.abs(x_test_flat - decoded_imgs))
pca_error = np.mean(np.abs(x_test_flat - x_test_pca_reconstructed))

print(f"\nError de reconstrucción (MAE):")
print(f"  Autoencoder: {ae_error:.6f}")
print(f"  PCA:         {pca_error:.6f}")
print(f"\n{' Autoencoder gana!' if ae_error < pca_error else 'PCA gana (sorprendente para datos lineales)'}")


# 7. Visualizar el bottleneck 
print("\n Visualizando el espacio del bottleneck...")
bottleneck_features = encoder.predict(x_test_flat)

# Usar PCA para reducir el bottleneck de 32D a 2D para visualizar
bottleneck_2d = PCA(n_components=2).fit_transform(bottleneck_features)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(bottleneck_2d[:, 0], bottleneck_2d[:, 1], 
                      c=y_test, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(scatter, label='Dígito')
plt.title(f' Espacio del Bottleneck (32D → 2D con PCA)\nCada punto es una imagen comprimida', 
          fontsize=14, fontweight='bold')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Día 11 completado!")
print("💡 El autoencoder aprendió a comprimir imágenes de 784 → 32 dimensiones")
print("   y reconstruirlas con poca pérdida de información.")