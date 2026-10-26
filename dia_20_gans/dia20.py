import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout,
    Conv2D, Conv2DTranspose,
    BatchNormalization, LeakyReLU, ReLU
)
from tensorflow.keras.optimizers import Adam
import time

print(f"TensorFlow versión: {tf.__version__}")


# 1. Cargar dataset MNIST
print("\n Cargando dataset MNIST...")
(x_train, _), (_, _) = mnist.load_data()

# Normalizar a [-1, 1] (importante para GANs con tanh)
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)  # Agregar canal: (28, 28, 1)

print(f"Shape de datos: {x_train.shape}")
print(f"Rango de valores: [{x_train.min():.2f}, {x_train.max():.2f}]")


# 2. Parámetros de la GAN
LATENT_DIM = 128        # Dimensión del ruido de entrada
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNELS = 1
BATCH_SIZE = 64
EPOCHS = 30             # En proyecto real: 100-200

print(f"\n Parámetros:")
print(f"   Dimensión latente: {LATENT_DIM}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Épocas: {EPOCHS}")


# 3. Crear el generador
print("\n" + "="*70)
print("Creando el Generador")
print("="*70)

print("""
Arquitectura del Generador:
Input (128) → ruido aleatorio
    ↓
Dense(7*7*128) + BatchNorm + ReLU
    ↓
Reshape(7, 7, 128)
    ↓
Conv2DTranspose(128, 5x5) + BatchNorm + ReLU  → 14x14
    ↓
Conv2DTranspose(64, 5x5) + BatchNorm + ReLU   → 28x28
    ↓
Conv2DTranspose(1, 5x5) + Tanh                → imagen final
""")

def build_generator():
    model = Sequential([
        Input(shape=(LATENT_DIM,)),
        Dense(7 * 7 * 128, use_bias=False),
        Reshape((7, 7, 128)),
        BatchNormalization(),
        ReLU(),
        
        Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        
        Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator = build_generator()
generator.summary()


# 4. Crear el discriminador
print("\n" + "="*70)
print(" Creando el discriminador")
print("="*70)

print("""
Arquitectura del Discriminador:
Input (28, 28, 1) → imagen
    ↓
Conv2D(64, 5x5) + LeakyReLU + Dropout      → 14x14
    ↓
Conv2D(128, 5x5) + LeakyReLU + Dropout     → 7x7
    ↓
Flatten
    ↓
Dense(1) + Sigmoid                         → probabilidad real/falso
""")

def build_discriminator():
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

discriminator = build_discriminator()
discriminator.summary()


# 5. Compilador Discriminador
discriminator.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# 6. Crear la GAN Completa 
print("\n" + "="*70)
print("Creando la GAN (Generador + Discriminador)")
print("="*70)

# Congelamos el discriminador para entrenar solo el generador en esta fase
discriminator.trainable = False

gan_input = Input(shape=(LATENT_DIM,))
generated_img = generator(gan_input)
gan_output = discriminator(generated_img)

gan = Model(gan_input, gan_output)

gan.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

gan.summary()


# 7. Funciones Auxiliares
def generate_noise(batch_size):
    """Genera ruido aleatorio para el generador"""
    return np.random.normal(0, 1, (batch_size, LATENT_DIM))

def save_generated_images(epoch, generator, examples=16):
    """Guarda imágenes generadas para visualizar la evolución"""
    noise = generate_noise(examples)
    generated_imgs = generator.predict(noise, verbose=0)
    
    # Escalar de [-1, 1] a [0, 1] para mostrar
    generated_imgs = (generated_imgs + 1) / 2.0
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(generated_imgs[i, :, :, 0], cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'Imágenes generadas en Época {epoch+1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# 8. Entrenar la GAN
print("\n" + "="*70)
print("Entrenando la GAN")
print("="*70)
print("️  Esto puede tardar varios minutos. Observa cómo mejora el generador.")

# Etiquetas reales y falsas
real_labels = np.ones((BATCH_SIZE, 1))
fake_labels = np.zeros((BATCH_SIZE, 1))

# Historial para gráficas
d_losses = []
g_losses = []
d_accuracies = []

start_time = time.time()

for epoch in range(EPOCHS):
    # ========================================
    # Fase 1: Entrenar Discriminador
    # ========================================
    # Seleccionar batch de imágenes reales
    idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
    real_imgs = x_train[idx]
    
    # Generar imágenes falsas
    noise = generate_noise(BATCH_SIZE)
    fake_imgs = generator.predict(noise, verbose=0)
    
    # Entrenar Discriminador con reales y falsos
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    
    d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
    d_acc = 0.5 * np.add(d_loss_real[1], d_loss_fake[1])
    
    # ========================================
    # FASE 2: Entrenar Generador
    # ========================================
    noise = generate_noise(BATCH_SIZE)
    # El generador quiere que el discriminador diga "real" (1) para sus imágenes falsas
    g_loss = gan.train_on_batch(noise, real_labels)
    
    # Guardar historial
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    d_accuracies.append(d_acc)
    
    # Imprimir progreso cada 5 épocas
    if (epoch + 1) % 5 == 0:
        elapsed = time.time() - start_time
        print(f"Época {epoch+1}/{EPOCHS} | "
              f"D Loss: {d_loss:.4f} | D Acc: {d_acc:.4f} | "
              f"G Loss: {g_loss:.4f} | Tiempo: {elapsed:.1f}s")
        
        # Mostrar imágenes generadas
        save_generated_images(epoch, generator)

total_time = time.time() - start_time
print(f"\n✅ Entrenamiento completado en {total_time:.2f} segundos")


# 9. Visualizar Evolución del entrenamiento
print("\n" + "="*70)
print("Visualizando La Evolución")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss del Discriminador
axes[0].plot(d_losses, color='blue', linewidth=2)
axes[0].set_title('🕵️  Discriminator Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

# Loss del Generador
axes[1].plot(g_losses, color='orange', linewidth=2)
axes[1].set_title('🎨 Generator Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].grid(True, alpha=0.3)

# Accuracy del Discriminador
axes[2].plot(d_accuracies, color='green', linewidth=2)
axes[2].axhline(y=0.5, color='red', linestyle='--', label='Aleatorio (50%)')
axes[2].set_title('🎯 Discriminator Accuracy', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Época')
axes[2].set_ylabel('Accuracy')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 10. Galería final de imágenes generadas
print("\n Generando galería final de 25 imágenes inventadas...")

noise = generate_noise(25)
final_imgs = generator.predict(noise, verbose=0)
final_imgs = (final_imgs + 1) / 2.0

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(final_imgs[i, :, :, 0], cmap='gray')
    ax.axis('off')

plt.suptitle(' 25 DÍGITOS INVENTADOS POR LA GAN\n(¡Ninguno existe en el dataset real!)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# 11. Lecciones Clave
print("\n" + "="*70)
print("Lecciones clave del día 20")
print("="*70)

print("""
1. GAN = Dos redes en competencia:
   - Generador: crea imágenes falsas desde ruido
   - Discriminador: distingue real de falso
   - Se entrenan alternadamente

2. Arquitectura DCGAN:
   - Generador usa Conv2DTranspose (deconvolución)
   - Discriminador usa Conv2D normal
   - BatchNormalization estabiliza el entrenamiento
   - LeakyReLU en Discriminador evita "neuronas muertas"

3. Señales de buen entrenamiento:
   - D Loss se estabiliza alrededor de 0.7-1.0
   - G Loss baja gradualmente
   - D Accuracy se acerca a 0.5 (no puede distinguir)

4. Señales de problemas:
   - D Loss = 0 muy rápido → Discriminador demasiado fuerte
   - G Loss explota → Inestabilidad
   - Imágenes todas iguales → Mode Collapse

5. Aplicaciones reales de GANs:
   - Generación de rostros (StyleGAN de NVIDIA)
   - Traducción de imagen a imagen (Pix2Pix)
   - Super-resolución (SRGAN)
   - Arte generativo
   - Aumento de datos médicos

6. VAEs (Variational Autoencoders):
   - Alternativa a GANs para generación
   - Más estables pero imágenes más borrosas
   - Útiles para interpolación en espacio latente

CONCLUSIÓN: Las GANs son la herramienta más poderosa
para generación de contenido visual con IA.
""")

print("\n✅ Día 20 completado!")