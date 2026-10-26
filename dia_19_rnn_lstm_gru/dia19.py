import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

print(f"TensorFlow versión: {tf.__version__}")


# 1. Crear dataset de serie de tiempo
print("\n Creando dataset de serie de tiempo sintética...")

# Función para generar datos con tendencia + estacionalidad + ruido
def generate_time_series(n_steps=1000, trend=0.01, seasonality=10):
    np.random.seed(42)
    t = np.arange(n_steps)
    
    # Tendencia lineal
    trend_component = trend * t
    
    # Estacionalidad (onda sinusoidal)
    seasonal_component = 5 * np.sin(2 * np.pi * t / seasonality)
    
    # Ruido aleatorio
    noise = np.random.randn(n_steps) * 0.5
    
    # Serie final
    series = trend_component + seasonal_component + noise
    
    return series

# Generar datos
series = generate_time_series(n_steps=1000)

print(f"Total de puntos: {len(series)}")
print(f"Rango: [{series.min():.2f}, {series.max():.2f}]")

# Visualizar la serie
plt.figure(figsize=(14, 5))
plt.plot(series, linewidth=1)
plt.title('Serie de Tiempo Sintética (Tendencia + Estacionalidad + Ruido)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Tiempo (pasos)')
plt.ylabel('Valor')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 2. Preparar datos para RNN
print("\n" + "="*70)
print(" Preparando datos secuenciales ")
print("="*70)

# Función para crear secuencias (ventana deslizante)
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Parámetros
SEQ_LENGTH = 20  # Usamos 20 pasos para predecir el siguiente
TRAIN_SPLIT = 0.8

# Crear secuencias
X, y = create_sequences(series, SEQ_LENGTH)

# Dividir en train/test
split_idx = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# RNN necesita shape: (samples, timesteps, features)
# Nuestra serie es 1D, así que agregamos dimensión de features
X_train = X_train.reshape(-1, SEQ_LENGTH, 1)
X_test = X_test.reshape(-1, SEQ_LENGTH, 1)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# 3. Crear Modelos: RNN, LSTM, GRU

print("\n" + "="*70)
print("Creando Modelos")
print("="*70)

def create_rnn_model(model_type='simple'):
    """Crea un modelo RNN, LSTM o GRU"""
    
    model = Sequential()
    
    if model_type == 'simple':
        model.add(SimpleRNN(50, activation='relu', input_shape=(SEQ_LENGTH, 1)))
    elif model_type == 'lstm':
        model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)))
    elif model_type == 'gru':
        model.add(GRU(50, activation='relu', input_shape=(SEQ_LENGTH, 1)))
    
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Salida: un valor (predicción)
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Crear los 3 modelos
models = {
    'RNN Simple': create_rnn_model('simple'),
    'LSTM': create_rnn_model('lstm'),
    'GRU': create_rnn_model('gru')
}

for name, model in models.items():
    print(f"\n📌 {name}:")
    model.summary()


# 4. Entrenar Modelos
print("\n" + "="*70)
print("Entrenando Modelos")
print("="*70)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

results = {}
histories = {}

for name, model in models.items():
    print(f"\n Entrenando {name}...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )
    
    train_time = time.time() - start_time
    
    # Evaluar
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    results[name] = {
        'test_loss': test_loss,
        'train_time': train_time,
        'history': history
    }
    
    print(f"   ✅ Test Loss (MSE): {test_loss:.4f}")
    print(f"   ️  Tiempo: {train_time:.2f}s")


# 5. Comparar Resultados
print("\n" + "="*70)
print(" Comparación De Modelos")
print("="*70)

df_results = pd.DataFrame([{
    'Modelo': name,
    'Test Loss (MSE)': res['test_loss'],
    'Tiempo (s)': res['train_time']
} for name, res in results.items()])

print(df_results.to_string(index=False))

# Gráfico comparativo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss comparado
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
bars = axes[0].bar(df_results['Modelo'], df_results['Test Loss (MSE)'], 
                   color=colors, edgecolor='black')
axes[0].set_ylabel('Test Loss (MSE)', fontsize=12)
axes[0].set_title('📉 Error de Predicción (menor = mejor)', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

for bar, loss in zip(bars, df_results['Test Loss (MSE)']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{loss:.4f}', ha='center', fontweight='bold')

# Tiempo comparado
bars2 = axes[1].bar(df_results['Modelo'], df_results['Tiempo (s)'], 
                    color=colors, edgecolor='black')
axes[1].set_ylabel('Tiempo de Entrenamiento (s)', fontsize=12)
axes[1].set_title('⏱️  Velocidad de Entrenamiento', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for bar, time_val in zip(bars2, df_results['Tiempo (s)']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{time_val:.1f}s', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()


# 6. Visualizar curvas de aprendizaje
fig, ax = plt.subplots(figsize=(12, 6))

for (name, res), color in zip(results.items(), colors):
    ax.plot(res['history'].history['loss'], label=f'{name} - Train', 
            color=color, linewidth=2)
    ax.plot(res['history'].history['val_loss'], label=f'{name} - Val', 
            color=color, linewidth=2, linestyle='--')

ax.set_title('Curvas de Aprendizaje: Loss por Época', fontsize=14, fontweight='bold')
ax.set_xlabel('Época')
ax.set_ylabel('Loss (MSE)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 7. Visualizar Predicciónes
print("\n Visualizando Predicciónes Del Mejor Modelo...")

# Encontrar el mejor modelo (menor loss)
best_model_name = min(results, key=lambda x: results[x]['test_loss'])
best_model = models[best_model_name]

print(f"Mejor modelo: {best_model_name}")

# Hacer predicciones
y_pred = best_model.predict(X_test)

# Visualizar últimas 100 predicciones
plt.figure(figsize=(14, 6))
plt.plot(y_test[-100:], label='Real', linewidth=2, color='blue')
plt.plot(y_pred[-100:], label='Predicción', linewidth=2, color='orange')
plt.title(f'🎯 Predicciones del Mejor Modelo ({best_model_name})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Pasos de tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 8. Lecciónes Clave
print("\n" + "="*70)
print("Lecciones Clave del dia 19")
print("="*70)

print("""
1. RNN Simple:
   - Tiene memoria (estado oculto)
   - Sufre de vanishing/exploding gradients
   - No puede recordar información de pasos lejanos
   - Casi nunca se usa en la práctica

2. LSTM (Long Short-Term Memory):
   - Tiene 3 puertas (Forget, Input, Output)
   - Puede recordar información por muchos pasos
   - Es el estándar para secuencias largas
   - Más lento que GRU pero más potente

3. GRU (Gated Recurrent Unit):
   - Versión simplificada de LSTM (2 puertas)
   - Más rápida de entrenar
   - Casi tan buena como LSTM en la mayoría de casos
   - Ideal para tiempo real o secuencias cortas

4. Preparación de datos secuenciales:
   - Se usa "ventana deslizante" (sliding window)
   - Shape: (samples, timesteps, features)
   - SEQ_LENGTH es un hiperparámetro clave

5. Aplicaciones reales:
   - Predicción de precios de acciones
   - Procesamiento de lenguaje natural (NLP)
   - Reconocimiento de voz
   - Predicción de clima
   - Análisis de sentimiento en texto

CONCLUSIÓN: Para datos secuenciales, usa LSTM o GRU.
LSTM para precisión máxima, GRU para velocidad.
""")

print("\n✅ Día 19 completado!")