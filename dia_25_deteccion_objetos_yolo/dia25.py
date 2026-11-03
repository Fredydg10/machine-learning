
# 🚀 DÍA 25: DETECCIÓN DE OBJETOS CON YOLOv8
# El Gran Final del Roadmap

# 1. Instalar Ultralytics (la librería oficial de YOLO)
!pip install -q ultralytics matplotlib opencv-python

import cv2
import matplotlib.pyplot as plt
import urllib.request
import os

print("="*70)
print("🏆 DÍA 25: DETECCIÓN DE OBJETOS EN TIEMPO REAL (YOLOv8)")
print("="*70)


# 2. Cargar modelo pre-entrenado
print("\n Cargando modelo YOLOv8 Nano (rápido y preciso)...")
# 'yolov8n.pt' es la versión más ligera. También existe 'yolov8s.pt', 'yolov8m.pt', etc.
model = YOLO('yolov8n.pt')
print("✅ Modelo cargado exitosamente")


# 3. Descargar imágenes de prueba
print("\n Descargando imágenes de prueba...")

urls = [
    "https://images.unsplash.com/photo-1517423568366-8b83523034fd?w=800&q=80",  # Calle con gente y carros
    "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=800&q=80",  # Perros jugando
    "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800&q=80"   # Carros estacionados
]

image_paths = []
for i, url in enumerate(urls):
    path = f"test_image_{i}.jpg"
    urllib.request.urlretrieve(url, path)
    image_paths.append(path)

print(f"✅ {len(image_paths)} imágenes descargadas")


# 4. Ejecutar detección
print("\n Ejecutando detección de objetos...")

# Ejecutamos el modelo en las 3 imágenes
results = model(image_paths, conf=0.5)  # conf=0.5 filtra detecciones con menos del 50% de confianza


# 5. Visualizar resultados
print("\n Generando visualizaciones...")

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for idx, (result, ax) in enumerate(zip(results, axes)):
    # YOLO dibuja las cajas directamente en la imagen
    annotated_img = result.plot()

    # Convertir de BGR (OpenCV) a RGB (Matplotlib)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    ax.imshow(annotated_img_rgb)
    ax.set_title(f'Imagen {idx+1}', fontsize=14, fontweight='bold')
    ax.axis('off')

plt.suptitle(' Detección de Objetos con YOLOv8', fontsize=20, fontweight='bold', color='darkblue')
plt.tight_layout()
plt.show()


# 6. Análisis detallado de una imagen
print("\n" + "="*70)
print(" Análisis detallado de la primera imagen ")
print("="*70)

first_result = results[0]
boxes = first_result.boxes

print(f"Total de objetos detectados: {len(boxes)}\n")
print(f"{'Clase':<15} {'Confianza':<12} {'Coordenadas (x1, y1, x2, y2)'}")
print("-" * 60)

for box in boxes:
    # Obtener nombre de la clase
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    # Obtener confianza
    confidence = float(box.conf[0])

    # Obtener coordenadas
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    print(f"{class_name:<15} {confidence:.2%}      [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")


# 7. Filtrar por clase específica (Mini-Reto)
print("\n" + "="*70)
print("Filtrando solo: 'PERSONAS' Y 'CARROS'")
print("="*70)

# Clases de COCO: 0='person', 2='car'
results_filtered = model(image_paths[0], classes=[0, 2], conf=0.5)

plt.figure(figsize=(10, 8))
filtered_img = results_filtered[0].plot()
filtered_img_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
plt.imshow(filtered_img_rgb)
plt.title('Solo Personas y Carros detectados', fontsize=16, fontweight='bold')
plt.axis('off')
plt.show()


# 8. Lecciones Claves
print("\n" + "="*70)
print("Lecciones Claves")
print("="*70)

print("""
1. YOLO (You Only Look Once):
   - Es un detector de objetos de un solo paso (one-stage detector).
   - Extremadamente rápido, ideal para tiempo real (video, cámaras).

2. Componentes de una detección:
   - Bounding Box: Dónde está el objeto.
   - Clase: Qué es el objeto (de las 80 clases de COCO).
   - Confianza: Qué tan seguro está el modelo (0.0 a 1.0).

3. Ultralytics YOLOv8:
   - Es el estado del arte (SOTA) en detección de objetos.
   - Fácil de usar: 3 líneas de código para detectar cualquier cosa.
   - Se puede re-entrenar (fine-tuning) con tus propios datos.

4. IoU (Intersection over Union):
   - Métrica que evalúa qué tan bien se ajusta la caja predicha a la real.
   - NMS (Non-Maximum Suppression) elimina cajas duplicadas del mismo objeto.
""")

print("\n✅ ¡DÍA 25 COMPLETADO! ")
