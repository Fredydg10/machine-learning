import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# 1. Importar librerías de NLP
print("Importando librerías de NLP...")

import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Diccionario de POS tags en español
POS_DESCRIPTIONS = {
    'ADJ': 'Adjetivo',
    'ADP': 'Adposición',
    'ADV': 'Adverbio',
    'AUX': 'Verbo auxiliar',
    'CCONJ': 'Conjunción coordinante',
    'DET': 'Determinante',
    'INTJ': 'Interjección',
    'NOUN': 'Sustantivo',
    'NUM': 'Número',
    'PART': 'Partícula',
    'PRON': 'Pronombre',
    'PROPN': 'Nombre propio',
    'PUNCT': 'Puntuación',
    'SCONJ': 'Conjunción subordinante',
    'SYM': 'Símbolo',
    'VERB': 'Verbo',
    'X': 'Otro'
}

# Diccionario de tipos de entidades NER en español
NER_DESCRIPTIONS = {
    'PER': 'Persona',
    'LOC': 'Lugar',
    'ORG': 'Organización',
    'MISC': 'Misceláneo',
    'NORP': 'Grupo nacional/religioso/político',
    'FAC': 'Instalación/Edificio',
    'GPE': 'País/Ciudad/Estado',
    'PRODUCT': 'Producto',
    'EVENT': 'Evento',
    'WORK_OF_ART': 'Obra de arte',
    'LAW': 'Ley',
    'LANGUAGE': 'Idioma',
    'DATE': 'Fecha',
    'TIME': 'Hora',
    'PERCENT': 'Porcentaje',
    'MONEY': 'Dinero',
    'QUANTITY': 'Cantidad',
    'ORDINAL': 'Ordinal',
    'CARDINAL': 'Número cardinal'
}

# Cargar modelo de spaCy en español
print("🤖 Cargando modelo de spaCy en español...")
nlp_es = spacy.load('es_core_news_sm')
print("✅ Modelo cargado correctamente")


# 2. Datos de ejemplo (Tweets/Noticias en español)
print("\n" + "="*70)
print("Datos de ejemplo")
print("="*70)

textos = [
    "El presidente de España anunció nuevas medidas económicas en Madrid.",
    "¡Increíble! El Real Madrid ganó el campeonato con un gol en el último minuto.",
    "La inteligencia artificial está transformando la industria tecnológica en 2026.",
    "Los científicos descubrieron un nuevo planeta habitable cerca de Marte.",
    "Apple presentó su nuevo iPhone 18 con funciones de IA revolucionarias.",
    "El cambio climático afecta a millones de personas en América Latina.",
    "Messi marcó tres goles en el partido contra Brasil ayer por la noche.",
    "Google, Microsoft y Amazon compiten por dominar el mercado de la nube."
]

print(f"\n Tenemos {len(textos)} textos en español:")
for i, texto in enumerate(textos, 1):
    print(f"  {i}. {texto}")


# 3. Tokenización
print("\n" + "="*70)
print(" Tokenización (Dividir texto en unidades)")
print("="*70)

# Tokenización con spaCy
print("\n🔵 Tokenización con spaCy:")
doc = nlp_es(textos[0])
tokens_spacy = [token.text for token in doc]
print(f"   Texto: '{textos[0]}'")
print(f"   Tokens: {tokens_spacy}")

# Tokenización con NLTK
print("\n🟢 Tokenización con NLTK:")
tokens_nltk = word_tokenize(textos[0], language='spanish')
print(f"   Tokens: {tokens_nltk}")

# Tokenización de oraciones
print("\n📄 Tokenización de oraciones:")
texto_largo = "El presidente visitó Madrid. Luego viajó a Barcelona. Finalmente regresó a casa."
oraciones = sent_tokenize(texto_largo, language='spanish')
print(f"   Oraciones: {oraciones}")


# 4. STEMMING vs LEMATIZACIÓN
print("\n" + "="*70)
print("STEMMING vs LEMATIZACIÓN")
print("="*70)

# Stemmer en español
stemmer = SnowballStemmer('spanish')

palabras_ejemplo = [
    "corriendo", "corrió", "correrán", "corrida",
    "gatos", "gata", "gatitos",
    "programando", "programador", "programación",
    "feliz", "felices", "felicidad"
]

print(f"\n{'Palabra':<15} {'Stemming':<15} {'Lematización (spaCy)':<20}")
print("-" * 50)

resultados_stem_lem = []

for palabra in palabras_ejemplo:
    # Stemming con NLTK
    stem = stemmer.stem(palabra)
    
    # Lematización con spaCy
    doc_palabra = nlp_es(palabra)
    lemma = doc_palabra[0].lemma_
    
    resultados_stem_lem.append({
        'palabra': palabra,
        'stem': stem,
        'lemma': lemma
    })
    
    print(f"{palabra:<15} {stem:<15} {lemma:<20}")

# Visualizar comparación
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(palabras_ejemplo))
width = 0.35

ax.barh(x - width/2, [len(r['stem']) for r in resultados_stem_lem], 
        width, label='Stemming (longitud)', color='#ff6b6b')
ax.barh(x + width/2, [len(r['lemma']) for r in resultados_stem_lem], 
        width, label='Lematización (longitud)', color='#4ecdc4')

ax.set_yticks(x)
ax.set_yticklabels(palabras_ejemplo)
ax.set_xlabel('Longitud del resultado')
ax.set_title('Stemming vs Lematización: Longitud de palabras procesadas', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n Observación:")
print("   - Stemming: Corta la palabra (a veces produce no-palabras como 'corr')")
print("   - Lematización: Devuelve una palabra real del diccionario")


# 5. Pos Tagging(Etiquetado gramatical)
print("\n" + "="*70)
print("Pos Tagging (Etiquetado gramatical)")
print("="*70)

texto_pos = "El presidente de España anunció nuevas medidas económicas en Madrid."
doc_pos = nlp_es(texto_pos)

print(f"\n Texto: '{texto_pos}'\n")
print(f"{'Palabra':<15} {'POS':<10} {'Descripción':<30}")
print("-" * 55)

pos_data = []
for token in doc_pos:
    if not token.is_punct and not token.is_space:
        pos_data.append({
            'palabra': token.text,
            'pos': token.pos_,
            'descripcion': POS_DESCRIPTIONS.get(token.pos_, token.pos_)
        })
        print(f"{token.text:<15} {token.pos_:<10} {pos_data[-1]['descripcion']:<30}")

# Visualizar POS tagging
fig, ax = plt.subplots(figsize=(12, 6))
pos_counts = Counter([p['pos'] for p in pos_data])

colors = plt.cm.Set3(np.linspace(0, 1, len(pos_counts)))

# CORRECCIÓN: Envolver en list() para satisfacer al type checker
bars = ax.bar(list(pos_counts.keys()), list(pos_counts.values()), color=colors, edgecolor='black')

ax.set_xlabel('Etiqueta POS', fontsize=12)
ax.set_ylabel('Cantidad', fontsize=12)
ax.set_title('🏷️  Distribución de Etiquetas POS en el texto', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, list(pos_counts.values())):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(count), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()


# 6. NER (Named Entity Recognition)
print("\n" + "="*70)
print("NER (Reconocimiento de Entidades Nombradas)")
print("="*70)

texto_ner = "El presidente Pedro Sánchez visitó Madrid y Barcelona en enero de 2026. Luego se reunió con Emmanuel Macron en París."
doc_ner = nlp_es(texto_ner)

print(f"\n Texto: '{texto_ner}'\n")
print(f"{'Entidad':<20} {'Tipo':<15} {'Descripción'}")
print("-" * 60)

ner_data = []
for ent in doc_ner.ents:
    ner_data.append({
        'entidad': ent.text,
        'tipo': ent.label_,
        'descripcion': NER_DESCRIPTIONS.get(ent.label_, ent.label_)
    })
    print(f"{ent.text:<20} {ent.label_:<15} {ner_data[-1]['descripcion']}")

# Visualizar NER
if ner_data:
    fig, ax = plt.subplots(figsize=(12, 6))
    tipo_counts = Counter([n['tipo'] for n in ner_data])
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(tipo_counts)))
    
    # CORRECCIÓN: Envolver en list() para satisfacer al type checker
    bars = ax.bar(list(tipo_counts.keys()), list(tipo_counts.values()), 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Tipo de Entidad', fontsize=12)
    ax.set_ylabel('Cantidad', fontsize=12)
    ax.set_title(' Entidades Nombradas detectadas en el texto', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, list(tipo_counts.values())):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# 7. Stopwords (Palabras vacías)
print("\n" + "="*70)
print("Stopwords (Palabras vacías)")
print("="*70)

stop_words_es = set(stopwords.words('spanish'))
print(f"\n Total de stopwords en español: {len(stop_words_es)}")
print(f"\n Primeras 20 stopwords:")
print(f"   {sorted(list(stop_words_es))[:20]}")

# Filtrar stopwords de un texto
texto_ejemplo = "El presidente de España anunció nuevas medidas económicas en Madrid."
doc_ejemplo = nlp_es(texto_ejemplo)

tokens_sin_stopwords = [token.text for token in doc_ejemplo 
                        if not token.is_stop and not token.is_punct]

print(f"\n Texto original: '{texto_ejemplo}'")
print(f" Sin stopwords: {tokens_sin_stopwords}")


# 8. Pipeline Completo de NLP 

print("\n" + "="*70)
print("Pipeline Completo de NLP")
print("="*70)

def procesar_texto(texto, nlp_model):
    """Pipeline completo de procesamiento de texto"""
    
    doc = nlp_model(texto)
    
    # Tokenización
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    # Lematización
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    
    # POS Tagging
    pos_tags = [(token.text, token.pos_) for token in doc 
                if not token.is_punct and not token.is_space]
    
    # Sin stopwords
    tokens_filtrados = [token.lemma_ for token in doc 
                        if not token.is_stop and not token.is_punct and not token.is_space]
    
    # Entidades nombradas
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    
    return {
        'tokens': tokens,
        'lemmas': lemmas,
        'pos_tags': pos_tags,
        'tokens_filtrados': tokens_filtrados,
        'entidades': entidades
    }

# Procesar todos los textos
print("\n Procesando todos los textos...\n")

for i, texto in enumerate(textos, 1):
    resultado = procesar_texto(texto, nlp_es)
    
    print(f"   Texto {i}: '{texto}'")
    print(f"   Tokens: {resultado['tokens']}")
    print(f"   Lemas: {resultado['lemmas']}")
    print(f"   Sin stopwords: {resultado['tokens_filtrados']}")
    if resultado['entidades']:
        print(f"   Entidades: {resultado['entidades']}")
    print()

# 
# 9. Análisis de Frecuencia de Palabras
print("\n" + "="*70)
print("Análisis de Frecuencia de Palabras")
print("="*70)

# Procesar todos los textos y contar lemas
todos_lemmas = []
for texto in textos:
    doc = nlp_es(texto)
    lemmas = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and token.is_alpha]
    todos_lemmas.extend(lemmas)

# Contar frecuencia
frecuencia = Counter(todos_lemmas)
top_15 = frecuencia.most_common(15)

print(f"\n🔝 15 palabras más frecuentes (lematizadas):")
for palabra, count in top_15:
    print(f"   {palabra:<20} {count} veces")

# Visualizar frecuencia
fig, ax = plt.subplots(figsize=(12, 6))

# CORRECCIÓN: Extraer listas explícitas para evitar warnings de typing
palabras = [item[0] for item in top_15]
counts = [item[1] for item in top_15]

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(palabras)))
bars = ax.barh(range(len(palabras)), counts, color=colors, edgecolor='black')

ax.set_yticks(range(len(palabras)))
ax.set_yticklabels(palabras)
ax.set_xlabel('Frecuencia', fontsize=12)
ax.set_title(' Top 15 Palabras más frecuentes (lematizadas)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(count), ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()


# 10. Lecciones Clave
print("\n" + "="*70)
print("Lecciones Clave del Día 22")
print("="*70)

print("""
1. Tokenización:
   - spaCy y NLTK tokenizan de forma ligeramente diferente
   - spaCy es más inteligente con signos de puntuación
   - NLTK es más flexible para personalización

2. Stemming vs Lematización:
   - Stemming: Rápido pero brutal (corta palabras)
   - Lematización: Más lento pero devuelve palabras reales
   - Para NLP serio, usa siempre lematización

3. POS Tagging:
   - Etiqueta cada palabra con su rol gramatical
   - Útil para extraer sustantivos, verbos, adjetivos
   - Base para análisis más complejos

4. NER (Named Entity Recognition):
   - Detecta nombres propios, lugares, fechas, organizaciones
   - Fundamental para extracción de información
   - spaCy tiene modelos pre-entrenados para muchos idiomas

5. Stopwords:
   - Palabras comunes que no aportan significado ("el", "la", "de")
   - Se filtran para reducir ruido en el análisis
   - Cada idioma tiene su propia lista

6. Pipeline de NLP:
   - Tokenización → Lematización → POS Tagging → NER
   - Cada paso refina el texto para el siguiente
   - spaCy hace todo esto en un solo paso con nlp(texto)

CONCLUSIÓN: El NLP básico es la base para todo lo que viene.
Sin tokenización y lematización correctas, los modelos de IA
no pueden entender el texto correctamente.
""")

print("\n✅ Día 22 completado!")