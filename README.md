# Semantic Policy Matching (Embeddings Only)

Sistema de emparejamiento semántico entre descripciones de proyectos y un catálogo de políticas públicas utilizando exclusivamente embeddings (sin LLM).

---

## 📌 Descripción

Este proyecto implementa un pipeline eficiente y modular que:

1. Genera embeddings para políticas.
2. Genera embeddings para proyectos.
3. Calcula similitud coseno mediante Nearest Neighbors.
4. Devuelve las Top-K políticas más similares por proyecto.
5. Exporta los resultados en CSV.

No utiliza LLM ni re-ranking adicional.  
Es rápido, determinístico y estable.

---

## 🧠 Flujo del Modelo

Proyecto  
→ Embedding  
→ Recuperación Top-K  
→ Ranking por similitud  
→ Exportación  

Fórmula de similitud:

```
similarity_score = 1 - distancia_coseno
```

---

## 📁 Estructura del Proyecto

```
semantic-policy-matching-embeddings/
├── src/
│   └── matching/
│       ├── matcher.py
│       ├── io.py
│       ├── config.py
│       └── __init__.py
├── scripts/
│   └── run_matching.py
├── data/
│   ├── raw/
│   └── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📂 data/raw

Coloca aquí tus archivos de entrada:

```
data/raw/politicas.xlsx
data/raw/proyectos.xlsx
```

También puedes usar CSV:

```
data/raw/politicas.csv
data/raw/proyectos.csv
```

Los nombres deben coincidir con `config.py`.

---

## ⚙️ Instalación

### Crear entorno virtual

**Windows**
```
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux**
```
python -m venv .venv
source .venv/bin/activate
```

### Instalar dependencias
```
pip install -r requirements.txt
```

---

## ▶️ Ejecución

**Windows**
```
set PYTHONPATH=src
python scripts\run_matching.py
```

**Mac / Linux**
```
PYTHONPATH=src python scripts/run_matching.py
```

---

## 📊 Output

Archivo generado:

```
data/outputs/matching_top10_embeddings.csv
```

Columnas principales:

- matched_politica_text
- similarity_score
- rank
- columnas originales del proyecto

---

## 🔧 Parámetros Configurables

En `src/matching/config.py` puedes modificar:

- politicas_path
- proyectos_path
- col_text_politica
- col_text_proyecto
- col_id_proyecto
- top_k
- model_name
- batch_size
- device

Modelo por defecto:

```
BAAI/bge-m3
```

---

## 🛠 Stack Tecnológico

- Python
- SentenceTransformers
- PyTorch
- scikit-learn
- Pandas
- NumPy
