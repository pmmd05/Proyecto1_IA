# Clasificación de Solicitudes a Mesa de Ayuda
### Proyecto No.1 — Universidad Rafael Landívar - Primer Semestre 2026 IA

Sistema de clasificación automática de tickets de soporte al cliente usando
el algoritmo **Naïve Bayes Multinomial** implementado completamente desde cero.

---

## Estructura del Proyecto

```
proyecto_ia_naivebayes/
│
├── backend/                    # Motor de inferencia (Python)
│   ├── __init__.py             # Exports del paquete
│   ├── preprocessor.py         # Limpieza, tokenización, stopwords, lematización
│   ├── vocabulary.py           # Vocabulario (Bag of Words)
│   ├── naive_bayes.py          # Clasificador Naïve Bayes Multinomial
│   ├── evaluator.py            # K-Folds CV, métricas, matriz de confusión
│   ├── model_io.py             # Guardado y carga del modelo (pickle/JSON)
│   └── data_loader.py          # Carga del dataset Bitext
│
├── frontend/                   # Interfaz web (HTML + CSS + JS + Flask)
│   └── ...
│
├── models/                     # Modelos entrenados guardados
│   ├── naive_bayes_model.pkl   # Formato binario (rápido)
│   └── naive_bayes_model.json  # Formato legible
│
├── data/                       # Dataset
│   ├── README.md               # Instrucciones de descarga
│   └── bitext_customer_support.csv  (generado al entrenar)
│
├── tests/                      # Pruebas unitarias
│
├── train.py                    # Script principal de entrenamiento
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este archivo
```

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd proyecto_ia_naivebayes
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## Uso

### Entrenar el modelo
```bash
python train.py
```

El script:
1. Descarga el dataset de HuggingFace (o carga el CSV local si existe).
2. Preprocesa todos los textos (limpieza, tokenización, lematización).
3. Ejecuta K-Folds Cross Validation con K=5.
4. Entrena el modelo final sobre el 80% de los datos.
5. Evalúa sobre el 20% de prueba e imprime métricas completas.
6. Guarda el modelo en `models/naive_bayes_model.pkl` y `.json`.

### Ejecutar la interfaz web
```bash
python app.py
```

---

## Técnicas Implementadas

| Técnica               | Módulo              | Descripción                                                   |
|-----------------------|---------------------|---------------------------------------------------------------|
| Bag of Words          | `vocabulary.py`     | Vocabulario indexado con frecuencias globales                 |
| Laplace Smoothing     | `naive_bayes.py`    | P(w\|c) = (count+α) / (total+α×\|V\|), evita P=0            |
| Log-Sum Inference     | `naive_bayes.py`    | score(c) = log P(c) + Σ log P(w\|c), evita underflow         |
| K-Folds CV (K=5)      | `evaluator.py`      | Implementación manual, con análisis de varianza entre folds   |
| Matriz de Confusión   | `evaluator.py`      | Construcción e interpretación para 11 clases                  |
| Métricas por clase    | `evaluator.py`      | Precisión, Recall, F1, Accuracy global, Macro F1             |
| Guardado del modelo   | `model_io.py`       | Pickle (.pkl) y JSON (.json) con metadatos                    |

---

## Dataset

**Bitext Customer Support LLM Chatbot Training Dataset**
- URL: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- 26,872 instancias · 11 clases
- Columnas usadas: `instruction` (texto) y `category` (etiqueta)

---

## Dependencias Principales

- **nltk** — Tokenización y stopwords (solo preprocesamiento)
- **flask** — Servidor web para la interfaz
- **flask-cors** — CORS para comunicación frontend-backend

> ⚠️ **No se usa** scikit-learn, TensorFlow, Keras, PyTorch ni ninguna
> librería que resuelva Naïve Bayes automáticamente.

---

## Autores
- Carlos Javier Santizo Mérida - 1080423
- Juan Pablo Orozco Ramos - 1093223
- Paula María Marroquín Diéguez - 1081123