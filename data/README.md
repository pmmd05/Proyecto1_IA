# Carpeta de datos

Esta carpeta almacena el dataset utilizado para entrenar el modelo.

## Dataset: Bitext Customer Support LLM Chatbot Training Dataset

- **Fuente:** https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- **Instancias:** 26,872 solicitudes de clientes
- **Clases:** 11 categorías (ORDER, BILLING, SHIPPING, REFUND, ACCOUNT, etc.)

## Opciones de carga

### Opción 1 — Descarga automática (recomendada)
Al ejecutar `train.py`, el script descargará el dataset automáticamente
si no encuentra el archivo local `bitext_customer_support.csv`.

Requiere conexión a internet y tener instalado:
```
pip install datasets
```

### Opción 2 — Descarga manual
1. Visita: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
2. Descarga los archivos parquet y conviértelos a CSV con las columnas `instruction` y `category`.
3. Guarda el archivo como `bitext_customer_support.csv` en esta carpeta.

## Nota sobre el dataset
Los placeholders del tipo `{{Order Number}}`, `{{Name}}`, etc.
son eliminados automáticamente durante el preprocesamiento en `backend/preprocessor.py`.