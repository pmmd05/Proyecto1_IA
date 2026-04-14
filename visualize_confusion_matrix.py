#!/usr/bin/env python3
"""
Script para visualizar la matriz de confusión del modelo entrenado.

Genera:
  - Visualización en terminal (tabla ASCII)
  - Gráfico PNG (heatmap con matplotlib)
  - Estadísticas por clase
"""

import json
from pathlib import Path
from backend.model_io import ModelManager
from backend.data_loader import DataLoader
from backend.preprocessor import TextPreprocessor
from backend.evaluator import MetricsCalculator


def print_confusion_matrix_table(matrix: list[list[int]], classes: list[str]):
    """Imprime la matriz de confusión en formato tabla ASCII."""
    print("\n" + "=" * 100)
    print("MATRIZ DE CONFUSIÓN")
    print("=" * 100)
    print("Filas = Clases REALES | Columnas = Clases PREDICHAS\n")
    
    # Header con nombres de clases (abreviados)
    max_class_len = max(len(c) for c in classes)
    abbrev = [c[:7] for c in classes]  # Abreviar a 7 caracteres
    
    header = "".rjust(max_class_len + 2)
    for col_name in abbrev:
        header += f"{col_name:>8}"
    print(header)
    print("-" * len(header))
    
    # Filas
    for i, row_name in enumerate(classes):
        row_str = row_name.rjust(max_class_len + 2)
        for j in range(len(classes)):
            row_str += f"{matrix[i][j]:>8}"
        print(row_str)
    print("=" * 100 + "\n")


def plot_confusion_matrix_heatmap(
    matrix: list[list[int]], 
    classes: list[str],
    output_path: str = "models/confusion_matrix.png"
):
    """
    Crea una visualización heatmap de la matriz de confusión con matplotlib.
    
    Parámetros
    ----------
    matrix : list[list[int]]
        Matriz de confusión.
    classes : list[str]
        Nombres de las clases.
    output_path : str
        Ruta donde guardar la imagen PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib no está instalado. Instala con: pip install matplotlib")
        return
    
    matrix_array = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crear heatmap
    im = ax.imshow(matrix_array, cmap='Blues', aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    
    # Title y labels de ejes
    ax.set_xlabel("Clase Predicha", fontsize=12, fontweight='bold')
    ax.set_ylabel("Clase Real", fontsize=12, fontweight='bold')
    ax.set_title("Matriz de Confusión - Modelo Naïve Bayes", fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cantidad', rotation=270, labelpad=20)
    
    # Texto en las celdas
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = matrix_array[i, j]
            text_color = 'white' if value > matrix_array.max() / 2 else 'black'
            text = ax.text(j, i, value, ha="center", va="center", 
                          color=text_color, fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Guardar
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfico guardado en: {output_path}")
    
    # Mostrar (opcional, descomenta si usas en GUI)
    # plt.show()
    plt.close()


def print_per_class_metrics(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str]
):
    """Imprime precisión, recall y F1-score por clase."""
    print("\n" + "=" * 100)
    print("MÉTRICAS POR CLASE")
    print("=" * 100)
    
    metrics = MetricsCalculator.precision_recall_f1_per_class(y_true, y_pred, classes)
    
    print(f"{'Clase':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 65)
    
    total_support = 0
    for c in classes:
        m = metrics[c]
        support = m.get('support', 0)
        total_support += support
        print(f"{c:<15} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1']:>12.4f} {int(support):>10}")
    
    print("-" * 65)
    
    # Macro average
    macro_precision = sum(metrics[c]['precision'] for c in classes) / len(classes)
    macro_recall = sum(metrics[c]['recall'] for c in classes) / len(classes)
    macro_f1 = MetricsCalculator.macro_f1(metrics)
    
    print(f"{'macro avg':<15} {macro_precision:>12.4f} {macro_recall:>12.4f} {macro_f1:>12.4f} {int(total_support):>10}")
    print("=" * 100 + "\n")


def main():
    print("\n" + "=" * 100)
    print("VISUALIZACIÓN - MATRIZ DE CONFUSIÓN")
    print("=" * 100)
    
    # Cargar modelo
    try:
        manager = ModelManager()
        model, metadata = manager.load()
        print(f"✓ Modelo cargado")
        print(f"  Vocabulario: {model.vocabulary_.size} palabras")
        print(f"  Clases: {model.classes_}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Ejecuta 'python train.py' primero")
        return
    
    # Cargar dataset
    try:
        loader = DataLoader()
        X_texts, y = loader.load()
        print(f"✓ Dataset cargado: {len(X_texts)} instancias")
    except Exception as e:
        print(f"✗ Error al cargar dataset: {e}")
        return
    
    # Preprocesar
    preprocessor = TextPreprocessor(use_lemmatization=True, min_token_length=2)
    print("  Preprocesando...")
    X_tokens = preprocessor.preprocess_batch(X_texts)
    
    # Predecir
    print("  Prediciendo...")
    y_pred = model.predict_batch(X_tokens)
    
    # Calcular matriz
    matrix = MetricsCalculator.confusion_matrix(y, y_pred, model.classes_)
    
    # Mostrar resultados
    print_confusion_matrix_table(matrix, model.classes_)
    print_per_class_metrics(y, y_pred, model.classes_)
    
    # Generar gráfico
    try:
        plot_confusion_matrix_heatmap(matrix, model.classes_)
    except Exception as e:
        print(f"⚠️  No se pudo generar gráfico: {e}")
    
    # Accuracy global
    accuracy = MetricsCalculator.accuracy(y, y_pred)
    print(f"\nAccuracy Global: {accuracy:.4f} ({accuracy*100:.2f}%)\n")


if __name__ == "__main__":
    main()
