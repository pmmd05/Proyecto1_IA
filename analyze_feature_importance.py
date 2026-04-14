#!/usr/bin/env python3
"""
Script para analizar la importancia de palabras (feature importance) en el modelo Naïve Bayes.

Muestra:
  1. Top N palabras más discriminativas por clase
  2. Cómo contribuye cada palabra a una predicción específica
  3. Visualización de pesos
"""

import math
from backend.model_io import ModelManager
from backend.preprocessor import TextPreprocessor


def get_top_words_per_class(model, top_n: int = 20) -> dict[str, list[tuple[str, float]]]:
    """
    Extrae las palabras más importantes para cada clase basadas en P(w|c).
    
    En Naïve Bayes, P(w|c) = (count(w,c) + α) / (Σ count(w,c) + α*|V|)
    
    Parámetros
    ----------
    model : NaiveBayesClassifier
        Modelo entrenado.
    top_n : int
        Número de palabras principales a retornar.
        
    Retorna
    -------
    dict[str, list[tuple[str, float]]]
        Para cada clase, lista de (palabra, probabilidad)
    """
    result = {}
    
    for class_label in model.classes_:
        # Calcular P(w|c) para cada palabra en el vocabulario
        word_probs = {}
        
        for word in model.vocabulary_.words:
            # Aplicar fórmula de Laplace Smoothing en escala logarítmica
            count = model.word_counts_[class_label].get(word, 0)
            total = model.class_word_totals_[class_label]
            vocab_sz = model.vocabulary_.size
            
            # log P(w|c) = log((count + α) / (total + α*|V|))
            log_prob = math.log((count + model.alpha) / (total + model.alpha * vocab_sz))
            word_probs[word] = log_prob
        
        # Ordenar y obtener top N
        sorted_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)
        result[class_label] = sorted_words[:top_n]
    
    return result


def print_feature_importance_per_class(model, top_n: int = 15):
    """Imprime las palabras más importantes para cada clase."""
    print("\n" + "=" * 100)
    print("PALABRAS MÁS IMPORTANTES POR CLASE (Feature Importance)")
    print("=" * 100)
    print("Ordenadas por P(w|c) - Probabilidad de la palabra dada la clase\n")
    
    top_words = get_top_words_per_class(model, top_n=top_n)
    
    for class_label in model.classes_:
        print(f"\n{'─' * 80}")
        print(f"  CLASE: {class_label}")
        print(f"{'─' * 80}")
        print(f"  {'Rango':<6} {'Palabra':<20} {'log P(w|c)':<15}")
        print(f"  {'-'*50}")
        
        for rank, (word, log_prob) in enumerate(top_words[class_label], 1):
            print(f"  {rank:<6} {word:<20} {log_prob:>15.6f}")
    
    print("\n" + "=" * 100 + "\n")


def analyze_prediction(model, text: str, preprocessor: TextPreprocessor):
    """
    Analiza cómo contribuye cada palabra a la predicción final.
    
    Muestra:
      - Tokens extraídos del texto
      - Score logarítmico para cada clase
      - Contribución de cada palabra al score
      - Clase predicha
    """
    print("\n" + "=" * 100)
    print(f"ANÁLISIS DE PREDICCIÓN: \"{text}\"")
    print("=" * 100 + "\n")
    
    # Preprocesar
    tokens = preprocessor.preprocess(text)
    print(f"Tokens extraídos: {tokens}\n")
    
    # Predicción
    pred_class = model.predict(tokens)
    proba = model.predict_proba(tokens)
    
    print(f"Predicción: {pred_class}")
    print(f"Confianza: {proba[pred_class]:.6f}\n")
    
    # Calcular scores
    print("Scores por clase (log escala):")
    print(f"{'Clase':<15} {'Score':<15} {'Probabilidad':<15} {'Contribución'}")
    print("-" * 70)
    
    # Solo palabras en vocabulario
    vocab_tokens = [t for t in tokens if model.vocabulary_.contains(t)]
    
    for class_label in sorted(model.classes_):
        # Score = log P(c) + Σ log P(w|c)
        class_prior = model.class_log_priors_[class_label]
        likelihood_sum = sum(model._log_likelihood(t, class_label) for t in vocab_tokens)
        score = class_prior + likelihood_sum
        
        indicator = " ← PREDICCIÓN" if class_label == pred_class else ""
        print(f"{class_label:<15} {score:>15.6f} {proba[class_label]:>15.6f}{indicator}")
    
    print("\n" + "-" * 70)
    print("Contribución de palabras al score de la clase predicha:\n")
    print(f"  Prior log P({pred_class}): {model.class_log_priors_[pred_class]:.6f}")
    
    total_likelihood = 0
    for word in vocab_tokens:
        if model.vocabulary_.contains(word):
            ll = model._log_likelihood(word, pred_class)
            total_likelihood += ll
            # Visualizar con barras
            bar_len = max(1, int((ll + 5) * 5))  # Escalar para visualización
            bar = "█" * max(0, bar_len // 2) if ll > 0 else "▌" * max(1, -bar_len // 3)
            print(f"  log P({word}|{pred_class}): {ll:>10.6f}  {bar}")
    
    print(f"  Total likelihood: {total_likelihood:.6f}")
    print(f"  Score final: {model.class_log_priors_[pred_class] + total_likelihood:.6f}")
    
    print("\n" + "=" * 100 + "\n")


def interactive_mode():
    """Modo interactivo para analizar predicciones."""
    print("\n" + "=" * 100)
    print("ANÁLISIS INTERACTIVO DE PREDICCIONES")
    print("=" * 100)
    
    # Cargar modelo
    try:
        manager = ModelManager()
        model, _ = manager.load()
        print(f"\n✓ Modelo cargado: {len(model.classes_)} clases, {model.vocabulary_.size} palabras")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    preprocessor = TextPreprocessor(use_lemmatization=True, min_token_length=2)
    
    while True:
        print("\nIngresa un texto (o 'salir' para terminar):")
        text = input("> ").strip()
        
        if text.lower() == 'salir':
            break
        
        if text:
            try:
                analyze_prediction(model, text, preprocessor)
            except Exception as e:
                print(f"Error al procesar: {e}")


def main():
    import sys
    
    # Cargar modelo
    try:
        manager = ModelManager()
        model, _ = manager.load()
        print(f"\n✓ Modelo cargado: {len(model.classes_)} clases, {model.vocabulary_.size} palabras\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Ejecuta 'python train.py' primero")
        return
    
    preprocessor = TextPreprocessor(use_lemmatization=True, min_token_length=2)
    
    # Modo por línea de comandos o interactivo
    if len(sys.argv) > 1:
        # Analizar texto pasado como argumento
        text = " ".join(sys.argv[1:])
        analyze_prediction(model, text, preprocessor)
    else:
        # Mostrar feature importance
        print_feature_importance_per_class(model, top_n=15)
        
        # Ejemplos
        print("\nEJEMPLOS DE ANÁLISIS:\n")
        examples = [
            "I want to cancel my order",
            "I was charged twice for my last purchase, please refund one payment.",
            "How do I contact support",
            "I haven't received my delivery yet",
        ]
        
        for i, text in enumerate(examples, 1):
            print(f"\n[Ejemplo {i}/{len(examples)}]")
            analyze_prediction(model, text, preprocessor)
        
        # Ofrecer modo interactivo
        print("\n¿Deseas analizar más textos? Ingresa 'si' para modo interactivo:")
        if input("> ").strip().lower() in ['si', 'yes', 'y', 's']:
            interactive_mode()


if __name__ == "__main__":
    main()
