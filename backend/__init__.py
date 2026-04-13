"""
backend/
--------
Paquete del motor de inferencia Naïve Bayes para clasificación de tickets.

Módulos:
  - preprocessor  : Limpieza, tokenización, stopwords, stemming/lematización.
  - vocabulary    : Construcción del vocabulario (Bag of Words).
  - naive_bayes   : Clasificador Naïve Bayes Multinomial con Laplace Smoothing.
  - evaluator     : K-Folds CV, métricas (Accuracy, Precision, Recall, F1, Confusion Matrix).
  - model_io      : Guardado y carga del modelo entrenado (pickle/JSON).
  - data_loader   : Carga y preparación del dataset Bitext.
"""

from .preprocessor import TextPreprocessor
from .vocabulary   import Vocabulary
from .naive_bayes  import NaiveBayesClassifier
from .evaluator    import (
    MetricsCalculator,
    KFoldsCrossValidator,
    print_classification_report,
    print_confusion_matrix,
)
from .model_io     import ModelManager
from .data_loader  import DataLoader

__all__ = [
    "TextPreprocessor",
    "Vocabulary",
    "NaiveBayesClassifier",
    "MetricsCalculator",
    "KFoldsCrossValidator",
    "print_classification_report",
    "print_confusion_matrix",
    "ModelManager",
    "DataLoader",
]