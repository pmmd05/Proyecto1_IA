"""
evaluator.py
------------
Módulo de evaluación del modelo Naïve Bayes.

Implementa desde cero (sin scikit-learn):
  ✔ K-Folds Cross Validation (K configurable, mínimo 5 según el proyecto).
  ✔ Matriz de Confusión.
  ✔ Precisión, Recall y F1-Score por clase.
  ✔ Accuracy global.
  ✔ Macro F1.
  ✔ Análisis de varianza entre folds.
"""

import math
import random
from collections import defaultdict
from typing import Callable


class MetricsCalculator:
    """
    Calculadora de métricas de evaluación para clasificación multiclase.

    Todos los métodos son estáticos y no requieren instanciación.
    Trabajan únicamente con listas de Python estándar.
    """

    @staticmethod
    def accuracy(y_true: list[str], y_pred: list[str]) -> float:
        """
        Calcula la exactitud global (Accuracy).

        Fórmula:
            Accuracy = (TP + TN) / N  =  |predicciones correctas| / |total|

        Parámetros
        ----------
        y_true : list[str]   Etiquetas reales.
        y_pred : list[str]   Etiquetas predichas.

        Retorna
        -------
        float
            Valor entre 0 y 1.
        """
        if len(y_true) != len(y_pred) or len(y_true) == 0:
            raise ValueError("y_true e y_pred deben tener la misma longitud no vacía.")
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    @staticmethod
    def confusion_matrix(
        y_true: list[str],
        y_pred: list[str],
        classes: list[str],
    ) -> list[list[int]]:
        """
        Construye la matriz de confusión para clasificación multiclase.

        La celda [i][j] contiene el número de instancias de la clase real i
        que fueron predichas como clase j.

        Fila    → clase REAL
        Columna → clase PREDICHA

        Parámetros
        ----------
        y_true  : list[str]   Etiquetas reales.
        y_pred  : list[str]   Etiquetas predichas.
        classes : list[str]   Lista ordenada de clases.

        Retorna
        -------
        list[list[int]]
            Matriz de tamaño len(classes) × len(classes).
        """
        n = len(classes)
        class_idx = {c: i for i, c in enumerate(classes)}

        # Inicializar matriz con ceros
        matrix = [[0] * n for _ in range(n)]

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in class_idx and pred_label in class_idx:
                matrix[class_idx[true_label]][class_idx[pred_label]] += 1

        return matrix

    @staticmethod
    def precision_recall_f1_per_class(
        y_true: list[str],
        y_pred: list[str],
        classes: list[str],
    ) -> dict[str, dict[str, float]]:
        """
        Calcula Precisión, Recall y F1-Score para cada clase.

        Para cada clase c:
          - TP (True Positive) : predicho c, real c
          - FP (False Positive): predicho c, real ≠ c
          - FN (False Negative): predicho ≠ c, real c

        Fórmulas:
            Precisión(c) = TP / (TP + FP)
            Recall(c)    = TP / (TP + FN)
            F1(c)        = 2 × Precisión × Recall / (Precisión + Recall)

        Parámetros
        ----------
        y_true  : list[str]   Etiquetas reales.
        y_pred  : list[str]   Etiquetas predichas.
        classes : list[str]   Lista de clases a evaluar.

        Retorna
        -------
        dict[str, dict[str, float]]
            Para cada clase: {'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'support'}.
        """
        # Contadores por clase
        tp_count = defaultdict(int)
        fp_count = defaultdict(int)
        fn_count = defaultdict(int)
        support  = defaultdict(int)

        for true_label, pred_label in zip(y_true, y_pred):
            support[true_label] += 1
            if true_label == pred_label:
                tp_count[true_label] += 1
            else:
                fp_count[pred_label] += 1
                fn_count[true_label]  += 1

        results = {}
        for c in classes:
            tp = tp_count[c]
            fp = fp_count[c]
            fn = fn_count[c]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results[c] = {
                "precision": round(precision, 4),
                "recall":    round(recall,    4),
                "f1":        round(f1,        4),
                "tp":        tp,
                "fp":        fp,
                "fn":        fn,
                "support":   support[c],
            }

        return results

    @staticmethod
    def macro_f1(per_class_metrics: dict[str, dict[str, float]]) -> float:
        """
        Calcula el Macro F1-Score (promedio no ponderado de F1 por clase).

        Fórmula:
            Macro F1 = (1 / |C|) × Σ F1(c)

        Parámetros
        ----------
        per_class_metrics : dict
            Resultado de precision_recall_f1_per_class().

        Retorna
        -------
        float
            Macro F1 entre 0 y 1.
        """
        f1_scores = [m["f1"] for m in per_class_metrics.values()]
        return round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0

    @staticmethod
    def weighted_f1(per_class_metrics: dict[str, dict[str, float]]) -> float:
        """
        Calcula el Weighted F1-Score (promedio ponderado por soporte de clase).

        Parámetros
        ----------
        per_class_metrics : dict
            Resultado de precision_recall_f1_per_class().

        Retorna
        -------
        float
            Weighted F1 entre 0 y 1.
        """
        total_support = sum(m["support"] for m in per_class_metrics.values())
        if total_support == 0:
            return 0.0
        weighted_sum = sum(
            m["f1"] * m["support"] for m in per_class_metrics.values()
        )
        return round(weighted_sum / total_support, 4)

    @staticmethod
    def std_dev(values: list[float]) -> float:
        """
        Calcula la desviación estándar de una lista de valores.
        Utilizado para el análisis de varianza entre folds.

        Parámetros
        ----------
        values : list[float]
            Lista de métricas numéricas.

        Retorna
        -------
        float
            Desviación estándar.
        """
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return round(math.sqrt(variance), 4)


# ──────────────────────────────────────────────────────────────────────────────
# K-Folds Cross Validator
# ──────────────────────────────────────────────────────────────────────────────

class KFoldsCrossValidator:
    """
    Implementación manual de K-Folds Cross Validation.

    El dataset se divide en K particiones (folds) de tamaño aproximadamente
    igual. En cada iteración:
      - K-1 folds se usan para entrenamiento.
      - 1 fold se usa para validación.
    Se repite K veces, rotando el fold de validación.

    Parámetros
    ----------
    k : int
        Número de folds (mínimo 5 según el proyecto). Por defecto 5.
    shuffle : bool
        Si es True, mezcla los índices antes de dividir. Recomendado True
        para evitar sesgo por orden de los datos. Por defecto True.
    random_seed : int
        Semilla para reproducibilidad del shuffle. Por defecto 42.
    """

    def __init__(self, k: int = 5, shuffle: bool = True, random_seed: int = 42):
        if k < 2:
            raise ValueError("k debe ser al menos 2.")
        self.k           = k
        self.shuffle     = shuffle
        self.random_seed = random_seed

    def get_folds(self, n_samples: int) -> list[tuple[list[int], list[int]]]:
        """
        Genera los índices de entrenamiento y validación para cada fold.

        Parámetros
        ----------
        n_samples : int
            Número total de muestras en el dataset.

        Retorna
        -------
        list[tuple[list[int], list[int]]]
            Lista de K tuplas (train_indices, val_indices).
        """
        indices = list(range(n_samples))

        if self.shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(indices)

        fold_size = n_samples // self.k
        remainder = n_samples % self.k

        # Distribuir el remanente entre los primeros folds
        folds_idx = []
        start = 0
        for i in range(self.k):
            end = start + fold_size + (1 if i < remainder else 0)
            folds_idx.append(indices[start:end])
            start = end

        # Construir pares (train, val)
        result = []
        for i in range(self.k):
            val_idx   = folds_idx[i]
            train_idx = []
            for j in range(self.k):
                if j != i:
                    train_idx.extend(folds_idx[j])
            result.append((train_idx, val_idx))

        return result

    def run(
        self,
        X_tokens:  list[list[str]],
        y_labels:  list[str],
        train_fn:  Callable,
        predict_fn: Callable,
        classes:   list[str],
        verbose:   bool = True,
    ) -> dict:
        """
        Ejecuta la validación cruzada completa.

        Parámetros
        ----------
        X_tokens   : list[list[str]]  Corpus tokenizado completo.
        y_labels   : list[str]        Etiquetas completas.
        train_fn   : Callable         Función train(X_train, y_train) → modelo.
        predict_fn : Callable         Función predict(model, X_val)   → list[str].
        classes    : list[str]        Lista de clases únicas.
        verbose    : bool             Imprime progreso por fold.

        Retorna
        -------
        dict
            Resultados agregados: métricas por fold y promedios finales.
        """
        n = len(X_tokens)
        folds = self.get_folds(n)

        fold_results = []
        calc = MetricsCalculator()

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Construir conjuntos de entrenamiento y validación
            X_train = [X_tokens[i] for i in train_idx]
            y_train = [y_labels[i]  for i in train_idx]
            X_val   = [X_tokens[i] for i in val_idx]
            y_val   = [y_labels[i]  for i in val_idx]

            # Entrenar modelo en este fold
            model = train_fn(X_train, y_train)

            # Predecir sobre el fold de validación
            y_pred = predict_fn(model, X_val)

            # Calcular métricas
            acc           = MetricsCalculator.accuracy(y_val, y_pred)
            per_class     = MetricsCalculator.precision_recall_f1_per_class(y_val, y_pred, classes)
            macro_f1      = MetricsCalculator.macro_f1(per_class)
            weighted_f1_v = MetricsCalculator.weighted_f1(per_class)
            conf_matrix   = MetricsCalculator.confusion_matrix(y_val, y_pred, classes)

            fold_result = {
                "fold":         fold_idx + 1,
                "n_train":      len(X_train),
                "n_val":        len(X_val),
                "accuracy":     round(acc, 4),
                "macro_f1":     macro_f1,
                "weighted_f1":  weighted_f1_v,
                "per_class":    per_class,
                "conf_matrix":  conf_matrix,
            }
            fold_results.append(fold_result)

            if verbose:
                print(
                    f"  Fold {fold_idx + 1}/{self.k} | "
                    f"Train: {len(X_train):>5} | Val: {len(X_val):>5} | "
                    f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}"
                )

        # ── Promedios y desviaciones estándar entre folds ──────────────
        accuracies   = [r["accuracy"]    for r in fold_results]
        macro_f1s    = [r["macro_f1"]    for r in fold_results]
        weighted_f1s = [r["weighted_f1"] for r in fold_results]

        summary = {
            "k":                 self.k,
            "fold_results":      fold_results,
            "mean_accuracy":     round(sum(accuracies)   / self.k, 4),
            "std_accuracy":      MetricsCalculator.std_dev(accuracies),
            "mean_macro_f1":     round(sum(macro_f1s)    / self.k, 4),
            "std_macro_f1":      MetricsCalculator.std_dev(macro_f1s),
            "mean_weighted_f1":  round(sum(weighted_f1s) / self.k, 4),
            "std_weighted_f1":   MetricsCalculator.std_dev(weighted_f1s),
            "classes":           classes,
        }

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  K-Folds Summary (K={self.k})")
            print(f"{'─'*60}")
            print(f"  Accuracy  : {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
            print(f"  Macro F1  : {summary['mean_macro_f1']:.4f} ± {summary['std_macro_f1']:.4f}")
            print(f"  Weighted F1:{summary['mean_weighted_f1']:.4f} ± {summary['std_weighted_f1']:.4f}")
            print(f"{'─'*60}\n")

        return summary


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de visualización en consola
# ──────────────────────────────────────────────────────────────────────────────

def print_classification_report(
    per_class_metrics: dict[str, dict[str, float]],
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
) -> None:
    """
    Imprime un reporte de clasificación formateado en consola.

    Parámetros
    ----------
    per_class_metrics : dict   Resultado de MetricsCalculator.precision_recall_f1_per_class().
    accuracy          : float  Accuracy global.
    macro_f1          : float  Macro F1.
    weighted_f1       : float  Weighted F1.
    """
    col_w = max(len(c) for c in per_class_metrics) + 2
    header = f"  {'Clase':<{col_w}} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    for cls, m in sorted(per_class_metrics.items()):
        print(
            f"  {cls:<{col_w}} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['support']:>10}"
        )

    print(f"{'─'*len(header)}")
    print(f"  {'accuracy':<{col_w}} {'':>10} {'':>10} {accuracy:>10.4f} {sum(m['support'] for m in per_class_metrics.values()):>10}")
    print(f"  {'macro avg':<{col_w}} {'':>10} {'':>10} {macro_f1:>10.4f}")
    print(f"  {'weighted avg':<{col_w}} {'':>10} {'':>10} {weighted_f1:>10.4f}")
    print(f"{'─'*len(header)}\n")


def print_confusion_matrix(
    matrix: list[list[int]],
    classes: list[str],
) -> None:
    """
    Imprime la matriz de confusión en consola con formato legible.

    Parámetros
    ----------
    matrix  : list[list[int]]   Matriz de confusión.
    classes : list[str]         Nombres de las clases (en el mismo orden).
    """
    col_w = max(len(c) for c in classes) + 2
    cell_w = 7

    print("\n  Matriz de Confusión (filas=real, columnas=predicho):\n")

    # Encabezado de columnas (abreviado)
    header = f"  {'':<{col_w}}" + "".join(f"{c[:6]:>{cell_w}}" for c in classes)
    print(header)
    print(f"  {'─' * (col_w + cell_w * len(classes))}")

    for i, row in enumerate(matrix):
        row_str = f"  {classes[i]:<{col_w}}" + "".join(f"{val:>{cell_w}}" for val in row)
        print(row_str)
    print()