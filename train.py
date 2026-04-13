"""
train.py
--------
Script principal de entrenamiento del clasificador Naïve Bayes.

Flujo completo:
  1. Carga del dataset (local o HuggingFace).
  2. Preprocesamiento de todos los textos.
  3. K-Folds Cross Validation (K=5) con análisis de métricas.
  4. Entrenamiento del modelo final sobre el 100% de los datos.
  5. Guardado del modelo entrenado en disco (pickle + JSON).

Uso:
  python train.py

Opciones de configuración: ver sección CONFIG al inicio del script.
"""

import time
from pathlib import Path

# Importar módulos del backend
from backend.data_loader   import DataLoader
from backend.preprocessor  import TextPreprocessor
from backend.naive_bayes   import NaiveBayesClassifier
from backend.evaluator     import (
    KFoldsCrossValidator,
    MetricsCalculator,
    print_classification_report,
    print_confusion_matrix,
)
from backend.model_io      import ModelManager


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# Ajusta estos valores según tus necesidades.
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Dataset
    "prefer_local_dataset": True,    # Si True, intenta cargar CSV local primero

    # Preprocesamiento
    "use_lemmatization":  True,      # True = lematización; False = stemming
    "min_token_length":   2,         # Tokens más cortos se descartan

    # Vocabulario
    "vocab_min_freq":     2,         # Frecuencia mínima para incluir una palabra
    "vocab_max_features": None,      # None = sin límite de vocabulario

    # Modelo Naïve Bayes
    "alpha":              1.0,       # Parámetro de Laplace Smoothing

    # K-Folds Cross Validation
    "k_folds":            5,         # Número de folds (mínimo 5)
    "shuffle":            True,
    "random_seed":        42,

    # Train/Test split final (para evaluación post-entrenamiento)
    "test_size":          0.2,

    # Guardado
    "save_pickle":        True,
    "save_json":          True,
    "model_filename_base": "naive_bayes_model",
}
# ══════════════════════════════════════════════════════════════════════════════


def banner(title: str) -> None:
    """Imprime un banner de sección en consola."""
    line = "═" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def main():
    start_total = time.time()

    # ──────────────────────────────────────────────────────────────────
    # 1. CARGA DEL DATASET
    # ──────────────────────────────────────────────────────────────────
    banner("PASO 1 / 5 — Carga del dataset")

    loader = DataLoader()
    texts, labels = loader.load(prefer_local=CONFIG["prefer_local_dataset"])

    print(f"  Total de instancias : {len(texts)}")
    print(f"  Clases únicas       : {sorted(set(labels))}")

    # Split train/test estratificado para evaluación final
    X_train_raw, X_test_raw, y_train, y_test = loader.train_test_split(
        texts, labels,
        test_size=CONFIG["test_size"],
        random_seed=CONFIG["random_seed"],
        stratify=True,
    )

    # ──────────────────────────────────────────────────────────────────
    # 2. PREPROCESAMIENTO
    # ──────────────────────────────────────────────────────────────────
    banner("PASO 2 / 5 — Preprocesamiento de textos")

    preprocessor = TextPreprocessor(
        use_lemmatization=CONFIG["use_lemmatization"],
        min_token_length=CONFIG["min_token_length"],
    )

    print("  Procesando corpus de entrenamiento...", end="", flush=True)
    t0 = time.time()
    X_train_tokens = preprocessor.preprocess_batch(X_train_raw)
    print(f" {time.time()-t0:.1f}s")

    print("  Procesando corpus de prueba...", end="", flush=True)
    t0 = time.time()
    X_test_tokens = preprocessor.preprocess_batch(X_test_raw)
    print(f" {time.time()-t0:.1f}s")

    avg_len = sum(len(t) for t in X_train_tokens) / len(X_train_tokens)
    print(f"  Longitud promedio de tokens por documento: {avg_len:.1f}")

    # ──────────────────────────────────────────────────────────────────
    # 3. K-FOLDS CROSS VALIDATION
    # ──────────────────────────────────────────────────────────────────
    banner(f"PASO 3 / 5 — K-Folds Cross Validation (K={CONFIG['k_folds']})")

    classes = sorted(set(labels))

    def train_fn(X_tr, y_tr):
        """Función de entrenamiento para K-Folds."""
        model = NaiveBayesClassifier(
            alpha=CONFIG["alpha"],
            vocab_min_freq=CONFIG["vocab_min_freq"],
            vocab_max_features=CONFIG["vocab_max_features"],
        )
        model.train(X_tr, y_tr)
        return model

    def predict_fn(model, X_val):
        """Función de predicción para K-Folds."""
        return model.predict_batch(X_val)

    kfcv = KFoldsCrossValidator(
        k=CONFIG["k_folds"],
        shuffle=CONFIG["shuffle"],
        random_seed=CONFIG["random_seed"],
    )

    print()
    cv_results = kfcv.run(
        X_tokens=X_train_tokens,
        y_labels=y_train,
        train_fn=train_fn,
        predict_fn=predict_fn,
        classes=classes,
        verbose=True,
    )

    # Análisis de varianza entre folds
    print("  Varianza entre folds:")
    print(f"    Accuracy  → μ={cv_results['mean_accuracy']:.4f}  σ={cv_results['std_accuracy']:.4f}")
    print(f"    Macro F1  → μ={cv_results['mean_macro_f1']:.4f}  σ={cv_results['std_macro_f1']:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # 4. ENTRENAMIENTO FINAL (sobre todos los datos de entrenamiento)
    # ──────────────────────────────────────────────────────────────────
    banner("PASO 4 / 5 — Entrenamiento del modelo final")

    print(f"  Entrenando con {len(X_train_tokens)} instancias...", end="", flush=True)
    t0 = time.time()

    final_model = NaiveBayesClassifier(
        alpha=CONFIG["alpha"],
        vocab_min_freq=CONFIG["vocab_min_freq"],
        vocab_max_features=CONFIG["vocab_max_features"],
    )
    final_model.train(X_train_tokens, y_train)

    print(f" {time.time()-t0:.1f}s")
    print(f"  Vocabulario final: {final_model.vocabulary_.size} palabras")
    print(f"  Clases: {final_model.classes_}")

    # Evaluación sobre el conjunto de prueba (hold-out)
    print("\n  Evaluando sobre conjunto de prueba (hold-out 20%)...")
    y_pred = final_model.predict_batch(X_test_tokens)

    accuracy     = MetricsCalculator.accuracy(y_test, y_pred)
    per_class    = MetricsCalculator.precision_recall_f1_per_class(y_test, y_pred, classes)
    macro_f1     = MetricsCalculator.macro_f1(per_class)
    weighted_f1  = MetricsCalculator.weighted_f1(per_class)
    conf_matrix  = MetricsCalculator.confusion_matrix(y_test, y_pred, classes)

    print_classification_report(per_class, accuracy, macro_f1, weighted_f1)
    print_confusion_matrix(conf_matrix, classes)

    # ──────────────────────────────────────────────────────────────────
    # 5. GUARDADO DEL MODELO
    # ──────────────────────────────────────────────────────────────────
    banner("PASO 5 / 5 — Guardado del modelo")

    metadata = {
        "test_accuracy":    accuracy,
        "test_macro_f1":    macro_f1,
        "test_weighted_f1": weighted_f1,
        "cv_mean_accuracy": cv_results["mean_accuracy"],
        "cv_mean_macro_f1": cv_results["mean_macro_f1"],
        "n_train":          len(X_train_tokens),
        "n_test":           len(X_test_tokens),
        "k_folds":          CONFIG["k_folds"],
        "alpha":            CONFIG["alpha"],
        "lemmatization":    CONFIG["use_lemmatization"],
        "vocab_min_freq":   CONFIG["vocab_min_freq"],
    }

    manager = ModelManager()

    saved_paths = []
    if CONFIG["save_pickle"]:
        pkl_path = manager.save_pickle(
            final_model,
            filename=f"{CONFIG['model_filename_base']}.pkl",
            metadata=metadata,
        )
        saved_paths.append(pkl_path)

    if CONFIG["save_json"]:
        json_path = manager.save_json(
            final_model,
            filename=f"{CONFIG['model_filename_base']}.json",
            metadata=metadata,
        )
        saved_paths.append(json_path)

    # ──────────────────────────────────────────────────────────────────
    # RESUMEN FINAL
    # ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    banner("ENTRENAMIENTO COMPLETADO")
    print(f"  Tiempo total          : {elapsed:.1f}s")
    print(f"  Accuracy (test)       : {accuracy:.4f}")
    print(f"  Macro F1 (test)       : {macro_f1:.4f}")
    print(f"  CV Accuracy (media)   : {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"  CV Macro F1 (media)   : {cv_results['mean_macro_f1']:.4f} ± {cv_results['std_macro_f1']:.4f}")
    print(f"  Modelos guardados     : {[str(p) for p in saved_paths]}")
    print()

    return final_model, cv_results


if __name__ == "__main__":
    main()