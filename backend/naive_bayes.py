"""
naive_bayes.py
--------------
Implementación manual del clasificador Naïve Bayes Multinomial.

Técnicas implementadas (según especificación del proyecto):
  ✔ Bag of Words como representación de documentos.
  ✔ Probabilidades a priori P(c) por clase.
  ✔ Verosimilitud P(w|c) con Laplace Smoothing para palabras no vistas.
  ✔ Suma de logaritmos durante la inferencia para evitar underflow numérico.

Restricción: NO se usa scikit-learn, TensorFlow, Keras, PyTorch ni similares.
             Todo el cálculo probabilístico es implementado desde cero.
"""

import math
from collections import Counter

from .vocabulary import Vocabulary


class NaiveBayesClassifier:
    """
    Clasificador Naïve Bayes Multinomial implementado desde cero.

    Modelo generativo que asume independencia condicional entre palabras
    dada la clase (supuesto Naïve), lo que permite una inferencia eficiente.

    Parámetros
    ----------
    alpha : float
        Parámetro de suavizado de Laplace. alpha=1 es el suavizado estándar.
        Valores más altos suavizan más la distribución. Por defecto 1.0.
    vocabulary : Vocabulary | None
        Vocabulario pre-construido. Si es None, se construye durante el
        entrenamiento a partir del corpus.
    vocab_min_freq : int
        Frecuencia mínima para incluir una palabra en el vocabulario
        (solo aplica si vocabulary es None). Por defecto 2.
    vocab_max_features : int | None
        Tamaño máximo del vocabulario. Por defecto None (sin límite).

    Atributos (disponibles tras train())
    ------------------------------------
    classes_ : list[str]
        Lista de clases únicas en orden alfabético.
    class_log_priors_ : dict[str, float]
        log P(c) para cada clase.
    word_counts_ : dict[str, Counter]
        Frecuencias de cada palabra por clase.
    class_word_totals_ : dict[str, int]
        Total de tokens por clase (para normalización).
    vocabulary_ : Vocabulary
        Vocabulario utilizado durante el entrenamiento.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        vocabulary: Vocabulary | None = None,
        vocab_min_freq: int = 2,
        vocab_max_features: int | None = None,
    ):
        if alpha <= 0:
            raise ValueError("alpha debe ser un valor positivo (ej. 1.0).")

        self.alpha             = alpha
        self._external_vocab   = vocabulary
        self.vocab_min_freq    = vocab_min_freq
        self.vocab_max_features = vocab_max_features

        # Estos atributos se poblan al llamar train()
        self.classes_:           list[str]        = []
        self.class_log_priors_:  dict[str, float] = {}
        self.word_counts_:       dict[str, Counter] = {}
        self.class_word_totals_: dict[str, int]   = {}
        self.vocabulary_:        Vocabulary | None = None
        self._is_trained:        bool              = False

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(self, X_tokens: list[list[str]], y_labels: list[str]) -> "NaiveBayesClassifier":
        """
        Entrena el modelo a partir de documentos tokenizados y sus etiquetas.

        Proceso:
          1. Construye (o usa) el vocabulario.
          2. Cuenta documentos por clase → calcula log P(c).
          3. Cuenta palabras por clase → base para log P(w|c) con Laplace.

        Parámetros
        ----------
        X_tokens : list[list[str]]
            Lista de documentos; cada documento es una lista de tokens.
        y_labels : list[str]
            Etiquetas de clase correspondientes a cada documento.

        Retorna
        -------
        NaiveBayesClassifier
            La instancia misma (para encadenamiento).
        """
        if len(X_tokens) != len(y_labels):
            raise ValueError("X_tokens e y_labels deben tener la misma longitud.")
        if len(X_tokens) == 0:
            raise ValueError("El corpus de entrenamiento está vacío.")

        n_docs = len(y_labels)

        # ── 1. Vocabulario ─────────────────────────────────────────────
        if self._external_vocab is not None:
            self.vocabulary_ = self._external_vocab
        else:
            self.vocabulary_ = Vocabulary(
                min_freq=self.vocab_min_freq,
                max_features=self.vocab_max_features,
            ).fit(X_tokens)

        # ── 2. Clases y log-priors ─────────────────────────────────────
        class_doc_counts: Counter = Counter(y_labels)
        self.classes_ = sorted(class_doc_counts.keys())

        # log P(c) = log(|docs en clase c| / |docs totales|)
        self.class_log_priors_ = {
            c: math.log(count / n_docs)
            for c, count in class_doc_counts.items()
        }

        # ── 3. Conteos de palabras por clase ───────────────────────────
        self.word_counts_       = {c: Counter() for c in self.classes_}
        self.class_word_totals_ = {c: 0         for c in self.classes_}

        for tokens, label in zip(X_tokens, y_labels):
            # Solo contamos palabras que están en el vocabulario
            for token in tokens:
                if self.vocabulary_.contains(token):
                    self.word_counts_[label][token] += 1
                    self.class_word_totals_[label]  += 1

        self._is_trained = True
        return self

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict(self, tokens: list[str]) -> str:
        """
        Predice la clase más probable para un documento tokenizado.

        Utiliza suma de logaritmos para evitar underflow numérico:
            score(c) = log P(c) + Σ log P(w|c)  para w en tokens

        El Laplace Smoothing garantiza que ninguna probabilidad sea 0,
        incluso para palabras que no aparecieron en la clase durante el
        entrenamiento.

        Parámetros
        ----------
        tokens : list[str]
            Tokens del documento a clasificar.

        Retorna
        -------
        str
            Nombre de la clase con mayor puntuación logarítmica.
        """
        self._check_trained()
        log_scores = self._compute_log_scores(tokens)
        return max(log_scores, key=log_scores.get)

    def predict_proba(self, tokens: list[str]) -> dict[str, float]:
        """
        Retorna la probabilidad normalizada de cada clase.

        Para convertir de log-scores a probabilidades, se aplica la
        técnica del 'log-sum-exp' que es numéricamente estable:
            prob(c) = exp(score(c) - max_score) / Σ exp(score(i) - max_score)

        Parámetros
        ----------
        tokens : list[str]
            Tokens del documento.

        Retorna
        -------
        dict[str, float]
            Probabilidad estimada por clase (suman 1.0).
        """
        self._check_trained()
        log_scores = self._compute_log_scores(tokens)

        # Estabilización numérica: restar el máximo antes de aplicar exp
        max_score = max(log_scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in log_scores.items()}
        total = sum(exp_scores.values())

        return {c: v / total for c, v in exp_scores.items()}

    def predict_batch(self, X_tokens: list[list[str]]) -> list[str]:
        """
        Predice la clase para una lista de documentos.

        Parámetros
        ----------
        X_tokens : list[list[str]]
            Lista de documentos tokenizados.

        Retorna
        -------
        list[str]
            Lista de predicciones.
        """
        return [self.predict(tokens) for tokens in X_tokens]

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _log_likelihood(self, word: str, class_: str) -> float:
        """
        Calcula log P(w | c) con Laplace Smoothing.

        Fórmula:
            P(w|c) = (count(w, c) + α) / (Σ_w count(w, c) + α × |V|)

        Donde:
          - count(w, c) : frecuencia de w en la clase c.
          - α           : parámetro de suavizado (Laplace: α=1).
          - |V|         : tamaño del vocabulario.

        Parámetros
        ----------
        word    : str   Palabra a evaluar.
        class_  : str   Nombre de la clase.

        Retorna
        -------
        float
            Logaritmo natural de P(w|c).
        """
        count    = self.word_counts_[class_].get(word, 0)
        total    = self.class_word_totals_[class_]
        vocab_sz = self.vocabulary_.size

        # Laplace smoothing en escala logarítmica
        return math.log((count + self.alpha) / (total + self.alpha * vocab_sz))

    def _log_likelihood_unknown(self, class_: str) -> float:
        """
        Calcula log P(w | c) para palabras fuera del vocabulario.

        Con Laplace Smoothing, una palabra no vista tiene count = 0,
        por lo que la fórmula se reduce a:
            P(w_unk | c) = α / (total_c + α × |V|)

        Parámetros
        ----------
        class_ : str
            Nombre de la clase.

        Retorna
        -------
        float
            Logaritmo natural de P(w_unk | c).
        """
        total    = self.class_word_totals_[class_]
        vocab_sz = self.vocabulary_.size
        return math.log(self.alpha / (total + self.alpha * vocab_sz))

    def _compute_log_scores(self, tokens: list[str]) -> dict[str, float]:
        """
        Calcula el score logarítmico para cada clase dado un documento.

        score(c) = log P(c) + Σ_{w ∈ tokens} log P(w | c)

        Usando logaritmos, el producto de probabilidades se convierte en
        suma, evitando el underflow numérico que ocurre al multiplicar
        muchos valores pequeños.

        Parámetros
        ----------
        tokens : list[str]
            Tokens del documento a clasificar.

        Retorna
        -------
        dict[str, float]
            Score logarítmico por clase.
        """
        scores = {}
        for c in self.classes_:
            # Comenzar con el log-prior de la clase
            score = self.class_log_priors_[c]

            # Sumar log-verosimilitud de cada token
            for word in tokens:
                if self.vocabulary_.contains(word):
                    score += self._log_likelihood(word, c)
                else:
                    # Palabra fuera del vocabulario: Laplace igual maneja cuenta=0
                    score += self._log_likelihood_unknown(c)

            scores[c] = score
        return scores

    def _check_trained(self):
        """Lanza error si el modelo no ha sido entrenado."""
        if not self._is_trained:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llama a train() primero."
            )

    # ------------------------------------------------------------------
    # Serialización (para guardar/cargar el modelo)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serializa el modelo a un diccionario (compatible con pickle/JSON)."""
        self._check_trained()
        return {
            "alpha":              self.alpha,
            "vocab_min_freq":     self.vocab_min_freq,
            "vocab_max_features": self.vocab_max_features,
            "classes":            self.classes_,
            "class_log_priors":   self.class_log_priors_,
            "word_counts":        {c: dict(cnt) for c, cnt in self.word_counts_.items()},
            "class_word_totals":  self.class_word_totals_,
            "vocabulary":         self.vocabulary_.to_dict(),
            "is_trained":         True,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NaiveBayesClassifier":
        """Reconstruye el modelo desde un diccionario serializado."""
        instance = cls(
            alpha=data["alpha"],
            vocab_min_freq=data["vocab_min_freq"],
            vocab_max_features=data["vocab_max_features"],
        )
        instance.classes_           = data["classes"]
        instance.class_log_priors_  = data["class_log_priors"]
        instance.word_counts_       = {c: Counter(cnt) for c, cnt in data["word_counts"].items()}
        instance.class_word_totals_ = data["class_word_totals"]
        instance.vocabulary_        = Vocabulary.from_dict(data["vocabulary"])
        instance._is_trained        = True
        return instance

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"NaiveBayesClassifier("
            f"alpha={self.alpha}, "
            f"status={status}, "
            f"classes={self.classes_ if self._is_trained else 'N/A'}"
            f")"
        )