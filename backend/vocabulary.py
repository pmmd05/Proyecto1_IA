"""
vocabulary.py
-------------
Módulo de construcción de vocabulario y representación Bag of Words (BoW).

Responsabilidades:
  - Construir un vocabulario indexado a partir del corpus de entrenamiento.
  - Representar documentos como vectores de frecuencia de palabras.
  - Limitar el vocabulario por frecuencia mínima o tamaño máximo.

Este módulo NO usa scikit-learn ni ninguna biblioteca de ML;
todo se implementa manualmente con estructuras Python estándar.
"""

from collections import Counter


class Vocabulary:
    """
    Vocabulario indexado construido a partir de un corpus tokenizado.

    Parámetros
    ----------
    min_freq : int
        Frecuencia mínima que debe tener una palabra para ser incluida.
        Elimina tokens muy raros que no aportan al modelo. Por defecto 2.
    max_features : int | None
        Número máximo de palabras en el vocabulario (las más frecuentes).
        Si es None, se incluyen todas las que superen min_freq.
    """

    # Token especial para palabras fuera del vocabulario
    UNK_TOKEN = "<UNK>"

    def __init__(self, min_freq: int = 2, max_features: int | None = None):
        self.min_freq    = min_freq
        self.max_features = max_features

        # Mapeos palabra ↔ índice
        self._word2idx: dict[str, int] = {}
        self._idx2word: dict[int, str] = {}

        # Frecuencias globales del corpus de entrenamiento
        self._word_freq: Counter = Counter()

        self._fitted = False

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Número de palabras en el vocabulario (sin contar UNK)."""
        return len(self._word2idx)

    @property
    def words(self) -> list[str]:
        """Lista de palabras en orden de índice."""
        return [self._idx2word[i] for i in range(len(self._idx2word))]

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def fit(self, corpus: list[list[str]]) -> "Vocabulary":
        """
        Construye el vocabulario a partir de un corpus tokenizado.

        Parámetros
        ----------
        corpus : list[list[str]]
            Lista de documentos, donde cada documento es una lista de tokens.

        Retorna
        -------
        Vocabulary
            La instancia misma (para encadenamiento).
        """
        # Contar frecuencias globales
        self._word_freq = Counter()
        for tokens in corpus:
            self._word_freq.update(tokens)

        # Filtrar por frecuencia mínima
        candidates = [
            (word, freq)
            for word, freq in self._word_freq.items()
            if freq >= self.min_freq
        ]

        # Ordenar por frecuencia descendente y aplicar max_features
        candidates.sort(key=lambda x: x[1], reverse=True)
        if self.max_features is not None:
            candidates = candidates[: self.max_features]

        # Construir mapeos
        self._word2idx = {}
        self._idx2word = {}
        for idx, (word, _) in enumerate(candidates):
            self._word2idx[word] = idx
            self._idx2word[idx]  = word

        self._fitted = True
        return self

    def transform(self, tokens: list[str]) -> dict[str, int]:
        """
        Convierte una lista de tokens en un vector de frecuencias (BoW).

        Solo se cuentan las palabras que están en el vocabulario.
        Las palabras desconocidas se ignoran aquí; el modelo Naïve Bayes
        las maneja con Laplace Smoothing.

        Parámetros
        ----------
        tokens : list[str]
            Tokens del documento a representar.

        Retorna
        -------
        dict[str, int]
            Diccionario {palabra: frecuencia} para palabras en vocabulario.
        """
        self._check_fitted()
        bow = Counter()
        for token in tokens:
            if token in self._word2idx:
                bow[token] += 1
        return dict(bow)

    def contains(self, word: str) -> bool:
        """Indica si una palabra está en el vocabulario."""
        self._check_fitted()
        return word in self._word2idx

    def get_index(self, word: str) -> int | None:
        """Retorna el índice de una palabra, o None si no está."""
        return self._word2idx.get(word)

    def get_word(self, index: int) -> str | None:
        """Retorna la palabra correspondiente a un índice."""
        return self._idx2word.get(index)

    def get_frequency(self, word: str) -> int:
        """Retorna la frecuencia global de una palabra en el corpus de fit."""
        return self._word_freq.get(word, 0)

    def summary(self) -> dict:
        """
        Retorna un resumen estadístico del vocabulario.

        Retorna
        -------
        dict
            Estadísticas del vocabulario.
        """
        self._check_fitted()
        total_tokens = sum(self._word_freq.values())
        return {
            "vocabulary_size":    self.size,
            "total_tokens_seen":  total_tokens,
            "min_freq_threshold": self.min_freq,
            "max_features":       self.max_features,
            "top_10_words":       self._word_freq.most_common(10),
        }

    # ------------------------------------------------------------------
    # Serialización (para guardado del modelo)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serializa el vocabulario a un diccionario (compatible con JSON/pickle)."""
        return {
            "min_freq":     self.min_freq,
            "max_features": self.max_features,
            "word2idx":     self._word2idx,
            "word_freq":    dict(self._word_freq),
            "fitted":       self._fitted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        """Reconstruye un Vocabulary desde un diccionario serializado."""
        vocab = cls(min_freq=data["min_freq"], max_features=data["max_features"])
        vocab._word2idx = data["word2idx"]
        vocab._idx2word = {v: k for k, v in data["word2idx"].items()}
        vocab._word_freq = Counter(data["word_freq"])
        vocab._fitted = data["fitted"]
        return vocab

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "El vocabulario no ha sido construido. Llama a fit() primero."
            )