"""
preprocessor.py
---------------
Módulo de preprocesamiento de texto para el clasificador Naïve Bayes.

Responsabilidades:
  - Limpieza de texto (eliminación de ruido, caracteres especiales, placeholders)
  - Tokenización usando NLTK
  - Eliminación de stopwords
  - Stemming (PorterStemmer) o Lematización (WordNetLemmatizer), configurable

Nota: Se utiliza NLTK únicamente para tokenización y stopwords,
      tal como lo permite la especificación del proyecto.
"""

import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ──────────────────────────────────────────────────────────────────────────────
# Stopwords en inglés embebidas (fallback sin necesidad de descarga NLTK)
# ──────────────────────────────────────────────────────────────────────────────
_ENGLISH_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn","get","got","let","also","would","could","like","one","two","three",
    "want","need","please","hello","hi","dear","thanks","thank","regards","sir",
    "madam","help","via","us","make","take","know","use","time","way","may",
    "see","come","go","look","think","really","still","already","even","back",
    "since","any","much","many","well","must","able","good","new","first","last",
    "long","great","little","own","right","big","high","next","early","never",
    "always","often","said","told","asked","says","made","shall","provide",
    "try","keep","give","show","seem","feel","put","turn","mean","start","might",
    "ever","say","man","woman","today","thing","day","week","month","year",
    "rather","however","though","although","therefore","thus","yet","hence",
    "despite","another","every","else","without","within","whether","against",
    "among","across","along","around","behind","beside","besides","beyond",
    "except","following","toward","towards","upon","whose","whom","both","either",
    "neither","several","enough","almost","quite","just","rather","nearly",
}


def _try_download_nltk_resources():
    """
    Intenta descargar recursos NLTK. Si falla (sin conexión), usa el fallback
    embebido. Esta función es tolerante a fallos de red.
    """
    resources = [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
        ("corpora/omw-1.4",      "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass  # Sin conexión: se usará el fallback embebido


_try_download_nltk_resources()


class TextPreprocessor:
    """
    Preprocesador de texto para solicitudes de soporte al cliente.

    Parámetros
    ----------
    use_lemmatization : bool
        Si es True usa lematización; si es False usa stemming.
        Por defecto True (lematización produce tokens más legibles).
    language : str
        Idioma para las stopwords de NLTK. Por defecto 'english'.
    min_token_length : int
        Longitud mínima de un token para ser incluido en el resultado.
    """

    # Patrón para eliminar placeholders del tipo {{Order Number}}, {{Name}}, etc.
    PLACEHOLDER_PATTERN = re.compile(r"\{\{[^}]+\}\}")

    # Caracteres de puntuación a eliminar (se agrega al set estándar de Python)
    EXTRA_PUNCT = set("''""—–…")

    def __init__(
        self,
        use_lemmatization: bool = True,
        language: str = "english",
        min_token_length: int = 2,
    ):
        self.use_lemmatization = use_lemmatization
        self.language = language
        self.min_token_length = min_token_length

        # Stopwords: NLTK si está disponible, lista embebida como fallback
        try:
            self._stopwords = set(stopwords.words(language))
        except Exception:
            self._stopwords = _ENGLISH_STOPWORDS.copy()

        # Stemmer / Lematizador — con fallback si WordNet no está disponible
        if use_lemmatization:
            try:
                lem = WordNetLemmatizer()
                lem.lemmatize("test")   # verificar que WordNet está disponible
                self._normalizer = lem
                self._use_lemmatization_active = True
            except Exception:
                # Fallback: usar stemmer que no requiere corpus externo
                self._normalizer = PorterStemmer()
                self._use_lemmatization_active = False
        else:
            self._normalizer = PorterStemmer()
            self._use_lemmatization_active = False

        # Verificar si NLTK punkt está disponible para tokenización
        self._nltk_tokenizer_available = False
        try:
            nltk.data.find("tokenizers/punkt_tab")
            self._nltk_tokenizer_available = True
        except LookupError:
            try:
                nltk.data.find("tokenizers/punkt")
                self._nltk_tokenizer_available = True
            except LookupError:
                pass

        # Set completo de puntuación
        self._punct_table = str.maketrans("", "", string.punctuation + "".join(self.EXTRA_PUNCT))

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Limpia el texto crudo aplicando (en orden):
          1. Conversión a minúsculas.
          2. Eliminación de placeholders {{...}}.
          3. Eliminación de URLs.
          4. Eliminación de menciones y hashtags.
          5. Eliminación de números sueltos.
          6. Eliminación de caracteres de puntuación.
          7. Normalización de espacios.

        Parámetros
        ----------
        text : str
            Texto de entrada sin procesar.

        Retorna
        -------
        str
            Texto limpio.
        """
        if not isinstance(text, str):
            return ""

        # 1. Minúsculas
        text = text.lower()

        # 2. Eliminar placeholders  {{Order Number}}, {{Name}}, etc.
        text = self.PLACEHOLDER_PATTERN.sub(" ", text)

        # 3. Eliminar URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 4. Menciones y hashtags
        text = re.sub(r"[@#]\w+", " ", text)

        # 5. Números sueltos (deja palabras que contengan letras)
        text = re.sub(r"\b\d+\b", " ", text)

        # 6. Puntuación
        text = text.translate(self._punct_table)

        # 7. Normalizar espacios
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> list[str]:
        """
        Tokeniza el texto usando el tokenizador de NLTK.

        Parámetros
        ----------
        text : str
            Texto ya limpio.

        Retorna
        -------
        list[str]
            Lista de tokens.
        """
        return word_tokenize(text, language=self.language)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """
        Filtra los tokens que son stopwords o tienen longitud menor
        al mínimo configurado.

        Parámetros
        ----------
        tokens : list[str]
            Tokens de entrada.

        Retorna
        -------
        list[str]
            Tokens sin stopwords.
        """
        return [
            t for t in tokens
            if t not in self._stopwords and len(t) >= self.min_token_length
        ]

    def normalize(self, tokens: list[str]) -> list[str]:
        """
        Aplica stemming o lematización a cada token según la configuración.

        Parámetros
        ----------
        tokens : list[str]
            Tokens filtrados.

        Retorna
        -------
        list[str]
            Tokens normalizados.
        """
        if self.use_lemmatization:
            return [self._normalizer.lemmatize(t) for t in tokens]
        else:
            return [self._normalizer.stem(t) for t in tokens]

    def preprocess(self, text: str) -> list[str]:
        """
        Pipeline completo: clean → tokenize → remove_stopwords → normalize.

        Parámetros
        ----------
        text : str
            Texto crudo de entrada.

        Retorna
        -------
        list[str]
            Lista de tokens listos para ser usados en el modelo.
        """
        cleaned   = self.clean(text)
        tokens    = self.tokenize(cleaned)
        filtered  = self.remove_stopwords(tokens)
        normalized = self.normalize(filtered)
        return normalized

    def preprocess_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Aplica preprocess() a una lista de textos.

        Parámetros
        ----------
        texts : list[str]
            Lista de textos crudos.

        Retorna
        -------
        list[list[str]]
            Lista de listas de tokens.
        """
        return [self.preprocess(text) for text in texts]