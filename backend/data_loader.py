"""
data_loader.py
--------------
Módulo de carga y preparación del dataset Bitext Customer Support.

Dataset: Bitext Customer Support LLM Chatbot Training Dataset
URL: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

Columnas relevantes:
  - instruction : texto de la solicitud del cliente (input del modelo).
  - category    : categoría de soporte asignada (label del modelo).

El módulo ofrece dos métodos de carga:
    1. Descarga directa del CSV publicado en Hugging Face (urllib estándar).
  2. Desde un archivo CSV local (si el dataset fue descargado manualmente).
"""

import csv
import random
from collections import Counter
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


# Ruta por defecto para datos locales
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Nombre del archivo CSV esperado si se usa la carga local
LOCAL_CSV_FILENAME = "bitext_customer_support.csv"

# Nombre real del archivo CSV en el repositorio de Hugging Face
REMOTE_CSV_FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

# URL pública de descarga directa del CSV (sin librerías externas)
REMOTE_CSV_URL_TEMPLATE = (
    "https://huggingface.co/datasets/{dataset_name}/resolve/main/{filename}?download=true"
)


class DataLoader:
    """
    Cargador del dataset de soporte al cliente.

    Parámetros
    ----------
    data_dir : str | Path
        Directorio donde se almacena el dataset local.
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Métodos de carga
    # ------------------------------------------------------------------

    def load_from_huggingface(
        self,
        dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split: str = "train",
        save_local: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        Descarga el CSV directamente desde Hugging Face usando urllib.

        Parámetros
        ----------
        dataset_name : str   Identificador del dataset en HuggingFace.
        split        : str   Partición a cargar ('train' o 'test').
        save_local   : bool  Si es True, guarda el dataset como CSV local.

        Retorna
        -------
        tuple[list[str], list[str]]
            (textos, etiquetas)
        """
        if split != "train":
            print(f"[DataLoader] Aviso: split='{split}' no aplica para el CSV publicado; se usará train.")

        print(f"[DataLoader] Descargando dataset desde HuggingFace: {dataset_name}...")
        download_url = REMOTE_CSV_URL_TEMPLATE.format(
            dataset_name=dataset_name,
            filename=REMOTE_CSV_FILENAME,
        )
        temp_csv_path = self.data_dir / f"_tmp_{REMOTE_CSV_FILENAME}"

        try:
            with urlopen(download_url, timeout=120) as response:
                temp_csv_path.write_bytes(response.read())
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"No fue posible descargar el dataset desde HuggingFace: {dataset_name}\n"
                f"Verifica la conexión a internet o descarga manualmente el archivo CSV."
            ) from exc

        try:
            texts, labels = self._load_csv_file(
                temp_csv_path,
                text_col="instruction",
                label_col="category",
                encoding="utf-8",
            )
        finally:
            if temp_csv_path.exists():
                temp_csv_path.unlink()

        print(f"[DataLoader] Dataset cargado: {len(texts)} instancias.")
        self._print_class_distribution(labels)

        if save_local:
            self._save_as_csv(texts, labels)

        return texts, labels

    def load_from_csv(
        self,
        filename: str = LOCAL_CSV_FILENAME,
        text_col: str = "instruction",
        label_col: str = "category",
        encoding: str = "utf-8",
    ) -> tuple[list[str], list[str]]:
        """
        Carga el dataset desde un archivo CSV local.

        Parámetros
        ----------
        filename  : str   Nombre del archivo CSV (dentro de data_dir).
        text_col  : str   Nombre de la columna que contiene el texto.
        label_col : str   Nombre de la columna que contiene la etiqueta.
        encoding  : str   Codificación del archivo. Por defecto 'utf-8'.

        Retorna
        -------
        tuple[list[str], list[str]]
            (textos, etiquetas)
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {file_path}\n"
                f"Opciones:\n"
                f"  1. Descarga desde HuggingFace y guárdalo en: {self.data_dir}/{filename}\n"
                f"  2. Usa load_from_huggingface() para descarga automática."
            )

        texts, labels = self._load_csv_file(
            file_path,
            text_col=text_col,
            label_col=label_col,
            encoding=encoding,
        )

        print(f"[DataLoader] Dataset cargado desde CSV: {len(texts)} instancias.")
        self._print_class_distribution(labels)
        return texts, labels

    def load(
        self,
        prefer_local: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        Método de carga principal. Intenta primero desde CSV local;
        si no existe, descarga desde HuggingFace automáticamente.

        Parámetros
        ----------
        prefer_local : bool
            Si es True, intenta cargar el CSV local antes de descargar.

        Retorna
        -------
        tuple[list[str], list[str]]
            (textos, etiquetas)
        """
        csv_path = self.data_dir / LOCAL_CSV_FILENAME

        if prefer_local and csv_path.exists():
            print(f"[DataLoader] Usando dataset local: {csv_path}")
            return self.load_from_csv()
        else:
            print("[DataLoader] Dataset local no encontrado. Descargando desde HuggingFace...")
            return self.load_from_huggingface(save_local=True)

    # ------------------------------------------------------------------
    # Utilidades del dataset
    # ------------------------------------------------------------------

    @staticmethod
    def train_test_split(
        texts: list[str],
        labels: list[str],
        test_size: float = 0.2,
        random_seed: int = 42,
        stratify: bool = True,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Divide el dataset en conjuntos de entrenamiento y prueba.

        Soporta división estratificada para mantener la proporción de
        clases en ambos conjuntos.

        Parámetros
        ----------
        texts       : list[str]   Textos del corpus.
        labels      : list[str]   Etiquetas correspondientes.
        test_size   : float       Fracción del dataset para prueba (0-1).
        random_seed : int         Semilla de aleatoriedad.
        stratify    : bool        Si True, mantiene proporción de clases.

        Retorna
        -------
        tuple[list[str], list[str], list[str], list[str]]
            (X_train, X_test, y_train, y_test)
        """
        rng = random.Random(random_seed)
        paired = list(zip(texts, labels))

        if stratify:
            # Agrupar por clase
            class_groups: dict[str, list] = {}
            for text, label in paired:
                class_groups.setdefault(label, []).append((text, label))

            train_pairs, test_pairs = [], []
            for group in class_groups.values():
                rng.shuffle(group)
                n_test = max(1, int(len(group) * test_size))
                test_pairs.extend(group[:n_test])
                train_pairs.extend(group[n_test:])
        else:
            rng.shuffle(paired)
            n_test = int(len(paired) * test_size)
            test_pairs  = paired[:n_test]
            train_pairs = paired[n_test:]

        X_train = [p[0] for p in train_pairs]
        y_train = [p[1] for p in train_pairs]
        X_test  = [p[0] for p in test_pairs]
        y_test  = [p[1] for p in test_pairs]

        print(
            f"[DataLoader] Split — Train: {len(X_train)} | Test: {len(X_test)} "
            f"({'stratificado' if stratify else 'aleatorio'})"
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _save_as_csv(self, texts: list[str], labels: list[str]) -> None:
        """Guarda los datos como CSV local para uso futuro."""
        save_path = self.data_dir / LOCAL_CSV_FILENAME
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["instruction", "category"])
            writer.writeheader()
            for text, label in zip(texts, labels):
                writer.writerow({"instruction": text, "category": label})
        print(f"[DataLoader] Dataset guardado localmente en: {save_path}")

    @staticmethod
    def _load_csv_file(
        file_path: Path,
        text_col: str,
        label_col: str,
        encoding: str,
    ) -> tuple[list[str], list[str]]:
        """Carga textos y etiquetas desde un CSV arbitrario."""
        texts, labels = [], []

        with open(file_path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)

            if text_col not in reader.fieldnames:
                raise ValueError(
                    f"Columna '{text_col}' no encontrada. "
                    f"Columnas disponibles: {reader.fieldnames}"
                )
            if label_col not in reader.fieldnames:
                raise ValueError(
                    f"Columna '{label_col}' no encontrada. "
                    f"Columnas disponibles: {reader.fieldnames}"
                )

            for row in reader:
                text = row[text_col].strip()
                label = row[label_col].strip().upper()

                if text and label:
                    texts.append(text)
                    labels.append(label)

        return texts, labels

    @staticmethod
    def _print_class_distribution(labels: list[str]) -> None:
        """Imprime la distribución de clases del dataset."""
        counts = Counter(labels)
        total  = len(labels)
        print(f"\n  Distribución de clases ({len(counts)} clases):")
        for cls, cnt in sorted(counts.items()):
            bar = "█" * int(cnt / total * 40)
            print(f"    {cls:<25} {cnt:>5} ({cnt/total*100:.1f}%)  {bar}")
        print()