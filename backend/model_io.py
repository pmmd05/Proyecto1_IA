"""
model_io.py
-----------
Módulo para guardar y cargar el modelo entrenado.

Formatos soportados:
  - pickle (.pkl) : Serialización binaria de Python. Rápido y compacto.
  - JSON   (.json): Legible por humanos. Útil para inspección y portabilidad.

Ambos formatos guardan el modelo completo: vocabulario, conteos de palabras,
log-priors y metadatos de entrenamiento, de modo que el modelo puede ser
recargado y usado para inferencia sin necesidad de re-entrenar.
"""

import os
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from .naive_bayes import NaiveBayesClassifier


# Carpeta por defecto donde se guardan los modelos
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelManager:
    """
    Gestor de persistencia del modelo Naïve Bayes.

    Parámetros
    ----------
    models_dir : str | Path
        Directorio donde se guardarán y buscarán los modelos.
        Por defecto, la carpeta 'models/' en la raíz del proyecto.
    """

    def __init__(self, models_dir: str | Path = DEFAULT_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Guardado
    # ------------------------------------------------------------------

    def save_pickle(
        self,
        model: NaiveBayesClassifier,
        filename: str = "naive_bayes_model.pkl",
        metadata: dict | None = None,
    ) -> Path:
        """
        Guarda el modelo en formato pickle.

        Pickle serializa el diccionario del modelo junto con metadatos
        opcionales (accuracy, fold, fecha de entrenamiento, etc.).

        Parámetros
        ----------
        model    : NaiveBayesClassifier   Modelo entrenado.
        filename : str                    Nombre del archivo de salida.
        metadata : dict | None            Metadatos adicionales a guardar.

        Retorna
        -------
        Path
            Ruta completa del archivo guardado.
        """
        save_path = self.models_dir / filename
        payload = {
            "model":     model.to_dict(),
            "metadata":  self._build_metadata(model, metadata),
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[ModelManager] Modelo guardado (pickle): {save_path}")
        return save_path

    def save_json(
        self,
        model: NaiveBayesClassifier,
        filename: str = "naive_bayes_model.json",
        metadata: dict | None = None,
    ) -> Path:
        """
        Guarda el modelo en formato JSON legible por humanos.

        Útil para inspección, auditoría o portabilidad entre lenguajes.
        Los Counters se convierten a dicts para ser JSON-serializables.

        Parámetros
        ----------
        model    : NaiveBayesClassifier   Modelo entrenado.
        filename : str                    Nombre del archivo de salida.
        metadata : dict | None            Metadatos adicionales a guardar.

        Retorna
        -------
        Path
            Ruta completa del archivo guardado.
        """
        save_path = self.models_dir / filename
        payload = {
            "model":    model.to_dict(),
            "metadata": self._build_metadata(model, metadata),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[ModelManager] Modelo guardado (JSON): {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def load_pickle(self, filename: str = "naive_bayes_model.pkl") -> tuple[NaiveBayesClassifier, dict]:
        """
        Carga un modelo desde un archivo pickle.

        Parámetros
        ----------
        filename : str   Nombre del archivo a cargar (dentro de models_dir).

        Retorna
        -------
        tuple[NaiveBayesClassifier, dict]
            (modelo reconstruido, metadatos guardados)
        """
        load_path = self.models_dir / filename
        self._check_file_exists(load_path)

        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        model    = NaiveBayesClassifier.from_dict(payload["model"])
        metadata = payload.get("metadata", {})
        print(f"[ModelManager] Modelo cargado (pickle): {load_path}")
        return model, metadata

    def load_json(self, filename: str = "naive_bayes_model.json") -> tuple[NaiveBayesClassifier, dict]:
        """
        Carga un modelo desde un archivo JSON.

        Parámetros
        ----------
        filename : str   Nombre del archivo a cargar (dentro de models_dir).

        Retorna
        -------
        tuple[NaiveBayesClassifier, dict]
            (modelo reconstruido, metadatos guardados)
        """
        load_path = self.models_dir / filename
        self._check_file_exists(load_path)

        with open(load_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        model    = NaiveBayesClassifier.from_dict(payload["model"])
        metadata = payload.get("metadata", {})
        print(f"[ModelManager] Modelo cargado (JSON): {load_path}")
        return model, metadata

    def load(
        self,
        filename: str | None = None,
    ) -> tuple[NaiveBayesClassifier, dict]:
        """
        Carga automáticamente el modelo más reciente del directorio,
        o un archivo específico si se proporciona nombre.

        Detecta el formato (.pkl o .json) automáticamente.

        Parámetros
        ----------
        filename : str | None
            Nombre del archivo. Si es None, carga el modelo más reciente.

        Retorna
        -------
        tuple[NaiveBayesClassifier, dict]
        """
        if filename is None:
            filename = self._find_latest_model()

        load_path = self.models_dir / filename
        ext = load_path.suffix.lower()

        if ext == ".pkl":
            return self.load_pickle(filename)
        elif ext == ".json":
            return self.load_json(filename)
        else:
            raise ValueError(f"Formato no soportado: '{ext}'. Use .pkl o .json.")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """
        Lista todos los modelos guardados en el directorio.

        Retorna
        -------
        list[dict]
            Lista de dicts con nombre, tamaño y fecha de modificación.
        """
        models = []
        for path in sorted(self.models_dir.glob("*.pkl")) + sorted(self.models_dir.glob("*.json")):
            stat = path.stat()
            models.append({
                "filename":  path.name,
                "format":    path.suffix.lstrip("."),
                "size_kb":   round(stat.st_size / 1024, 2),
                "modified":  datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
        return models

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(
        model: NaiveBayesClassifier,
        extra: dict | None,
    ) -> dict:
        """Construye el diccionario de metadatos del modelo."""
        base = {
            "saved_at":      datetime.now(timezone.utc).isoformat(),
            "alpha":         model.alpha,
            "n_classes":     len(model.classes_),
            "classes":       model.classes_,
            "vocab_size":    model.vocabulary_.size if model.vocabulary_ else None,
        }
        if extra:
            base.update(extra)
        return base

    @staticmethod
    def _check_file_exists(path: Path) -> None:
        """Lanza error descriptivo si el archivo no existe."""
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {path}\n"
                f"Ejecuta train.py primero para generar el modelo."
            )

    def _find_latest_model(self) -> str:
        """Retorna el nombre del archivo de modelo más reciente."""
        candidates = list(self.models_dir.glob("*.pkl")) + list(self.models_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No se encontraron modelos en: {self.models_dir}\n"
                f"Ejecuta train.py primero."
            )
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest.name