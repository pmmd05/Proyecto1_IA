"""
app.py
------
Servidor Flask que integra el motor de inferencia Naïve Bayes
con la interfaz web del sistema de clasificación de tickets.

Endpoints:
  GET  /                  → Página principal (interfaz web).
  POST /api/classify      → Clasifica un ticket y retorna categoría + probabilidades.
  GET  /api/health        → Estado del servidor y del modelo.
  GET  /api/classes       → Lista de categorías disponibles.

Uso:
  python app.py
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from backend.preprocessor import TextPreprocessor
from backend.model_io import ModelManager

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Inicialización de Flask
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
)
CORS(app)

# ──────────────────────────────────────────────────────────────────────────────
# Carga del modelo al arrancar el servidor
# ──────────────────────────────────────────────────────────────────────────────
_model = None
_model_metadata = {}
_preprocessor = None
_model_load_error = None


def load_model():
    """Carga el modelo entrenado y el preprocesador al iniciar la aplicación."""
    global _model, _model_metadata, _preprocessor, _model_load_error

    try:
        manager = ModelManager()
        _model, _model_metadata = manager.load()
        _preprocessor = TextPreprocessor(use_lemmatization=True, min_token_length=2)
        logger.info(f"Modelo cargado exitosamente | Clases: {_model.classes_}")
        logger.info(f"Vocabulario: {_model.vocabulary_.size} palabras")
    except FileNotFoundError as e:
        _model_load_error = str(e)
        logger.warning(f"Modelo no encontrado: {e}")
        logger.warning("Ejecuta 'python train.py' para entrenar y guardar el modelo primero.")
    except Exception as e:
        _model_load_error = str(e)
        logger.error(f"Error al cargar el modelo: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────────────────────────────────────

# Mapeo de categorías a íconos y colores para el frontend
CATEGORY_META = {
    "ORDER":       {"icon": "📦", "color": "#3b82f6", "label": "Órdenes"},
    "BILLING":     {"icon": "💳", "color": "#f59e0b", "label": "Facturación"},
    "SHIPPING":    {"icon": "🚚", "color": "#10b981", "label": "Envíos"},
    "REFUND":      {"icon": "↩️",  "color": "#ef4444", "label": "Reembolsos"},
    "ACCOUNT":     {"icon": "👤", "color": "#8b5cf6", "label": "Cuenta"},
    "CANCEL":      {"icon": "❌", "color": "#f97316", "label": "Cancelaciones"},
    "CANCELLATION":{"icon": "❌", "color": "#f97316", "label": "Cancelaciones"},
    "CONTACT":     {"icon": "📞", "color": "#06b6d4", "label": "Contacto"},
    "DELIVERY":    {"icon": "📬", "color": "#84cc16", "label": "Entrega"},
    "FEEDBACK":    {"icon": "💬", "color": "#ec4899", "label": "Retroalimentación"},
    "INVOICE":     {"icon": "🧾", "color": "#14b8a6", "label": "Facturas"},
    "PAYMENT":     {"icon": "💰", "color": "#a855f7", "label": "Pagos"},
}

DEFAULT_META = {"icon": "🎫", "color": "#94a3b8", "label": "Soporte General"}


def _get_category_meta(category: str) -> dict:
    """Retorna metadatos visuales para una categoría."""
    return CATEGORY_META.get(category.upper(), DEFAULT_META)


def _generate_ticket_id() -> str:
    """Genera un ID de ticket único con formato legible."""
    prefix = "TKT"
    short_uuid = str(uuid.uuid4()).replace("-", "").upper()[:8]
    return f"{prefix}-{short_uuid}"


# ──────────────────────────────────────────────────────────────────────────────
# Rutas
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Sirve la página principal de la interfaz web."""
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Clasifica el texto de un ticket de soporte.

    Cuerpo de la petición (JSON):
    {
        "subject":     str,   (opcional) Asunto del ticket
        "description": str,   (requerido) Descripción completa del ticket
        "ticket_id":   str    (opcional) ID del ticket (se genera si no se proporciona)
    }

    Respuesta (JSON):
    {
        "ticket_id":    str,
        "category":     str,
        "confidence":   float,
        "probabilities": [{"class": str, "probability": float, "meta": {...}}, ...],
        "meta":         {"icon": str, "color": str, "label": str},
        "processed_at": str
    }
    """
    # Verificar que el modelo está disponible
    if _model is None:
        return jsonify({
            "error": "Modelo no disponible. Ejecuta 'python train.py' primero.",
            "detail": _model_load_error,
        }), 503

    # Parsear cuerpo de la petición
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Cuerpo de la petición inválido. Se esperaba JSON."}), 400

    subject     = data.get("subject", "").strip()
    description = data.get("description", "").strip()
    ticket_id   = data.get("ticket_id", "").strip() or _generate_ticket_id()

    # Validar que haya al menos una descripción
    if not description:
        return jsonify({"error": "El campo 'description' es requerido y no puede estar vacío."}), 400

    # Combinar subject y description para clasificar
    full_text = f"{subject} {description}".strip() if subject else description

    try:
        # Preprocesar el texto
        tokens = _preprocessor.preprocess(full_text)

        if not tokens:
            return jsonify({
                "error": "El texto no contiene palabras útiles después del preprocesamiento.",
                "hint": "Intenta con una descripción más detallada.",
            }), 422

        # Clasificar
        category = _model.predict(tokens)
        probabilities = _model.predict_proba(tokens)

        # Ordenar probabilidades de mayor a menor
        sorted_probs = sorted(
            [
                {
                    "class":       cls,
                    "probability": round(prob, 6),
                    "percentage":  round(prob * 100, 2),
                    "meta":        _get_category_meta(cls),
                }
                for cls, prob in probabilities.items()
            ],
            key=lambda x: x["probability"],
            reverse=True,
        )

        confidence = probabilities[category]
        meta = _get_category_meta(category)

        response = {
            "ticket_id":     ticket_id,
            "category":      category,
            "confidence":    round(confidence, 6),
            "confidence_pct": round(confidence * 100, 2),
            "probabilities": sorted_probs,
            "meta":          meta,
            "tokens_used":   len(tokens),
            "processed_at":  datetime.utcnow().isoformat() + "Z",
        }

        logger.info(
            f"Clasificado | Ticket: {ticket_id} | "
            f"Categoría: {category} | Confianza: {confidence:.2%}"
        )
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error al clasificar ticket {ticket_id}: {e}")
        return jsonify({"error": "Error interno al clasificar el ticket.", "detail": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Retorna el estado del servidor y del modelo."""
    model_ok = _model is not None

    status = {
        "status":     "ok" if model_ok else "degraded",
        "model":      {
            "loaded":      model_ok,
            "error":       _model_load_error if not model_ok else None,
            "classes":     _model.classes_    if model_ok else [],
            "vocab_size":  _model.vocabulary_.size if model_ok else 0,
            "alpha":       _model.alpha        if model_ok else None,
        },
        "metadata":    _model_metadata,
        "server_time": datetime.utcnow().isoformat() + "Z",
    }
    return jsonify(status), 200 if model_ok else 503


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Retorna las clases disponibles con sus metadatos visuales."""
    if _model is None:
        return jsonify({"error": "Modelo no disponible."}), 503

    classes = [
        {"class": cls, "meta": _get_category_meta(cls)}
        for cls in _model.classes_
    ]
    return jsonify({"classes": classes, "total": len(classes)}), 200


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )