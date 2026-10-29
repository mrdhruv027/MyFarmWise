"""
Plant Disease Classification REST API Routes.
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import io

from src.disease.predict import predict_disease, load_disease_catalog
from src.disease.model import get_model_architecture_summary

disease_bp = Blueprint("disease", __name__, url_prefix="/api/disease")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@disease_bp.route("/predict", methods=["POST"])
def predict():
    """Predicts plant disease from uploaded leaf image."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in multipart 'image' key"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    crop_filter = request.form.get("crop", None)
    if crop_filter:
        crop_filter = crop_filter.strip()

    try:
        image_bytes = file.read()
        if len(image_bytes) > MAX_CONTENT_LENGTH:
            return jsonify({"success": False, "error": "File size exceeds 10 MB limit"}), 413

        result = predict_disease(image_bytes, crop_filter=crop_filter)
        status_code = 200 if result.get("success") else 200
        return jsonify({"success": True, "data": result}), status_code
    except Exception as e:
        return jsonify({"success": False, "error": "Internal image diagnostic failure"}), 500


@disease_bp.route("/classes", methods=["GET"])
def classes():
    """Returns registry of 38 disease classes across 14 crops."""
    try:
        catalog = load_disease_catalog()
        return jsonify({"success": True, "count": len(catalog), "classes": catalog}), 200
    except Exception:
        return jsonify({"success": False, "error": "Failed to load disease class registry"}), 500


@disease_bp.route("/architecture", methods=["GET"])
def architecture():
    """Returns SE-Attention CNN architecture specifications."""
    try:
        arch = get_model_architecture_summary()
        return jsonify({"success": True, "data": arch}), 200
    except Exception:
        return jsonify({"success": False, "error": "Failed to inspect model architecture"}), 500
