"""
Fertilizer Advisory REST API Routes.
"""

from flask import Blueprint, request, jsonify
from src.fertilizer.recommender import (
    recommend_fertilizer,
    SOIL_TYPES,
)
from config.settings import COMMERCIAL_FERTILIZERS, CROP_NUTRIENT_REQUIREMENTS

fertilizer_bp = Blueprint("fertilizer", __name__, url_prefix="/api/fertilizer")


@fertilizer_bp.route("/recommend", methods=["POST"])
def recommend():
    """Generates nutrient deficit fertilizer recommendations."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request body must be valid JSON"}), 400

    data = request.get_json()
    crop = data.get("crop", "").strip()
    soil_type = data.get("soil_type", "Loamy").strip()

    if not crop:
        return jsonify({"success": False, "error": "Field 'crop' is required"}), 400

    try:
        soil_n = float(data.get("N", 200.0))
        soil_p = float(data.get("P", 15.0))
        soil_k = float(data.get("K", 150.0))
        is_district_estimate = bool(data.get("is_district_estimate", False))

        advisory = recommend_fertilizer(
            crop_name=crop,
            soil_n=soil_n,
            soil_p=soil_p,
            soil_k=soil_k,
            soil_type=soil_type,
            is_district_estimate=is_district_estimate,
        )
        return jsonify({"success": True, "data": advisory}), 200
    except ValueError as ve:
        return jsonify({"success": False, "error": f"Invalid numerical inputs: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "Fertilizer recommendation failed"}), 500


@fertilizer_bp.route("/formulations", methods=["GET"])
def formulations():
    """Returns catalog of 7 commercial fertilizer formulations."""
    return jsonify({
        "success": True,
        "count": len(COMMERCIAL_FERTILIZERS),
        "fertilizers": COMMERCIAL_FERTILIZERS,
        "supported_soils": SOIL_TYPES
    }), 200
