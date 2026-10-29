"""
Crop Recommendation REST API Routes.
"""

from flask import Blueprint, request, jsonify
from src.crop.predictor import predict_crop, FEATURE_BOUNDS
from src.crop.evaluate import run_crop_evaluation
from src.explainability.feature_importance import get_global_feature_importances

crop_bp = Blueprint("crop", __name__, url_prefix="/api/crop")


@crop_bp.route("/recommend", methods=["POST"])
def recommend_crop():
    """Predicts optimal crop from soil & climate features."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request body must be valid JSON"}), 400

    data = request.get_json()
    required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({
            "success": False,
            "error": f"Missing required parameters: {', '.join(missing)}"
        }), 400

    try:
        n = float(data["N"])
        p = float(data["P"])
        k = float(data["K"])
        temp = float(data["temperature"])
        hum = float(data["humidity"])
        ph = float(data["ph"])
        rain = float(data["rainfall"])
        is_district_estimate = bool(data.get("is_district_estimate", False))
        dist_info = data.get("district_context", {})

        result = predict_crop(
            nitrogen=n,
            phosphorus=p,
            potassium=k,
            temperature=temp,
            humidity=hum,
            ph=ph,
            rainfall=rain,
            is_district_estimate=is_district_estimate,
            district_info=dist_info,
        )

        return jsonify({"success": True, "data": result}), 200
    except ValueError as ve:
        return jsonify({"success": False, "error": f"Invalid numerical values: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "Crop prediction failed"}), 500


@crop_bp.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Returns global MDI feature importances for the Random Forest model."""
    try:
        importances = get_global_feature_importances()
        return jsonify({"success": True, "data": importances}), 200
    except Exception:
        return jsonify({"success": False, "error": "Failed to compute feature importances"}), 500


@crop_bp.route("/evaluate", methods=["GET"])
def evaluate_model():
    """Returns validation metrics on the crop dataset."""
    try:
        report = run_crop_evaluation()
        return jsonify({"success": True, "data": report}), 200
    except Exception:
        return jsonify({"success": False, "error": "Crop model evaluation failed"}), 500
