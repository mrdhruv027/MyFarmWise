"""
District Lookup REST API Routes.
"""

from flask import Blueprint, request, jsonify
from src.data.district_lookup import (
    get_all_states,
    get_districts_by_state,
    lookup_district,
)

district_bp = Blueprint("districts", __name__, url_prefix="/api/districts")


@district_bp.route("/states", methods=["GET"])
def list_states():
    """Returns sorted list of all available states."""
    try:
        states = get_all_states()
        return jsonify({"success": True, "count": len(states), "states": states}), 200
    except Exception as e:
        return jsonify({"success": False, "error": "Failed to load state registry"}), 500


@district_bp.route("", methods=["GET"])
def list_districts():
    """Returns districts for a query parameter state."""
    state = request.args.get("state", "").strip()
    if not state:
        return jsonify({"success": False, "error": "Query parameter 'state' is required"}), 400

    try:
        districts = get_districts_by_state(state)
        return jsonify({
            "success": True,
            "state": state,
            "count": len(districts),
            "districts": districts
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": "Failed to fetch districts"}), 500


@district_bp.route("/lookup", methods=["GET"])
def get_district_profile():
    """Returns agro-climatic profile for state and district."""
    state = request.args.get("state", "").strip()
    district = request.args.get("district", "").strip()

    if not state or not district:
        return jsonify({
            "success": False,
            "error": "Both 'state' and 'district' query parameters are required"
        }), 400

    try:
        profile = lookup_district(state, district)
        if not profile.get("found"):
            return jsonify({
                "success": False,
                "error": profile.get("error", "District not found in database")
            }), 404

        return jsonify({"success": True, "data": profile}), 200
    except Exception as e:
        return jsonify({"success": False, "error": "Database lookup encountered an error"}), 500
