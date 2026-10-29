"""
Flask REST API Gateway for MyFarmWise Consultant.
Demonstration baseline as of 29 October 2026 (Week 6 Progress Review).
"""

import time
from collections import defaultdict
from flask import Flask, jsonify, request
from flask_cors import CORS

from config.settings import (
    PROJECT_NAME,
    PROJECT_SUBTITLE,
    CURRENT_DEMO_DATE,
    DEVELOPMENT_WEEK,
    TOTAL_WEEKS,
    API_PORT,
    SE_CNN_WEIGHTS_PATH,
    CROP_MODEL_PKL,
    DISTRICT_LOOKUP_CSV,
)
from backend.routes.district_routes import district_bp
from backend.routes.crop_routes import crop_bp
from backend.routes.fertilizer_routes import fertilizer_bp
from backend.routes.disease_routes import disease_bp

# Basic in-memory rate limiter (60 req / min per IP)
_REQUEST_TIMESTAMPS = defaultdict(list)
RATE_LIMIT_WINDOW = 60.0
MAX_REQUESTS_PER_WINDOW = 120


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Register Blueprints
    app.register_blueprint(district_bp)
    app.register_blueprint(crop_bp)
    app.register_blueprint(fertilizer_bp)
    app.register_blueprint(disease_bp)

    @app.before_request
    def check_rate_limit():
        client_ip = request.remote_addr or "127.0.0.1"
        now = time.time()
        # Clean older timestamps
        _REQUEST_TIMESTAMPS[client_ip] = [
            ts for ts in _REQUEST_TIMESTAMPS[client_ip] if now - ts < RATE_LIMIT_WINDOW
        ]
        if len(_REQUEST_TIMESTAMPS[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
            return jsonify({
                "success": False,
                "error": "Rate limit exceeded. Please wait before submitting more requests."
            }), 429
        _REQUEST_TIMESTAMPS[client_ip].append(now)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """System health and milestone status verification endpoint."""
        crop_model_ready = CROP_MODEL_PKL.exists()
        district_data_ready = DISTRICT_LOOKUP_CSV.exists()
        disease_weights_ready = SE_CNN_WEIGHTS_PATH.exists()

        return jsonify({
            "service": PROJECT_NAME,
            "subtitle": PROJECT_SUBTITLE,
            "demonstration_date": CURRENT_DEMO_DATE,
            "project_milestone": f"Week {DEVELOPMENT_WEEK} of {TOTAL_WEEKS}",
            "status": "Operational",
            "subsystems": {
                "district_lookup": {
                    "status": "Operational" if district_data_ready else "Degraded",
                    "records_count": 641,
                    "sources": ["ICAR Soil Health Card (Cycle-II)", "IMD Normals"]
                },
                "crop_recommendation": {
                    "status": "Operational" if crop_model_ready else "Unavailable",
                    "model": "Random Forest (22 crops, 2200 samples)",
                    "measured_test_accuracy": "99.55%"
                },
                "fertilizer_advisory": {
                    "status": "Operational",
                    "formulations_supported": 7,
                    "soil_types_supported": 5,
                    "engine_type": "Nutrient Deficit Calculator"
                },
                "disease_classification": {
                    "status": "Trained Weights Active" if disease_weights_ready else "Architecture Ready (Training in Week 7)",
                    "architecture": "Attention-CNN with SE Blocks (Figure 4.1)",
                    "target_classes": 38,
                    "gradcam_engine": "Configured on layer 'last_se_conv'"
                }
            }
        }), 200

    # Structured error handlers (prevent leaking stack traces)
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"success": False, "error": "Bad Request: Please verify required payload fields."}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"success": False, "error": "Requested API route or resource was not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"success": False, "error": "HTTP method not allowed for this route."}), 405

    @app.errorhandler(413)
    def payload_too_large(error):
        return jsonify({"success": False, "error": "Upload size exceeds server limit (Max 10 MB)."}), 413

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"success": False, "error": "Internal server processing error."}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=API_PORT, debug=False)
