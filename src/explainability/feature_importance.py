"""
Explainable AI Feature Importance Module for Crop Recommendations (Module F).
Provides Mean Decrease in Impurity (MDI / Gini Importance) and local agronomic explanations.
"""

from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

from src.crop.predictor import load_crop_model, FEATURE_NAMES


def get_global_feature_importances() -> Dict[str, float]:
    """Retrieves normalized MDI feature importances from the Random Forest model."""
    model = load_crop_model()
    importances = model.feature_importances_
    return {
        feat: round(float(imp) * 100, 2)
        for feat, imp in zip(FEATURE_NAMES, importances)
    }


def generate_feature_importance_plot() -> Image.Image:
    """Generates an aesthetic matplotlib bar chart of model feature importances."""
    importances = get_global_feature_importances()
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [f[0].capitalize() for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    plt.figure(figsize=(7, 3.8), dpi=150)
    colors = ["#2e7d32", "#388e3c", "#43a047", "#4caf50", "#66bb6a", "#81c784", "#a5d6a7"]
    bars = plt.barh(names[::-1], values[::-1], color=colors[:len(names)], edgecolor="#1b5e20", height=0.6)

    plt.xlabel("Gini Feature Importance (%)", fontsize=10, fontweight="bold", color="#1b5e20")
    plt.title("Random Forest Feature Influence (2,200 Agricultural Records)", fontsize=11, fontweight="bold", color="#1b5e20")
    plt.xlim(0, max(values) + 5)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                 ha="left", va="center", fontsize=9, fontweight="bold", color="#2e7d32")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def explain_local_crop_decision(
    predicted_crop: str,
    inputs: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Generates a localized qualitative explanation showing which agro-climatic parameters
    most strongly motivated the recommended crop.
    """
    explanations = []
    crop_lower = predicted_crop.lower()

    # Rule-based agronomic interpretation based on crop biological traits
    if inputs.get("rainfall", 0) > 180.0:
        explanations.append({
            "feature": "Rainfall",
            "observation": f"{inputs['rainfall']:.1f} mm (High precipitation)",
            "impact": "Favors water-intensive semi-aquatic crops such as Rice and Jute."
        })
    elif inputs.get("rainfall", 0) < 70.0:
        explanations.append({
            "feature": "Rainfall",
            "observation": f"{inputs['rainfall']:.1f} mm (Low precipitation)",
            "impact": "Drives selection towards drought-resilient legumes, pulses, or millets."
        })

    if inputs.get("temperature", 25) > 30.0:
        explanations.append({
            "feature": "Temperature",
            "observation": f"{inputs['temperature']:.1f} °C (Warm tropical)",
            "impact": "Promotes warm-weather crops like Cotton, Sugarcane, or Maize."
        })
    elif inputs.get("temperature", 25) < 22.0:
        explanations.append({
            "feature": "Temperature",
            "observation": f"{inputs['temperature']:.1f} °C (Cool temperate/rabi)",
            "impact": "Supports rabi season crops such as Wheat, Barley, or Lentil."
        })

    if inputs.get("humidity", 60) > 80.0:
        explanations.append({
            "feature": "Humidity",
            "observation": f"{inputs['humidity']:.1f} % (High atmospheric moisture)",
            "impact": "Ideal for coastal and tropical crops like Coconut, Banana, and Rice."
        })

    if inputs.get("ph", 7.0) < 6.0:
        explanations.append({
            "feature": "Soil pH",
            "observation": f"{inputs['ph']:.1f} (Acidic soil)",
            "impact": "Prefers acid-tolerant crops like Coffee, Tea, or specific plantation crops."
        })

    if inputs.get("K", 50) > 120.0:
        explanations.append({
            "feature": "Potassium (K)",
            "observation": f"{inputs['K']:.1f} kg/ha (High potash)",
            "impact": "Essential for fruit crops with high osmotic potassium demand (Grapes, Apple, Banana)."
        })

    if not explanations:
        explanations.append({
            "feature": "Balanced Profile",
            "observation": "Moderate inputs across N-P-K and climate",
            "impact": f"Conditions fit the central optimal growth envelope for {predicted_crop}."
        })

    return explanations
