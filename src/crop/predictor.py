"""
Crop Recommendation Prediction Subsystem (Module C).
Uses Random Forest classifier trained on 2,200 agro-climatic records across 22 crop classes.
"""

import pickle
import warnings
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config.settings import (
    CROP_MODEL_PKL,
    LEGACY_CROP_MODEL_PKL,
    CROP_DATA_CSV,
    CROP_NUTRIENT_REQUIREMENTS,
)

_MODEL_INSTANCE: Optional[RandomForestClassifier] = None

# Feature definitions and expected input ranges
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURE_BOUNDS = {
    "N": (0.0, 140.0, "kg/ha"),
    "P": (5.0, 145.0, "kg/ha"),
    "K": (5.0, 205.0, "kg/ha"),
    "temperature": (8.0, 45.0, "°C"),
    "humidity": (14.0, 100.0, "%"),
    "ph": (3.5, 9.9, "pH"),
    "rainfall": (20.0, 300.0, "mm"),
}


def load_crop_model() -> RandomForestClassifier:
    """Loads the trained Random Forest model with robust version compatibility."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    target_paths = [CROP_MODEL_PKL, LEGACY_CROP_MODEL_PKL]
    for path in target_paths:
        if path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(path, "rb") as f:
                        model = pickle.load(f)
                    if hasattr(model, "predict"):
                        _MODEL_INSTANCE = model
                        return _MODEL_INSTANCE
            except Exception:
                continue

    # Fallback: Train immediately from Crop_recommendation.csv if pickle unavailable
    if CROP_DATA_CSV.exists():
        df = pd.read_csv(CROP_DATA_CSV)
        X = df[FEATURE_NAMES]
        y = df["label"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        with open(CROP_MODEL_PKL, "wb") as f:
            pickle.dump(model, f)
        _MODEL_INSTANCE = model
        return _MODEL_INSTANCE

    raise FileNotFoundError("Could not load or train crop recommendation model.")


def validate_crop_inputs(inputs: Dict[str, float]) -> List[str]:
    """Validates agricultural input parameters against realistic bounds."""
    warnings_list = []
    for feat, (min_val, max_val, unit) in FEATURE_BOUNDS.items():
        if feat not in inputs:
            warnings_list.append(f"Missing feature: {feat}")
            continue
        val = inputs[feat]
        if val < min_val or val > max_val:
            warnings_list.append(
                f"Parameter {feat}={val} {unit} is outside typical crop dataset range [{min_val}, {max_val}]."
            )
    return warnings_list


def predict_crop(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    is_district_estimate: bool = False,
    district_info: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Predicts the optimal crop and top candidate alternatives with probability confidence scores.
    Documents whether inputs are farm-specific or district averages.
    """
    model = load_crop_model()
    raw_inputs = {
        "N": float(nitrogen),
        "P": float(phosphorus),
        "K": float(potassium),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall),
    }

    validation_warnings = validate_crop_inputs(raw_inputs)

    feature_df = pd.DataFrame(
        [[raw_inputs[f] for f in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
        dtype=np.float32,
    )

    # Prediction and probabilities
    pred_class = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]
    classes = model.classes_

    # Rank top 3 recommendations
    ranked_indices = np.argsort(probabilities)[::-1][:3]
    top_candidates = [
        {
            "crop": classes[idx],
            "confidence_pct": round(float(probabilities[idx]) * 100, 2),
            "nutrient_profile": CROP_NUTRIENT_REQUIREMENTS.get(
                classes[idx].lower(),
                {"N": "N/A", "P": "N/A", "K": "N/A", "water_demand": "Moderate"}
            )
        }
        for idx in ranked_indices
    ]

    # Feature importance contribution from model
    feature_importances = dict(zip(FEATURE_NAMES, [round(float(imp), 4) for imp in model.feature_importances_]))

    return {
        "recommended_crop": pred_class,
        "confidence_pct": top_candidates[0]["confidence_pct"],
        "top_candidates": top_candidates,
        "input_features": raw_inputs,
        "is_district_estimate": is_district_estimate,
        "district_context": district_info or {},
        "model_metadata": {
            "model_type": "RandomForestClassifier",
            "n_estimators": getattr(model, "n_estimators", 100),
            "classes_count": len(classes),
            "feature_importances": feature_importances,
            "trained_dataset_size": 2200,
        },
        "validation_warnings": validation_warnings,
        "disclaimer": (
            "Recommendation generated using district-level agro-climatic averages as estimates. "
            "Site-specific micro-climate, soil testing, and local irrigation access should confirm selection."
            if is_district_estimate
            else "Recommendation based on direct soil and weather measurements."
        )
    }
