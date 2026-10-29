"""
Nutrient-Deficit Fertilizer Recommendation Engine (Module D).
Maps crop nutrient requirements and soil chemistry to 7 commercial formulations.
"""

import math
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from config.settings import (
    COMMERCIAL_FERTILIZERS,
    CROP_NUTRIENT_REQUIREMENTS,
    SOIL_NUTRIENT_AVAILABILITY,
    FERTILIZER_DATA_CSV,
)

SOIL_TYPES = ["Sandy", "Loamy", "Black", "Red", "Clayey"]


def get_crop_requirements(crop_name: str) -> Dict[str, Any]:
    """Retrieves target nutrient requirements (kg/ha) for a specified crop."""
    clean_crop = crop_name.strip().lower()
    return CROP_NUTRIENT_REQUIREMENTS.get(
        clean_crop,
        {"N": 90.0, "P": 45.0, "K": 45.0, "water_demand": "Moderate", "soil_pref": ["Loamy", "Alluvial"]}
    )


def calculate_nutrient_deficit(
    crop_name: str,
    soil_n: float,
    soil_p: float,
    soil_k: float,
    soil_type: str = "Loamy",
) -> Dict[str, Any]:
    """
    Computes effective nutrient deficits (kg/ha) by comparing crop uptake demand
    with season-available soil nutrient reserves, adjusted for soil texture retention.
    """
    crop_req = get_crop_requirements(crop_name)
    n_target = float(crop_req["N"])
    p_target = float(crop_req["P"])
    k_target = float(crop_req["K"])

    # Soil texture availability retention coefficient
    avail_coeff = SOIL_NUTRIENT_AVAILABILITY.get(soil_type.strip().title(), 0.80)

    # Seasonal plant-available fraction from bulk soil chemical test
    seasonal_n_avail = soil_n * avail_coeff * 0.35
    seasonal_p_avail = soil_p * avail_coeff * 0.40
    seasonal_k_avail = soil_k * avail_coeff * 0.30

    deficit_n = max(0.0, n_target - seasonal_n_avail)
    deficit_p = max(0.0, p_target - seasonal_p_avail)
    deficit_k = max(0.0, k_target - seasonal_k_avail)

    return {
        "crop": crop_name,
        "soil_type": soil_type,
        "availability_coefficient": avail_coeff,
        "crop_targets_kg_ha": {"N": n_target, "P": p_target, "K": k_target},
        "soil_reserves_kg_ha": {"N": soil_n, "P": soil_p, "K": soil_k},
        "season_available_kg_ha": {
            "N": round(seasonal_n_avail, 1),
            "P": round(seasonal_p_avail, 1),
            "K": round(seasonal_k_avail, 1),
        },
        "deficits_kg_ha": {
            "N": round(deficit_n, 1),
            "P": round(deficit_p, 1),
            "K": round(deficit_k, 1),
        },
    }


def find_nearest_dataset_recommendation(
    crop_name: str,
    soil_type: str,
    soil_n: float,
    soil_p: float,
    soil_k: float,
) -> Optional[Dict[str, Any]]:
    """Empirically matches inputs against the 99-row dataset as secondary cross-validation."""
    if not FERTILIZER_DATA_CSV.exists():
        return None

    try:
        df = pd.read_csv(FERTILIZER_DATA_CSV)
        candidates = df[(df["Soil Type"].str.lower() == soil_type.lower()) &
                        (df["Crop Type"].str.lower() == crop_name.lower())].copy()

        if candidates.empty:
            candidates = df[df["Soil Type"].str.lower() == soil_type.lower()].copy()
        if candidates.empty:
            candidates = df.copy()

        # Compute normalized euclidean distance on N, P, K
        n_spread = max(1.0, float(df["Nitrogen"].max() - df["Nitrogen"].min()))
        p_spread = max(1.0, float(df["Phosphorous"].max() - df["Phosphorous"].min()))
        k_spread = max(1.0, float(df["Potassium"].max() - df["Potassium"].min()))

        dist = (
            ((candidates["Nitrogen"] - soil_n) / n_spread) ** 2 +
            ((candidates["Phosphorous"] - soil_p) / p_spread) ** 2 +
            ((candidates["Potassium"] - soil_k) / k_spread) ** 2
        )

        best_row = candidates.loc[dist.idxmin()]
        return {
            "fertilizer": best_row["Fertilizer Name"],
            "dataset_soil": best_row["Soil Type"],
            "dataset_crop": best_row["Crop Type"],
        }
    except Exception:
        return None


def recommend_fertilizer(
    crop_name: str,
    soil_n: float,
    soil_p: float,
    soil_k: float,
    soil_type: str = "Loamy",
    is_district_estimate: bool = False,
) -> Dict[str, Any]:
    """
    Selects the optimal commercial fertilizer from 7 formulations based on calculated
    multi-nutrient deficits, and calculates indicative application doses per hectare.
    """
    deficit_info = calculate_nutrient_deficit(crop_name, soil_n, soil_p, soil_k, soil_type)
    deficits = deficit_info["deficits_kg_ha"]
    dn, dp, dk = deficits["N"], deficits["P"], deficits["K"]

    selected_fert = "17-17-17"
    indicative_dose_kg_ha = 0.0
    rationale = ""
    primary_limiting_nutrient = "Balanced"

    # All deficits zero: maintenance only
    if dn <= 2.0 and dp <= 2.0 and dk <= 2.0:
        return {
            "recommended_fertilizer": "Well-Decomposed Farmyard Manure (FYM) / Vermicompost",
            "indicative_quantity_kg_ha": 2500.0,
            "indicative_quantity_bags_ha": 50.0,
            "bag_size_kg": 50,
            "primary_nutrient_addressed": "Soil Organic Matter Maintenance",
            "nutrient_deficits_kg_ha": deficits,
            "soil_type_considered": soil_type,
            "application_schedule": "Incorporate 2.5 t/ha compost during secondary land preparation.",
            "rationale": "Soil reserves currently satisfy optimal crop uptake targets. No synthetic chemical fertilizer is acutely needed.",
            "assumptions_and_units": {
                "units": "kg/ha active elemental nutrient",
                "soil_retention_factor": deficit_info["availability_coefficient"],
                "data_source_mode": "District Aggregate Estimate" if is_district_estimate else "Soil Test Entry"
            },
            "disclaimer": "Indicative decision-support guidance only. Confirm with Soil Health Card laboratory testing before field application."
        }

    # Evaluate dominant deficit proportionally
    if dp >= 15.0 and dk >= 15.0:
        selected_fert = "10-26-26"
        primary_limiting_nutrient = "Phosphorus & Potassium"
        rate_active = 0.26
        target_val = max(dp, dk)
        indicative_dose_kg_ha = target_val / rate_active
        rationale = f"Both phosphorus ({dp} kg/ha) and potassium ({dk} kg/ha) show pronounced deficits. Complex 10-26-26 efficiently replenishes both."
    elif dn >= 25.0 and dp >= 15.0 and dk < 15.0 and dn < 2.0 * dp and dp < 2.0 * dn:
        selected_fert = "28-28" if soil_type in ["Black", "Clayey"] else "20-20"
        primary_limiting_nutrient = "Nitrogen & Phosphorus"
        rate_active = 0.28 if selected_fert == "28-28" else 0.20
        indicative_dose_kg_ha = max(dn, dp) / rate_active
        rationale = f"Co-deficiency of nitrogen ({dn} kg/ha) and phosphorus ({dp} kg/ha). Grade {selected_fert} supplies equal parts N and P."
    elif dn >= 25.0 and dn >= 1.5 * dp and dk < 15.0:
        selected_fert = "Urea"
        primary_limiting_nutrient = "Nitrogen (Vegetative Growth)"
        indicative_dose_kg_ha = dn / 0.46
        rationale = f"Pronounced nitrogen deficit ({dn} kg/ha) is the primary limiting factor for vegetative growth. Urea (46% N) provides high-analysis nitrogen."
    elif dp >= 15.0 and dp >= 1.2 * dn and dk < 15.0:
        selected_fert = "DAP"
        primary_limiting_nutrient = "Phosphorus (Root/Basal)"
        indicative_dose_kg_ha = dp / 0.46
        rationale = f"High phosphorus deficit ({dp} kg/ha) during root development stage. DAP (18-46-0) provides concentrated phosphate with starter nitrogen."
    elif dp >= 15.0 and dn < 15.0 and dk >= 10.0:
        selected_fert = "14-35-14"
        primary_limiting_nutrient = "Phosphorus & Potassium"
        indicative_dose_kg_ha = dp / 0.35
        rationale = f"High phosphate deficit ({dp} kg/ha) with moderate potash need. 14-35-14 offers optimal flowering and fruit set support."
    elif dn >= 20.0 and dp < 10.0 and dk < 10.0:
        selected_fert = "Urea"
        primary_limiting_nutrient = "Nitrogen (Vegetative)"
        indicative_dose_kg_ha = dn / 0.46
        rationale = f"Pronounced nitrogen deficit ({dn} kg/ha) limiting vegetative canopy and leaf greenness. Urea (46% N) is the most efficient source."
    else:
        selected_fert = "17-17-17"
        primary_limiting_nutrient = "Balanced N-P-K"
        max_def = max(dn, dp, dk)
        indicative_dose_kg_ha = max_def / 0.17
        rationale = f"Balanced deficits across N ({dn}), P ({dp}), and K ({dk}) kg/ha. Uniform 17-17-17 grade prevents nutrient imbalances."

    # Clamp dose to realistic agronomic limits (50 kg/ha to 350 kg/ha)
    indicative_dose_kg_ha = float(min(max(indicative_dose_kg_ha, 50.0), 350.0))
    bags_count = round(indicative_dose_kg_ha / 50.0, 1)

    # Empirical dataset check
    empirical_match = find_nearest_dataset_recommendation(crop_name, soil_type, soil_n, soil_p, soil_k)

    fert_meta = COMMERCIAL_FERTILIZERS.get(selected_fert, {"N": 0, "P": 0, "K": 0, "description": ""})

    return {
        "recommended_fertilizer": selected_fert,
        "indicative_quantity_kg_ha": round(indicative_dose_kg_ha, 1),
        "indicative_quantity_bags_ha": bags_count,
        "bag_size_kg": 50,
        "primary_nutrient_addressed": primary_limiting_nutrient,
        "fertilizer_composition": f"{fert_meta['N']}% N, {fert_meta['P']}% P2O5, {fert_meta['K']}% K2O",
        "description": fert_meta["description"],
        "nutrient_deficits_kg_ha": deficits,
        "soil_type_considered": soil_type,
        "soil_retention_multiplier": deficit_info["availability_coefficient"],
        "rationale": rationale,
        "application_schedule": "Apply 50% as basal dose at sowing/transplanting; remaining 50% split across tillering and flowering stages.",
        "empirical_dataset_match": empirical_match,
        "assumptions_and_units": {
            "target_crop": crop_name,
            "units": "Kilograms per hectare (kg/ha)",
            "bag_standard": "Standard 50 kg commercial bag",
            "soil_factor": f"{soil_type} soil retention {deficit_info['availability_coefficient']*100}%",
            "district_estimate_mode": is_district_estimate
        },
        "disclaimer": (
            "NOTICE: The doses computed above are indicative decision-support benchmarks based on generic "
            "crop uptake curves and regional soil estimates. Actual application rates must be adjusted based on "
            "certified local Soil Health Card laboratory tests, irrigation status, and advice from qualified agricultural extension officers."
        )
    }
