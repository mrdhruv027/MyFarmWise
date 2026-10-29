"""
District-Level Soil and Climate Lookup Subsystem (Module B).
Sources agro-climatic normals and ICAR Soil Health Card records.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any
from config.settings import (
    DISTRICT_LOOKUP_CSV,
    DISTRICT_LOOKUP_SQLITE,
    SOIL_HEALTH_CARD_CONVERSION,
)

_CACHED_DF: Optional[pd.DataFrame] = None


def get_district_dataframe() -> pd.DataFrame:
    """Loads district lookup dataset from SQLite database or fallback CSV."""
    global _CACHED_DF
    if _CACHED_DF is not None:
        return _CACHED_DF

    if DISTRICT_LOOKUP_SQLITE.exists():
        try:
            conn = sqlite3.connect(DISTRICT_LOOKUP_SQLITE)
            df = pd.read_sql_query("SELECT * FROM district_lookup", conn)
            conn.close()
            _CACHED_DF = df
            return df
        except Exception:
            pass

    if DISTRICT_LOOKUP_CSV.exists():
        _CACHED_DF = pd.read_csv(DISTRICT_LOOKUP_CSV)
        return _CACHED_DF

    raise FileNotFoundError("District lookup dataset not found in CSV or SQLite.")


def get_all_states() -> List[str]:
    """Returns sorted list of all available states and union territories."""
    df = get_district_dataframe()
    return sorted(df["state"].dropna().unique().tolist())


def get_districts_by_state(state_name: str) -> List[str]:
    """Returns sorted list of districts for the specified state."""
    df = get_district_dataframe()
    matched = df[df["state"].str.upper() == state_name.strip().upper()]
    return sorted(matched["district"].dropna().unique().tolist())


def convert_rating_to_numeric(parameter: str, rating: str) -> Optional[float]:
    """
    Converts ICAR Soil Health Card qualitative nutrient status (Low/Medium/High)
    into representative numeric value (kg/ha) according to documented ICAR thresholds.
    """
    param_clean = parameter.strip().lower()
    if param_clean in SOIL_HEALTH_CARD_CONVERSION:
        table = SOIL_HEALTH_CARD_CONVERSION[param_clean]
        return table.get(rating.strip().title())
    return None


def lookup_district(state_name: str, district_name: str) -> Dict[str, Any]:
    """
    Retrieves the agro-climatic and soil chemistry profile for a district.
    Handles missing soil chemistry honestly without inventing unverified numbers.
    """
    df = get_district_dataframe()
    state_clean = state_name.strip().upper()
    dist_clean = district_name.strip().upper()

    matched = df[(df["state"].str.upper() == state_clean) & (df["district"].str.upper() == dist_clean)]

    # Fallback to partial / prefix match (e.g., GAUTAM BUDDHA vs GAUTAM BUDDHA NAGAR)
    if matched.empty:
        state_rows = df[df["state"].str.upper() == state_clean]
        partial = state_rows[state_rows["district"].apply(
            lambda d: d.upper() in dist_clean or dist_clean in d.upper()
        )]
        if not partial.empty:
            matched = partial

    if matched.empty:
        return {
            "found": False,
            "state": state_clean,
            "district": dist_clean,
            "error": f"No data found for {dist_clean} in {state_clean}.",
            "status": "Record Missing"
        }

    row = matched.iloc[0]

    n_val = None if pd.isna(row["nitrogen_val"]) else float(row["nitrogen_val"])
    p_val = None if pd.isna(row["phosphorus_val"]) else float(row["phosphorus_val"])
    k_val = None if pd.isna(row["potassium_val"]) else float(row["potassium_val"])
    ph_val = None if pd.isna(row["ph_level"]) else float(row["ph_level"])

    # Fallback to documented rating conversion if numeric value is missing but rating is given
    if n_val is None and pd.notna(row.get("nitrogen_rating")):
        n_val = convert_rating_to_numeric("nitrogen", str(row["nitrogen_rating"]))
    if p_val is None and pd.notna(row.get("phosphorus_rating")):
        p_val = convert_rating_to_numeric("phosphorus", str(row["phosphorus_rating"]))
    if k_val is None and pd.notna(row.get("potassium_rating")):
        k_val = convert_rating_to_numeric("potassium", str(row["potassium_rating"]))

    has_soil_data = (n_val is not None) and (p_val is not None) and (k_val is not None) and (ph_val is not None)

    return {
        "found": True,
        "state": row["state"],
        "district": row["district"],
        "soil_type": row["soil_type"] if pd.notna(row["soil_type"]) else "Unknown",
        "has_verified_soil_data": has_soil_data,
        "soil_chemistry": {
            "nitrogen": {
                "value": n_val,
                "unit": "kg/ha",
                "rating": row["nitrogen_rating"] if pd.notna(row["nitrogen_rating"]) else "Not Available",
                "benchmarks": SOIL_HEALTH_CARD_CONVERSION["nitrogen"]["thresholds"]
            },
            "phosphorus": {
                "value": p_val,
                "unit": "kg/ha",
                "rating": row["phosphorus_rating"] if pd.notna(row["phosphorus_rating"]) else "Not Available",
                "benchmarks": SOIL_HEALTH_CARD_CONVERSION["phosphorus"]["thresholds"]
            },
            "potassium": {
                "value": k_val,
                "unit": "kg/ha",
                "rating": row["potassium_rating"] if pd.notna(row["potassium_rating"]) else "Not Available",
                "benchmarks": SOIL_HEALTH_CARD_CONVERSION["potassium"]["thresholds"]
            },
            "ph": {
                "value": ph_val,
                "unit": "pH scale",
                "category": row["ph_category"] if pd.notna(row["ph_category"]) else "Not Available",
                "benchmarks": SOIL_HEALTH_CARD_CONVERSION["ph"]["thresholds"]
            }
        },
        "climate_normals": {
            "annual_rainfall_mm": float(row["annual_rainfall"]) if pd.notna(row["annual_rainfall"]) else None,
            "monsoon_rainfall_mm": float(row["monsoon_rainfall"]) if pd.notna(row["monsoon_rainfall"]) else None,
            "mean_temperature_c": float(row["mean_temperature"]) if pd.notna(row["mean_temperature"]) else None,
            "mean_humidity_pct": float(row["mean_humidity"]) if pd.notna(row["mean_humidity"]) else None,
        },
        "metadata": {
            "source_soil": row["source_soil"] if pd.notna(row["source_soil"]) else "Pending ICAR Lab Ingestion",
            "source_climate": row["source_climate"] if pd.notna(row["source_climate"]) else "IMD 1981-2010 Normals",
            "status": row["status"] if pd.notna(row["status"]) else "Preliminary",
            "limitations": row["limitations"] if pd.notna(row["limitations"]) else "Aggregate statistics; field variation expected."
        }
    }
