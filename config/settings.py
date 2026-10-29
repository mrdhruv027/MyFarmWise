"""
Configuration and settings module for MyFarmWise Consultant.
Demonstration baseline as of 29 October 2026.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
BACKEND_DIR = BASE_DIR / "backend"
STATIC_DIR = BASE_DIR / "static"
DOCS_DIR = BASE_DIR / "docs"

# Data Files
DISTRICT_LOOKUP_CSV = DATA_DIR / "district_lookup.csv"
DISTRICT_LOOKUP_SQLITE = DATA_DIR / "district_lookup.sqlite"
CROP_DATA_CSV = DATA_DIR / "Crop_recommendation.csv"
FERTILIZER_DATA_CSV = DATA_DIR / "fertilizer_data.csv"
DISEASE_CLASSES_JSON = CONFIG_DIR / "disease_classes.json"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# Model Files
CROP_MODEL_PKL = MODELS_DIR / "crop_rf_model.pkl"
LEGACY_CROP_MODEL_PKL = MODELS_DIR / "RandomForest.pkl"
SE_CNN_WEIGHTS_PATH = MODELS_DIR / "se_attention_cnn_weights.weights.h5"
BASELINE_DISEASE_MODEL_PATH = MODELS_DIR / "baseline_plant_disease_model_128x128.keras"

# Static Assets
LOGO_PATH = STATIC_DIR / "logo.jpg"
HERO_IMAGE_PATH = STATIC_DIR / "amit-front2.jpg"

# System metadata
PROJECT_NAME = "MyFarmWise Consultant"
PROJECT_SUBTITLE = "Region-Aware, Explainable AI for Crop, Fertilizer & Disease Advisory"
CURRENT_DEMO_DATE = "29 October 2026"
DEVELOPMENT_WEEK = 6
TOTAL_WEEKS = 16
API_PORT = 5000
API_BASE_URL = f"http://127.0.0.1:{API_PORT}/api"

# Rating-to-Number Conversion Rules (ICAR Soil Health Card standard benchmarks)
SOIL_HEALTH_CARD_CONVERSION = {
    "nitrogen": {
        "Low": 200.0,       # Benchmark: < 280 kg/ha
        "Medium": 380.0,    # Benchmark: 280 - 560 kg/ha
        "High": 600.0,      # Benchmark: > 560 kg/ha
        "units": "kg/ha",
        "thresholds": "<280 Low, 280-560 Med, >560 High"
    },
    "phosphorus": {
        "Low": 8.5,         # Benchmark: < 10 kg/ha
        "Medium": 18.0,     # Benchmark: 10 - 25 kg/ha
        "High": 32.0,       # Benchmark: > 25 kg/ha
        "units": "kg/ha",
        "thresholds": "<10 Low, 10-25 Med, >25 High"
    },
    "potassium": {
        "Low": 85.0,        # Benchmark: < 108 kg/ha
        "Medium": 190.0,    # Benchmark: 108 - 280 kg/ha
        "High": 320.0,      # Benchmark: > 280 kg/ha
        "units": "kg/ha",
        "thresholds": "<108 Low, 108-280 Med, >280 High"
    },
    "ph": {
        "Acidic": 5.8,
        "Neutral": 6.8,
        "Alkaline": 8.1,
        "units": "pH scale",
        "thresholds": "<6.5 Acidic, 6.5-7.5 Neutral, >7.5 Alkaline"
    }
}

# Commercial Fertilizer Formulations (% N - % P2O5 - % K2O)
COMMERCIAL_FERTILIZERS = {
    "Urea": {
        "N": 46.0, "P": 0.0, "K": 0.0,
        "description": "High-nitrogen synthetic fertilizer (46-0-0) optimal for rapid vegetative leaf growth."
    },
    "DAP": {
        "N": 18.0, "P": 46.0, "K": 0.0,
        "description": "Diammonium Phosphate (18-46-0) optimal for vigorous root system development and early establishment."
    },
    "14-35-14": {
        "N": 14.0, "P": 35.0, "K": 14.0,
        "description": "Complex multi-nutrient formulation rich in phosphorus with balanced potassium for flowering and fruiting."
    },
    "28-28": {
        "N": 28.0, "P": 28.0, "K": 0.0,
        "description": "Balanced high-analysis nitrogen-phosphorus blend (28-28-0) for vegetative crops lacking potassium deficiency."
    },
    "17-17-17": {
        "N": 17.0, "P": 17.0, "K": 17.0,
        "description": "Completely balanced uniform N-P-K grade for multi-stage basal and top dressing in diverse crops."
    },
    "20-20": {
        "N": 20.0, "P": 20.0, "K": 0.0,
        "description": "Ammonium Phosphate Sulphate grade (20-20-0) providing accessible nitrogen, phosphorus and sulphur."
    },
    "10-26-26": {
        "N": 10.0, "P": 26.0, "K": 26.0,
        "description": "High phosphorus and potassium formulation for root tubers, pulses, and grain filling resistance."
    }
}

# Standard Crop Nutrient Targets (kg/ha optimal crop uptake range)
CROP_NUTRIENT_REQUIREMENTS = {
    "rice": {"N": 100.0, "P": 40.0, "K": 40.0, "water_demand": "High", "soil_pref": ["Clayey", "Loamy"]},
    "wheat": {"N": 120.0, "P": 60.0, "K": 40.0, "water_demand": "Moderate", "soil_pref": ["Loamy", "Alluvial"]},
    "maize": {"N": 120.0, "P": 60.0, "K": 50.0, "water_demand": "Moderate", "soil_pref": ["Sandy", "Loamy"]},
    "cotton": {"N": 100.0, "P": 50.0, "K": 50.0, "water_demand": "Moderate", "soil_pref": ["Black", "Alluvial"]},
    "sugarcane": {"N": 150.0, "P": 75.0, "K": 75.0, "water_demand": "High", "soil_pref": ["Loamy", "Black"]},
    "paddy": {"N": 100.0, "P": 40.0, "K": 40.0, "water_demand": "High", "soil_pref": ["Clayey", "Alluvial"]},
    "millets": {"N": 50.0, "P": 25.0, "K": 25.0, "water_demand": "Low", "soil_pref": ["Sandy", "Red"]},
    "pulses": {"N": 25.0, "P": 50.0, "K": 25.0, "water_demand": "Low", "soil_pref": ["Loamy", "Clayey"]},
    "ground nuts": {"N": 30.0, "P": 60.0, "K": 40.0, "water_demand": "Low", "soil_pref": ["Red", "Sandy"]},
    "barley": {"N": 60.0, "P": 30.0, "K": 30.0, "water_demand": "Low", "soil_pref": ["Sandy", "Loamy"]},
    "oil seeds": {"N": 60.0, "P": 40.0, "K": 40.0, "water_demand": "Low", "soil_pref": ["Black", "Loamy"]},
    "tobacco": {"N": 70.0, "P": 50.0, "K": 80.0, "water_demand": "Moderate", "soil_pref": ["Red", "Sandy"]},
    "apple": {"N": 80.0, "P": 40.0, "K": 80.0, "water_demand": "Moderate", "soil_pref": ["Loamy"]},
    "banana": {"N": 200.0, "P": 50.0, "K": 200.0, "water_demand": "High", "soil_pref": ["Clayey", "Loamy"]},
    "blackgram": {"N": 20.0, "P": 40.0, "K": 20.0, "water_demand": "Low", "soil_pref": ["Loamy", "Black"]},
    "chickpea": {"N": 25.0, "P": 50.0, "K": 25.0, "water_demand": "Low", "soil_pref": ["Loamy", "Sandy"]},
    "coconut": {"N": 100.0, "P": 50.0, "K": 150.0, "water_demand": "High", "soil_pref": ["Sandy", "Red"]},
    "coffee": {"N": 100.0, "P": 50.0, "K": 100.0, "water_demand": "High", "soil_pref": ["Red", "Loamy"]},
    "grapes": {"N": 80.0, "P": 50.0, "K": 120.0, "water_demand": "Moderate", "soil_pref": ["Sandy", "Loamy"]},
    "jute": {"N": 80.0, "P": 40.0, "K": 40.0, "water_demand": "High", "soil_pref": ["Alluvial", "Clayey"]},
    "kidneybeans": {"N": 30.0, "P": 60.0, "K": 30.0, "water_demand": "Moderate", "soil_pref": ["Loamy"]},
    "lentil": {"N": 20.0, "P": 40.0, "K": 20.0, "water_demand": "Low", "soil_pref": ["Loamy"]},
    "mango": {"N": 100.0, "P": 50.0, "K": 100.0, "water_demand": "Moderate", "soil_pref": ["Alluvial", "Red"]},
    "mothbeans": {"N": 20.0, "P": 40.0, "K": 20.0, "water_demand": "Low", "soil_pref": ["Sandy"]},
    "mungbean": {"N": 20.0, "P": 40.0, "K": 20.0, "water_demand": "Low", "soil_pref": ["Loamy"]},
    "muskmelon": {"N": 80.0, "P": 50.0, "K": 50.0, "water_demand": "Moderate", "soil_pref": ["Sandy", "Loamy"]},
    "orange": {"N": 100.0, "P": 40.0, "K": 80.0, "water_demand": "Moderate", "soil_pref": ["Loamy", "Red"]},
    "papaya": {"N": 120.0, "P": 60.0, "K": 100.0, "water_demand": "High", "soil_pref": ["Loamy"]},
    "pigeonpeas": {"N": 25.0, "P": 50.0, "K": 25.0, "water_demand": "Low", "soil_pref": ["Loamy", "Black"]},
    "pomegranate": {"N": 70.0, "P": 35.0, "K": 70.0, "water_demand": "Low", "soil_pref": ["Loamy", "Black"]},
    "watermelon": {"N": 80.0, "P": 50.0, "K": 60.0, "water_demand": "Moderate", "soil_pref": ["Sandy", "Loamy"]}
}

# Soil Nutrient Retention Availability Multiplier
SOIL_NUTRIENT_AVAILABILITY = {
    "Black": 0.90,    # High cation exchange capacity, high retention
    "Clayey": 0.85,   # Strong mineral binding, slow leaching
    "Alluvial": 0.80, # Balanced texture and moderate percolation
    "Loamy": 0.80,    # Ideal agricultural structure
    "Red": 0.70,      # Slightly acidic, lower phosphorus fixation
    "Sandy": 0.55     # High porosity, fast nutrient leaching
}

# CNN Model Architectural Parameters (Matching Synopsis Figure 4.1)
CNN_INPUT_SHAPE = (224, 224, 3)
SE_REDUCTION_RATIO = 16
NUM_DISEASE_CLASSES = 38
TARGET_INFERENCE_MS = 65.0
