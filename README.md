# 🌱 MyFarmWise Consultant: Region-Aware, Explainable AI for Crop, Fertilizer & Disease Advisory

**B.Tech CSE (AI/ML) Major Project**  
**Faculty Demonstration Date:** 29 October 2026 (Week 6 Progress Review)  
**Institution:** JSS Academy of Technical Education, Noida  
**Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
**Supervisor:** Mr. Harshit (Assistant Professor)  
**Student Team:**
- Aaditya (Roll No. 2300911530001)
- Devendra Gupta (Roll No. 2300911530041)
- Dhananjay Yadav (Roll No. 2300911530042)
- Dhruv Sharma (Roll No. 2300911530044)

---

## 📌 Project Overview

**MyFarmWise Consultant** is an agronomic decision-support platform designed to solve three critical limitations in current precision agriculture services:
1. **Manual Parameter Entry Friction:** Smallholder farmers cannot measure numeric soil nitrogen ($N$), phosphorus ($P$), potassium ($K$), and $pH$ without laboratory equipment. MyFarmWise integrates an **automated district lookup layer** sourcing ICAR Soil Health Card benchmarks and IMD climate normals across 641 districts in 35 Indian states and UTs.
2. **Lab-to-Field Pathology Domain Degradation:** Disease classifiers trained on curated laboratory leaves lose dramatic accuracy in field conditions. MyFarmWise designs an **Attention-CNN with Squeeze-and-Excitation (SE) identity blocks** (Figure 4.1) coupled with **field-realistic augmentations** (motion blur, shadows, illumination, background clutter) and **crop-conditioned candidate filtering**.
3. **Black-Box Opacity:** Farmers and agricultural extension workers distrust opaque class predictions. MyFarmWise provides **Explainable AI (XAI)** through **Random Forest Gini feature importance** for crop recommendations and **Grad-CAM visual lesion heatmaps** on leaf imagery.

---

## 🛠️ Technology Stack

- **Language:** Python 3.11 (Python 3.9+ compatible)
- **Frontend / Dashboard:** Streamlit (Custom responsive green-and-white agricultural theme)
- **Backend API Gateway:** Flask REST API with CORS, rate limiting, and structured error responses
- **Machine Learning Core:** Scikit-Learn, NumPy, Pandas, Joblib
- **Deep Learning & Computer Vision:** TensorFlow 2.x / Keras 3.x, OpenCV Headless, Pillow
- **Explainable AI (XAI):** Matplotlib, Seaborn, Grad-CAM on layer `last_se_conv`
- **Data Persistence:** SQLite database (`district_lookup.sqlite`), CSV, JSON

---

## 📁 Repository Structure

```
MyFarmWise/
│
├── app.py                     # Unified Streamlit Farmer Dashboard
├── requirements.txt           # Project package dependencies
├── README.md                  # Project documentation & run guide
├── .gitignore                 # Version control exclusions
│
├── config/
│   ├── settings.py            # Global paths, constants, thresholds & ICAR conversion rules
│   └── disease_classes.json   # Catalog of 38 disease classes spanning 14 crops
│
├── data/
│   ├── district_lookup.csv    # 641 districts with ICAR SHC and IMD normals
│   ├── district_lookup.sqlite # Cached SQLite database for rapid district queries
│   ├── Crop_recommendation.csv# 2,200 agro-climatic records across 22 crops
│   ├── fertilizer_data.csv    # 99 empirical fertilizer baseline records
│   ├── district_wise_rainfall_normal.csv # Official IMD district normals
│   └── sample_images/         # Sample leaf pathology imagery for offline demo
│
├── models/
│   ├── README.md              # Model architecture and evaluation documentation
│   ├── crop_rf_model.pkl      # Verified Random Forest crop model (99.55% test accuracy)
│   ├── RandomForest.pkl       # Original baseline model artifact
│   └── baseline_plant_disease_model_128x128.keras # Preliminary exploratory baseline CNN
│
├── src/
│   ├── data/
│   │   └── district_lookup.py # District lookup and ICAR rating conversion engine
│   ├── crop/
│   │   ├── predictor.py       # Random Forest inference, top candidates, bounds validation
│   │   └── evaluate.py        # Baseline evaluation, cross-validation, district sensitivity
│   ├── fertilizer/
│   │   └── recommender.py     # Multi-nutrient deficit calculator & 7 commercial grades
│   ├── disease/
│   │   ├── preprocessing.py   # 224x224 RGB image normalization & field augmentation
│   │   ├── model.py           # SE-Attention CNN architecture (Figure 4.1, 438K params)
│   │   ├── train.py           # Week 7 training routine (Adam, cosine decay, checkpoints)
│   │   ├── predict.py         # Crop-conditioned disease inference pipeline
│   │   └── gradcam.py         # Grad-CAM heatmap generator on last SE conv block
│   └── explainability/
│       └── feature_importance.py # MDI Gini importance bar charts & agronomic rationale
│
├── backend/
│   ├── app.py                 # Flask REST server with /api/health and rate limiting
│   └── routes/
│       ├── district_routes.py # /api/districts endpoints
│       ├── crop_routes.py     # /api/crop endpoints
│       ├── fertilizer_routes.py # /api/fertilizer endpoints
│       └── disease_routes.py  # /api/disease endpoints
│
├── tests/
│   ├── test_district_lookup.py # Unit tests for district queries and conversion rules
│   ├── test_crop.py           # Unit tests for crop prediction and test accuracy
│   ├── test_fertilizer.py     # Unit tests for deficit calculation and commercial grades
│   └── test_api.py            # Integration tests for Flask REST endpoints
│
├── notebooks/
│   └── disease_model_experiments.ipynb # Interactive CV & SE-CNN research notebook
│
├── docs/
│   ├── architecture.md        # Technical architecture, equations, and REST schema
│   └── progress_report.md     # Official progress audit as of 29 October 2026
│
└── static/                    # Figures from synopsis and logos
    ├── fig1_pipeline.png
    ├── fig3_architecture.png
    ├── fig4_cnn_architecture.png
    ├── fig6_gantt.png
    └── logo.jpg
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
Ensure **Python 3.11** (or compatible 3.9+) is installed.

### 2. Install Dependencies
Open PowerShell or Terminal inside the `MyFarmWise` directory:
```powershell
py -3.11 -m pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
Verify all 22 unit and integration tests:
```powershell
$env:PYTHONPATH = "."
py -3.11 -m unittest discover -s tests -v
```
*(All 22 tests should pass with `OK`)*

---

## 🖥️ Running the Application

### Option A: Launch the Farmer Dashboard (Streamlit)
```powershell
py -3.11 -m streamlit run app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`.*

### Option B: Launch the Flask REST API Gateway
```powershell
$env:PYTHONPATH = "."
py -3.11 backend/app.py
```
*The backend API will start on `http://127.0.0.1:5000`.*  
Verify the health check endpoint:
```powershell
curl http://127.0.0.1:5000/api/health
```

---

## 🎯 Faculty Demonstration Walkthrough (Step-by-Step)

For the **29 October 2026** project progress review:

1. **Open Dashboard (`http://localhost:8501`)**:
   - Present the **🏠 Home Dashboard**, showing the project objectives, paradigm comparison table, and the four-tier architectural diagram (Figure 1.1 & Figure 3.1).
2. **Demonstrate Automated District Lookup (Module B)**:
   - Navigate to **📍 District Soil & Climate Lookup**.
   - Select **GUJARAT** $\rightarrow$ **BHAVNAGAR** (specifically cited in the project synopsis survey) or **MAHARASHTRA** $\rightarrow$ **PUNE**.
   - Show how soil parameters ($N, P, K, pH$), annual/monsoon rainfall, temperature, and humidity are populated automatically from ICAR and IMD records without requiring the farmer to manually test soil.
3. **Demonstrate Crop Recommendation with Explainable AI (Module C & F)**:
   - Navigate to **🌾 Crop Recommendation**.
   - Observe that the active district parameters are pre-filled.
   - Click **Predict Optimal Crop 🌾**.
   - Point out the primary recommendation, the top-3 ranked alternatives, the confidence percentage, and the **Explainable AI Feature Importance chart** (showing how rainfall and humidity had the strongest discriminatory power).
4. **Demonstrate Nutrient-Deficit Fertilizer Advisory (Module D)**:
   - Navigate to **🧪 Fertilizer Advisory**.
   - Select the target crop (e.g. Wheat) and soil texture (Loamy).
   - Click **Generate Fertilizer Advisory 🧪**.
   - Show the calculated nutrient deficits ($\Delta N, \Delta P, \Delta K$), the prescribed commercial formulation (e.g. DAP for phosphorus deficit, Urea for vegetative growth, or 10-26-26 for balanced potash/phosphate), the indicative application dose per hectare, and the prominent agronomic disclaimer.
5. **Demonstrate Plant Pathology & SE-CNN Pipeline (Module E & F)**:
   - Navigate to **🍃 Leaf Disease Diagnosis**.
   - Select a sample leaf image (e.g. `TomatoEarlyBlight1.JPG` or `PotatoEarlyBlight1.JPG`).
   - Select **Tomato** as the crop-conditioning filter to show how the 38-class search space is narrowed to only Tomato pathologies.
   - Highlight the **Field-Realistic Augmentation Simulator** demonstrating the 4 transformations (motion blur, shadow casting, sunlight shift, and background clutter) designed to address lab-to-field domain shift.
   - Present the **Figure 4.1 SE-Attention CNN architecture** (438,886 parameters) with Squeeze-and-Excitation channel recalibration and the Grad-CAM hook on layer `last_se_conv`.
6. **Conclude with the Faculty Progress Review Page**:
   - Navigate to **📊 Faculty Progress Review**.
   - Review the 16-week Gantt chart (Figure 6.1), the Table 6.1 target criteria evaluation (99.55% crop accuracy vs 99.09% target), and the clear roadmap for **Week 7** (training the SE-CNN across 54,305 images).
