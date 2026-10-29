# MyFarmWise Models Repository

Demonstration Audit: **29 October 2026** (Week 6 Progress Review)

This directory houses trained classical machine learning artifacts, deep learning architectural definitions, and checkpoint specifications for the MyFarmWise Consultant agronomic platform.

---

## 1. Crop Recommendation Model (`crop_rf_model.pkl` & `RandomForest.pkl`)

- **Algorithm:** Random Forest Classifier (`RandomForestClassifier`)
- **Trained Corpus:** 2,200 agro-climatic observations across 22 major crop species (100 samples/class balanced).
- **Features (7 dimensions):**
  1. `N`: Nitrogen (kg/ha)
  2. `P`: Phosphorus (kg/ha)
  3. `K`: Potassium (kg/ha)
  4. `temperature`: Mean ambient temperature (°C)
  5. `humidity`: Relative humidity (%)
  6. `ph`: Soil pH level
  7. `rainfall`: Annual/seasonal precipitation (mm)
- **Measured Evaluation Metrics (Verified Stratified 80-20 Split):**
  - **Test Accuracy:** `99.55%` (Baseline in Synopsis Table 6.1: `99.09%`)
  - **Macro Precision:** `99.57%`
  - **Macro Recall:** `99.55%`
  - **Macro F1-Score:** `99.55%`
  - **5-Fold Cross-Validation Accuracy:** `99.55% ± 0.32%`
  - **Perturbed District-Level Estimate Tolerance:** `91.14%`
- **Global Feature Importance (MDI / Gini Index):**
  - `Rainfall`: 23.02%
  - `Humidity`: 22.42%
  - `Potassium (K)`: 17.54%
  - `Phosphorus (P)`: 15.08%
  - `Nitrogen (N)`: 9.64%
  - `Temperature`: 7.24%
  - `Soil pH`: 5.06%

---

## 2. Attention-Based CNN with Squeeze-and-Excitation (SE) Blocks (Figure 4.1)

- **Input Dimension:** $224 \times 224 \times 3$ RGB leaf imagery
- **Target Classes:** 38 disease categories spanning 14 crops (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato).
- **Architecture Specification:**
  - **Stem:** `Conv2D(32, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool(2x2)`
  - **SE Identity Block 1:** `Conv2D(64, 3x3)` + `BatchNorm` + `ReLU` + Squeeze-and-Excitation Attention ($r=16$) + `MaxPool(2x2)`
  - **SE Identity Block 2:** `Conv2D(128, 3x3)` + `BatchNorm` + `ReLU` + Squeeze-and-Excitation Attention ($r=16$) + `MaxPool(2x2)`
  - **SE Identity Block 3:** `Conv2D(256, 3x3)` + `BatchNorm` + `ReLU` + Squeeze-and-Excitation Attention ($r=16$) — **Grad-CAM Target Hook**
  - **Global Average Pooling (GAP):** 256-dimensional feature representation
  - **Classification Head:** `Dense(128)` + `Dropout(0.5)` + `Softmax(38)`
- **Parameter Count:**
  - Total Parameters: **438,886**
  - Trainable Parameters: **437,990**
  - Non-trainable Parameters: **896**
- **Milestone Status as of 29 October 2026:**
  - **Architecture:** Completed & verified in TensorFlow/Keras 2.x/3.x.
  - **Training Execution:** Scheduled for **Week 7** (Adam optimizer, batch size 32, categorical cross-entropy, 50-100 epochs across 54,305 PlantVillage and PlantDoc images).
  - **Exploratory Baseline:** `baseline_plant_disease_model_128x128.keras` preserved from preliminary exploratory work for demonstration fallback.
