# MyFarmWise Consultant: System Architecture & Technical Specifications

**Degree:** Bachelor of Technology in Computer Science & Engineering (AI/ML)  
**Institution:** JSS Academy of Technical Education, Noida  
**Supervision:** Mr. Harshit (Assistant Professor, Dept. of CSE-AIML)  
**Milestone Date:** 29 October 2026 (Week 6 Progress Review)

---

## 1. Architectural Overview

MyFarmWise Consultant addresses three pervasive failures of existing agricultural decision-support systems:
1. **Manual Parameter Friction:** Smallholders cannot measure soil $N$, $P$, $K$, and $pH$ without laboratory equipment.
2. **Lab-to-Field Domain Shift:** Laboratory leaf disease classifiers (e.g. on PlantVillage) experience massive degradation (falling from ~99% to 17–46%) when evaluated on uncurated, cluttered field images.
3. **Black-Box Opacity:** Classification models return raw class labels without visual evidence, impeding farmer trust and extension worker validation.

The platform resolves these through a **Four-Tier Client-Server Architecture** operating behind a single unified farmer interface.

```
┌────────────────────────────────────────────────────────────────────────┐
│               TIER 1: USER INTERFACE & CLIENT PORTAL                   │
│   • Streamlit Web / Mobile Dashboard (Green-and-White Farmer Theme)    │
│   • District Selection (Zero manual soil-testing entry required)       │
│   • Leaf Image Capture & Augmentation Preview Visualizer               │
│   • Explainable AI Heatmap & Decision Rationale Viewer                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST
┌───────────────────────────────────▼────────────────────────────────────┐
│      TIER 2: APPLICATION SERVICES & ORCHESTRATION GATEWAY              │
│   • Flask REST API Server (Port 5000) with CORS & Rate Limiting        │
│   • District Lookup Dispatcher (ICAR Soil Health Card / IMD Normals)   │
│   • Pipeline Orchestrator: District -> Crop -> Fertilizer Engine       │
│   • Image Preprocessing, Validation & Multipart File Gateway           │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │                                 │
┌──────────────────▼─────────────────┐ ┌─────────────▼───────────────────┐
│  TIER 3: ML COMPUTATIONAL CORE     │ │ TIER 4: PERSISTENCE & DATA      │
│  • Random Forest Crop Model        │ │  • SQLite District Normals Store │
│    (22 crops, 2,200 records)       │ │  • ICAR Soil Health Card Tables  │
│  • Multi-Nutrient Deficit Engine   │ │  • IMD Rainfall & Climate CSVs   │
│    (7 fertilizers, 5 soil types)   │ │  • Commercial Fertilizer Metadata│
│  • SE-Attention CNN (Figure 4.1)   │ │  • 38-Class Pathology Catalog    │
│    (3 SE Blocks, 224x224 RGB)      │ │  • Model Artifacts (.pkl, .h5)   │
│  • Explainable AI Engines:         │ └─────────────────────────────────┘
│    - Random Forest MDI Gini Weights│
│    - Grad-CAM on layer last_se_conv│
└────────────────────────────────────┘
```

---

## 2. Subsystem Specifications

### Module A — Farmer Dashboard
- **Framework:** Streamlit (Python 3.11)
- **Design Tokens:** Forest Green (`#1b5e20`), Leaf Green (`#2e7d32`), Pale Mint (`#f1f8e9`), Pure White (`#ffffff`).
- **Responsive Layout:** Adaptive sidebar navigation, metric cards, reactive drop-downs, and session-state caching.

### Module B — Automated District-Level Soil & Climate Lookup
- **Data Repositories:**
  - Soil Chemistry: ICAR Soil Health Card Cycle-II Aggregates.
  - Climate Normals: India Meteorological Department (IMD) 1981–2010 Normals (641 districts across 35 states/UTs).
- **Rating-to-Numeric Transformation Rules:**
  - Nitrogen ($N$, kg/ha): Low (<280) $\rightarrow$ 200.0; Medium (280–560) $\rightarrow$ 380.0; High (>560) $\rightarrow$ 600.0.
  - Phosphorus ($P_2O_5$, kg/ha): Low (<10) $\rightarrow$ 8.5; Medium (10–25) $\rightarrow$ 18.0; High (>25) $\rightarrow$ 32.0.
  - Potassium ($K_2O$, kg/ha): Low (<108) $\rightarrow$ 85.0; Medium (108–280) $\rightarrow$ 190.0; High (>280) $\rightarrow$ 320.0.
  - Soil $pH$: Acidic (<6.5) $\rightarrow$ 5.8; Neutral (6.5–7.5) $\rightarrow$ 6.8; Alkaline (>7.5) $\rightarrow$ 8.1.
- **Handling of Unverified Records:** Districts lacking laboratory chemical tests are explicitly labeled with `status: "Climate Verified; Soil Pending"` to prevent fabricating synthetic soil numbers.

### Module C — Crop Recommendation
- **Model:** Scikit-Learn `RandomForestClassifier` (100 estimators).
- **Features:** 7 dimensions $[N, P, K, \text{temp}, \text{humidity}, pH, \text{rainfall}]$.
- **Performance:** Measured Test Accuracy of **99.55%** and Macro F1 of **99.55%** on stratified 80-20 test split (2,200 rows, 22 balanced crop species).
- **Sensitivity:** Perturbed district estimate accuracy of **91.14%**, confirming that district-wide averages serve as robust preliminary estimates.

### Module D — Fertilizer Recommendation Engine
- **Nutrient Deficit Formulation:**
  $$\Delta N = \max\left(0, N_{\text{target}} - N_{\text{soil}} \cdot \eta_{\text{soil}} \cdot 0.35\right)$$
  $$\Delta P = \max\left(0, P_{\text{target}} - P_{\text{soil}} \cdot \eta_{\text{soil}} \cdot 0.40\right)$$
  $$\Delta K = \max\left(0, K_{\text{target}} - K_{\text{soil}} \cdot \eta_{\text{soil}} \cdot 0.30\right)$$
  Where $\eta_{\text{soil}}$ represents the soil texture retention factor (Black: 0.90, Clayey: 0.85, Loamy/Alluvial: 0.80, Red: 0.70, Sandy: 0.55).
- **Formulation Dispatch:** Matches the calculated deficit profile against 7 standardized commercial grades:
  - Urea (46-0-0)
  - DAP (18-46-0)
  - 14-35-14
  - 28-28-0 / 20-20-0
  - 17-17-17
  - 10-26-26
- **Indicative Application Dose:** Clamped to realistic agronomic limits ($50 - 350\text{ kg/ha}$) and reported in standard 50-kg commercial bags.

### Module E — Plant Disease Attention-CNN (Figure 4.1)
- **Input:** $224 \times 224 \times 3$ RGB leaf imagery.
- **Conv Stem:** 32 filters ($3 \times 3$), BatchNorm, ReLU, MaxPool ($2 \times 2$).
- **SE Identity Blocks 1–3:** 64, 128, and 256 filters with Squeeze-and-Excitation channel recalibration ($r=16$).
- **Squeeze & Excitation Mechanism:**
  $$\mathbf{z}_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j)$$
  $$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}\left(\mathbf{W}_1 \cdot \mathbf{z}\right)\right)$$
  $$\widetilde{\mathbf{x}}_c = \mathbf{s}_c \cdot \mathbf{u}_c + \mathbf{u}_c$$
- **Head:** Global Average Pooling ($256\text{-d}$) $\rightarrow$ Dense ($128$) $\rightarrow$ Dropout ($0.5$) $\rightarrow$ Softmax ($38\text{ classes}$).
- **Parameters:** 438,886 total parameters.
- **Crop Conditioning (Table 4.2):** User-selected or recommended crop restricts candidate disease classes, preventing biologically impossible predictions (e.g. citrus canker on a potato leaf).

### Module F — Explainable AI (XAI)
- **Crop Model:** Mean Decrease in Impurity (MDI / Gini Importance) and local agronomic rationale.
- **Disease Model:** Grad-CAM on layer `last_se_conv`:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{ij}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

---

## 3. REST API Specification

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health, subsystem audit & milestone date | `200 OK` |
| `GET` | `/api/districts/states` | List of 35 available states/UTs | `200 OK` |
| `GET` | `/api/districts?state=...` | List of districts for a state | `200 OK` |
| `GET` | `/api/districts/lookup?state=...&district=...` | Agro-climatic profile & soil chemistry | `200 OK` / `404` |
| `POST` | `/api/crop/recommend` | Random Forest crop prediction & XAI weights | `200 OK` / `400` |
| `POST` | `/api/fertilizer/recommend` | Multi-nutrient deficit fertilizer advisory | `200 OK` / `400` |
| `POST` | `/api/disease/predict` | Multipart leaf image upload & diagnosis | `200 OK` / `413` |
| `GET` | `/api/disease/architecture` | SE-CNN Figure 4.1 parameter summary | `200 OK` |
