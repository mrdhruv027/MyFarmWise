# MyFarmWise Consultant: Major Project Progress Report
**Demonstration & Faculty Review Date:** 29 October 2026  
**Project Execution Window:** 18 September 2026 – January 2027 (16 Weeks Total)  
**Current Milestone Stage:** Week 6 of 16 (Architecture, Data Pipelines & Baseline Verification)  
**Institution:** JSS Academy of Technical Education, Noida  
**Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
**Supervisor:** Mr. Harshit (Assistant Professor)

---

## 1. Executive Summary

As of **29 October 2026**, the **MyFarmWise Consultant** major project has completed its preliminary six-week development cycle. Commencing on 18 September 2026, the team has implemented:
1. **Automated District Lookup Layer:** 641 districts across 35 Indian states and UTs ingested from official IMD normals and ICAR Soil Health Card Cycle-II benchmarks, eliminating manual farmer entry friction.
2. **Crop Recommendation Subsystem:** Verified baseline Random Forest model achieving **99.55% test accuracy** and **99.55% macro F1-score** across 22 crop classes, with integrated Explainable AI (MDI Gini feature importance) and localized agronomic rationale.
3. **Fertilizer Advisory Subsystem:** A multi-nutrient deficit recommendation engine covering 7 commercial fertilizer formulations across 5 soil textures, with indicative per-hectare dosing and clear disclaimers.
4. **Plant Pathology & Attention-CNN Subsystem:** Formulated the complete 38-class pathology catalog across 14 crops, implemented field-realistic image augmentation (motion blur, shadows, clutter), and constructed the Attention-CNN architecture with 3 Squeeze-and-Excitation (SE) identity blocks (Figure 4.1, 438,886 parameters).
5. **Unified Web Application & REST Gateway:** Polished green-and-white Streamlit dashboard and modular Flask REST API with rate limiting, input validation, and 22 automated unit tests passing with zero errors.

---

## 2. Status of Project Milestones (Week-by-Week Audit)

In accordance with departmental instructions, milestones are categorized into **Completed**, **Partially Completed**, **Scaffolded**, and **Pending**.

| Timeline | Synopsis Milestone | Status as of 29 Oct 2026 | Verified Code Deliverable / Artifact |
| :--- | :--- | :--- | :--- |
| **Week 1** | Scope & Literature Review | **Completed** | Synopsis, architecture blueprint, mKisan survey analysis. |
| **Week 2** | Soil & Climate Data Ingestion | **Completed** | `data/district_wise_rainfall_normal.csv`, ICAR SHC benchmarks. |
| **Week 3** | Automated Lookup Construction | **Completed** | `src/data/district_lookup.py`, SQLite & CSV stores (641 districts). |
| **Week 4** | Crop & Fertilizer Modules | **Completed** | `src/crop/predictor.py` (99.55%), `src/fertilizer/recommender.py`. |
| **Week 5** | Disease Data Prep & Augmentation | **Completed** | `config/disease_classes.json` (38 classes), `src/disease/preprocessing.py`. |
| **Week 6** | Attention-CNN Architecture Design | **Completed** | `src/disease/model.py` (3 SE blocks, Figure 4.1), unified UI. |
| **Week 7** | Deep Neural Network Training | **Pending (Next Week)** | `src/disease/train.py` verified; 54,305 image corpus training begins. |
| **Week 8** | Cross-Dataset Benchmarking | **Pending** | Cross-validation on PlantDoc and CD&S; crop conditioning evaluation. |
| **Week 9** | Grad-CAM Full Integration | **Scaffolded** | `src/disease/gradcam.py` gradient hook tied to layer `last_se_conv`. |
| **Week 10** | Backend REST API Development | **Completed (Early)** | `backend/app.py` & 4 REST blueprints (districts, crop, fert, disease). |
| **Week 11** | Unified Farmer Interface Design | **Completed (Early)** | `app.py` Streamlit dashboard with green-and-white farmer UX. |
| **Week 12** | End-to-End Pipeline Integration | **Partially Completed**| District $\rightarrow$ Crop $\rightarrow$ Fertilizer $\rightarrow$ Disease workflow tethered. |
| **Week 13** | Stress & Usability Testing | **Pending** | Server response latency tests under noisy field imagery. |
| **Week 14** | Extension Worker Field Trials | **Pending** | Advisory ratings by agricultural extension functionaries. |
| **Week 15** | System Optimization & Docker | **Scaffolded** | Target inference latency $\le 65\text{ ms}$; Dockerfile specification. |
| **Week 16** | Viva Defense & Final Thesis | **Pending** | Final major project thesis compilation and oral examination. |

---

## 3. Evaluation Plan Target Criteria Verification (Table 6.1)

| Subsystem Module | Synopsis Target Criterion | Measured Empirical Outcome | Compliance Status |
| :--- | :--- | :--- | :--- |
| **District Lookup** | Complete values for all districts of selected states; honest missing handling. | 641 districts across 35 states/UTs; unverified soil values clearly labeled "Pending ICAR Lab Ingestion". | **Target Achieved** |
| **Crop Recommendation**| Retain 99.09% on original split; report district inputs separately. | **Test Accuracy: 99.55%** (Stratified split, 2,200 rows).<br>District Perturbation Accuracy: **91.14%**. | **Target Exceeded** |
| **Fertilizer Engine** | Multi-nutrient deficit mapping across 7 fertilizers and 5 soil types. | 7 standard grades (Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, 10-26-26) with soil retention factors. | **Target Achieved** |
| **Disease Attention-CNN**| Architecture with SE identity blocks; bridge lab-to-field gap. | 38 classes, 3 SE blocks ($r=16$), 438,886 parameters. 4 field augmentations implemented. | **Stage Objective Met** |
| **Explainability (XAI)** | Feature importance for Random Forest; Grad-CAM on CNN. | Global MDI Gini bar charts & local rationale active; Grad-CAM gradient tape verified on `last_se_conv`. | **Stage Objective Met** |

---

## 4. Academic Integrity & Transparency Notice

In compliance with academic guidelines for B.Tech Major Project progress assessments:
- **No synthetic or fabricated results:** Test accuracies reported for the Random Forest model reflect actual execution over `Crop_recommendation.csv` using stratified train-test splits.
- **Clear separation of training stages:** The 224x224 Attention-CNN architecture has been fully coded, validated, and verified. Full deep model training over the multi-gigabyte PlantVillage/PlantDoc corpus is officially scheduled for **Week 7** and is not falsely represented as complete. An earlier 128x128 exploratory baseline model is retained solely for fallback demonstration.
- **Transparent district estimates:** The application explicitly notifies users whenever recommendations rely on district-level agro-climatic averages rather than direct certified soil tests.

---

## 5. Next Steps for Milestone Review 2 (Weeks 7–10)

1. **Week 7:** Mount the full 54,305 leaf image training corpus and train the SE-Attention CNN using Adam optimizer and learning rate decay over 50–100 epochs.
2. **Week 8:** Conduct cross-dataset evaluation between PlantVillage laboratory images and PlantDoc/CD&S field images, measuring accuracy retention under attention recalibration.
3. **Week 9:** Generate live Grad-CAM lesion overlays using the newly trained SE weights and compute mean IoU on annotated lesion bounding boxes.
4. **Week 10:** Finalize REST API stress testing and extension worker usability protocols.

---
**Report Compiled By:**  
Dhruv Sharma (Roll No. 2300911530044)  
Aaditya (Roll No. 2300911530001)  
Devendra Gupta (Roll No. 2300911530041)  
Dhananjay Yadav (Roll No. 2300911530042)  

**Approved for Departmental Review:**  
Mr. Harshit (Assistant Professor, Project Supervisor)
