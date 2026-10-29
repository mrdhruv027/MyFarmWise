"""
MyFarmWise Consultant: Farmer Dashboard Application.
Major Project Demonstration as of 29 October 2026.
B.Tech CSE (AI/ML) Major Project.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

# Configure Root Paths
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    PROJECT_NAME,
    PROJECT_SUBTITLE,
    CURRENT_DEMO_DATE,
    DEVELOPMENT_WEEK,
    TOTAL_WEEKS,
    LOGO_PATH,
    HERO_IMAGE_PATH,
    STATIC_DIR,
    SAMPLE_IMAGES_DIR,
    SOIL_HEALTH_CARD_CONVERSION,
    COMMERCIAL_FERTILIZERS,
)
from src.data.district_lookup import get_all_states, get_districts_by_state, lookup_district
from src.crop.predictor import predict_crop, FEATURE_BOUNDS
from src.crop.evaluate import run_crop_evaluation
from src.fertilizer.recommender import recommend_fertilizer, SOIL_TYPES
from src.disease.preprocessing import load_and_preprocess_image, apply_field_realistic_augmentation
from src.disease.model import get_model_architecture_summary
from src.disease.predict import predict_disease, load_disease_catalog
from src.disease.gradcam import generate_gradcam_explanation
from src.explainability.feature_importance import (
    get_global_feature_importances,
    generate_feature_importance_plot,
    explain_local_crop_decision,
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="MyFarmWise Consultant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Agricultural Green-and-White Theme Styling
st.markdown(
    """
    <style>
        /* Base typography & color schemes */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #1c2d24;
        }
        
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }

        /* Top Brand Header */
        .brand-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1.4rem;
            background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(27, 94, 32, 0.15);
        }
        .brand-title {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            margin: 0;
            color: #ffffff;
        }
        .brand-sub {
            font-size: 0.88rem;
            opacity: 0.92;
            margin: 0;
            color: #e8f5e9;
        }
        .demo-pill {
            background-color: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 30px;
            padding: 0.35rem 0.9rem;
            font-size: 0.78rem;
            font-weight: 700;
            color: #ffffff;
            white-space: nowrap;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
            border: 1px solid #c8e6c9;
            border-radius: 18px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
        }
        .hero-banner h1 {
            color: #1b5e20;
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }
        .hero-banner p {
            color: #2e4a36;
            font-size: 1.05rem;
            line-height: 1.5;
            margin-bottom: 1.2rem;
        }

        /* Cards & Info boxes */
        .farm-card {
            background: #ffffff;
            border: 1px solid #e0ede2;
            border-radius: 14px;
            padding: 1.3rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            margin-bottom: 1rem;
            height: 100%;
        }
        .farm-card h3 {
            color: #1b5e20;
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 0.4rem;
        }
        .farm-card p {
            color: #496352;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 0.8rem;
        }

        /* Status tags */
        .status-tag {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.6rem;
        }
        .status-completed { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
        .status-progress { background: #fff8e1; color: #f57f17; border: 1px solid #ffe082; }
        .status-pending { background: #f5f5f5; color: #616161; border: 1px solid #e0e0e0; }

        /* Metric highlight box */
        .metric-box {
            background: #fbfdfb;
            border: 1px solid #dceede;
            border-radius: 10px;
            padding: 0.8rem;
            text-align: center;
        }
        .metric-val {
            font-size: 1.4rem;
            font-weight: 800;
            color: #1b5e20;
        }
        .metric-lbl {
            font-size: 0.75rem;
            color: #526e5a;
            font-weight: 600;
            text-transform: uppercase;
        }

        /* Disclaimer alerts */
        .disclaimer-box {
            background: #fffde7;
            border-left: 4px solid #fbc02d;
            padding: 0.8rem 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
            color: #615600;
            margin-top: 1rem;
        }

        /* Sidebar custom */
        [data-testid="stSidebar"] {
            background-color: #f7faf7;
            border-right: 1px solid #e2ece3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    f"""
    <div class="brand-header">
        <div>
            <div class="brand-title">🌱 {PROJECT_NAME}</div>
            <div class="brand-sub">{PROJECT_SUBTITLE}</div>
        </div>
        <div class="demo-pill">
            🗓️ Demo: {CURRENT_DEMO_DATE} | Week {DEVELOPMENT_WEEK} of {TOTAL_WEEKS}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Pages List
PAGES = [
    "🏠 Home Dashboard",
    "📍 District Soil & Climate Lookup",
    "🌾 Crop Recommendation",
    "🧪 Fertilizer Advisory",
    "🍃 Leaf Disease Diagnosis",
    "📊 Faculty Progress Review",
]

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]

# Sidebar Navigation
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=100)
    st.title("Navigation")
    menu = st.radio(
        "Select Subsystem Module:",
        PAGES,
        index=PAGES.index(st.session_state.current_page) if st.session_state.current_page in PAGES else 0,
        key="nav_radio"
    )
    st.session_state.current_page = menu
    st.markdown("---")
    st.markdown(
        f"""
        **Project Information**
        - **Institution:** JSS Academy of Technical Education, Noida
        - **Degree:** B.Tech CSE (AI/ML)
        - **Demonstration:** {CURRENT_DEMO_DATE}
        - **Supervisor:** Mr. Harshit (Asst. Prof.)
        - **Status:** Week {DEVELOPMENT_WEEK} (Architecture & Baselines)
        """
    )
    st.markdown("---")
    st.caption("MyFarmWise Consultant © 2026. Academic Research System.")


# Initialize session state for cross-module data flow
if "selected_state" not in st.session_state:
    st.session_state.selected_state = "MAHARASHTRA"
if "selected_district" not in st.session_state:
    st.session_state.selected_district = "PUNE"
if "district_profile" not in st.session_state:
    st.session_state.district_profile = lookup_district("MAHARASHTRA", "PUNE")
if "recommended_crop_result" not in st.session_state:
    st.session_state.recommended_crop_result = None


# ==============================================================================
# 1. HOME DASHBOARD
# ==============================================================================
if menu == "🏠 Home Dashboard":
    col_left, col_right = st.columns([1.3, 0.7])
    with col_left:
        st.markdown(
            f"""
            <div class="hero-banner">
                <h1>Smart Agronomic Advisories Backed by Explainable AI</h1>
                <p>
                    <b>MyFarmWise Consultant</b> is a four-tier decision-support platform designed for Indian smallholders.
                    It removes manual soil-testing friction by automatically sourcing district-level <b>ICAR Soil Health Card</b> 
                    and <b>IMD climate records</b>, runs high-accuracy Random Forest crop recommendation with <b>Explainable AI feature importance</b>,
                    calculates <b>multi-nutrient fertilizer deficits</b> across 7 commercial grades, and prepares an 
                    <b>Attention-CNN with Squeeze-and-Excitation (SE) blocks</b> for plant disease diagnosis with Grad-CAM visual lesion heatmaps.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        if HERO_IMAGE_PATH.exists():
            st.image(str(HERO_IMAGE_PATH), caption="Field Precision Agricultural Deployment", use_container_width=True)

    # Core Features Matrix
    st.subheader("Subsystem Architecture & Current Stage Status")
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)

    with fcol1:
        st.markdown(
            """
            <div class="farm-card">
                <span class="status-tag status-completed">Completed</span>
                <h3>Module B: District Lookup</h3>
                <p>Automated retrieval of N-P-K, pH, and IMD rainfall across 641 districts in 35 states/UTs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Lookup 📍", key="btn_home_lookup", use_container_width=True):
            st.session_state.current_page = "📍 District Soil & Climate Lookup"
            st.rerun()

    with fcol2:
        st.markdown(
            """
            <div class="farm-card">
                <span class="status-tag status-completed">Completed</span>
                <h3>Module C: Crop Predictor</h3>
                <p>Random Forest model (22 crops, 2,200 rows) achieving 99.55% test accuracy with XAI feature weights.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Run Crop Model 🌾", key="btn_home_crop", use_container_width=True):
            st.session_state.current_page = "🌾 Crop Recommendation"
            st.rerun()

    with fcol3:
        st.markdown(
            """
            <div class="farm-card">
                <span class="status-tag status-completed">Completed</span>
                <h3>Module D: Fertilizer Advisory</h3>
                <p>Nutrient deficit calculator mapping soil reserves to 7 commercial fertilizers across 5 soil textures.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Calculate Dosage 🧪", key="btn_home_fert", use_container_width=True):
            st.session_state.current_page = "🧪 Fertilizer Advisory"
            st.rerun()

    with fcol4:
        st.markdown(
            """
            <div class="farm-card">
                <span class="status-tag status-progress">Week 6 Design Complete</span>
                <h3>Module E: SE-Attention CNN</h3>
                <p>38-class CNN with 3 Squeeze-and-Excitation blocks, field augmentations, and Grad-CAM hooks.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Disease Pipeline 🍃", key="btn_home_disease", use_container_width=True):
            st.session_state.current_page = "🍃 Leaf Disease Diagnosis"
            st.rerun()

    st.markdown("---")

    # Architecture Diagram View
    st.subheader("System Architecture & Data Flow")
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        fig1_path = STATIC_DIR / "fig1_pipeline.png"
        if fig1_path.exists():
            st.image(str(fig1_path), caption="Figure 1.1: End-to-End Operational Pipeline", use_container_width=True)
    with arch_col2:
        fig3_path = STATIC_DIR / "fig3_architecture.png"
        if fig3_path.exists():
            st.image(str(fig3_path), caption="Figure 3.1: Four-Tier Modular Architecture", use_container_width=True)

    # Student Team Members
    st.markdown("---")
    st.subheader("Project Development Team")
    team = [
        {"name": "Aaditya", "roll": "2300911530001", "role": "Computer Vision & Deep Learning"},
        {"name": "Devendra Gupta", "roll": "2300911530041", "role": "Backend API & Data Architecture"},
        {"name": "Dhananjay Yadav", "roll": "2300911530042", "role": "Agronomic Models & Optimization"},
        {"name": "Dhruv Sharma", "roll": "2300911530044", "role": "System Integration, XAI & Frontend"},
    ]
    tcols = st.columns(4)
    for i, member in enumerate(team):
        with tcols[i]:
            st.markdown(
                f"""
                <div class="farm-card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.3rem;">👨‍💻</div>
                    <h4 style="margin: 0; color: #1b5e20;">{member['name']}</h4>
                    <p style="margin: 0.2rem 0; font-size: 0.8rem; color: #555;">Roll: {member['roll']}</p>
                    <span class="status-tag status-completed">{member['role']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 2. DISTRICT SOIL & CLIMATE LOOKUP
# ==============================================================================
elif menu == "📍 District Soil & Climate Lookup":
    st.header("📍 District-Level Soil & Climate Lookup Subsystem")
    st.write(
        "Sources official agro-climatic normals from the **India Meteorological Department (IMD)** and "
        "soil chemistry benchmarks from the **ICAR Soil Health Card Portal**. "
        "Allows smallholders to receive localized guidance without requiring private soil-testing lab kits."
    )

    states = get_all_states()
    col1, col2 = st.columns(2)
    with col1:
        default_state_idx = states.index(st.session_state.selected_state) if st.session_state.selected_state in states else 0
        selected_state = st.selectbox("Select State / Union Territory:", states, index=default_state_idx)
        st.session_state.selected_state = selected_state

    districts = get_districts_by_state(selected_state)
    with col2:
        default_dist_idx = (
            districts.index(st.session_state.selected_district)
            if st.session_state.selected_district in districts
            else 0
        )
        selected_district = st.selectbox("Select District:", districts, index=default_dist_idx)
        st.session_state.selected_district = selected_district

    # Fetch lookup profile
    profile = lookup_district(selected_state, selected_district)
    st.session_state.district_profile = profile

    st.markdown("---")

    if profile.get("found"):
        meta = profile["metadata"]
        st.markdown(
            f"""
            ### Agro-Climatic Profile: **{profile['district']}, {profile['state']}**
            <span class="status-tag status-completed">Data Status: {meta['status']}</span>
            """,
            unsafe_allow_html=True,
        )

        # Overview Metrics Row
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        clim = profile["climate_normals"]
        soil = profile["soil_chemistry"]

        with mcol1:
            st.markdown(
                f"""<div class="metric-box"><div class="metric-val">{profile['soil_type']}</div><div class="metric-lbl">Dominant Soil</div></div>""",
                unsafe_allow_html=True,
            )
        with mcol2:
            st.markdown(
                f"""<div class="metric-box"><div class="metric-val">{clim['annual_rainfall_mm'] or 'N/A'} mm</div><div class="metric-lbl">Annual Rainfall</div></div>""",
                unsafe_allow_html=True,
            )
        with mcol3:
            st.markdown(
                f"""<div class="metric-box"><div class="metric-val">{clim['mean_temperature_c'] or 'N/A'} °C</div><div class="metric-lbl">Mean Temp</div></div>""",
                unsafe_allow_html=True,
            )
        with mcol4:
            st.markdown(
                f"""<div class="metric-box"><div class="metric-val">{clim['mean_humidity_pct'] or 'N/A'}%</div><div class="metric-lbl">Humidity</div></div>""",
                unsafe_allow_html=True,
            )
        with mcol5:
            st.markdown(
                f"""<div class="metric-box"><div class="metric-val">{soil['ph']['value'] or 'N/A'}</div><div class="metric-lbl">Soil pH</div></div>""",
                unsafe_allow_html=True,
            )

        # Soil Chemistry Breakdown Table
        st.markdown("#### ICAR Soil Health Card Chemistry Parameters")
        sc1, sc2, sc3, sc4 = st.columns(4)

        with sc1:
            n_data = soil["nitrogen"]
            st.markdown(
                f"""
                <div class="farm-card">
                    <h4>Nitrogen (N)</h4>
                    <p style="font-size: 1.5rem; font-weight: 800; color: #1b5e20; margin: 0;">
                        {n_data['value'] or 'N/A'} <span style="font-size: 0.9rem;">{n_data['unit']}</span>
                    </p>
                    <p><b>Rating:</b> {n_data['rating']}</p>
                    <p style="font-size: 0.75rem; color: #666;">Threshold: {n_data['benchmarks']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sc2:
            p_data = soil["phosphorus"]
            st.markdown(
                f"""
                <div class="farm-card">
                    <h4>Phosphorus (P)</h4>
                    <p style="font-size: 1.5rem; font-weight: 800; color: #1b5e20; margin: 0;">
                        {p_data['value'] or 'N/A'} <span style="font-size: 0.9rem;">{p_data['unit']}</span>
                    </p>
                    <p><b>Rating:</b> {p_data['rating']}</p>
                    <p style="font-size: 0.75rem; color: #666;">Threshold: {p_data['benchmarks']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sc3:
            k_data = soil["potassium"]
            st.markdown(
                f"""
                <div class="farm-card">
                    <h4>Potassium (K)</h4>
                    <p style="font-size: 1.5rem; font-weight: 800; color: #1b5e20; margin: 0;">
                        {k_data['value'] or 'N/A'} <span style="font-size: 0.9rem;">{k_data['unit']}</span>
                    </p>
                    <p><b>Rating:</b> {k_data['rating']}</p>
                    <p style="font-size: 0.75rem; color: #666;">Threshold: {k_data['benchmarks']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sc4:
            ph_data = soil["ph"]
            st.markdown(
                f"""
                <div class="farm-card">
                    <h4>Soil Acidity (pH)</h4>
                    <p style="font-size: 1.5rem; font-weight: 800; color: #1b5e20; margin: 0;">
                        {ph_data['value'] or 'N/A'} <span style="font-size: 0.9rem;">{ph_data['unit']}</span>
                    </p>
                    <p><b>Category:</b> {ph_data['category']}</p>
                    <p style="font-size: 0.75rem; color: #666;">Threshold: {ph_data['benchmarks']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Transparency & Limitations Card
        st.markdown(
            f"""
            <div class="disclaimer-box">
                <b>📌 Data Provenance & Limitations:</b><br>
                - <b>Soil Source:</b> {meta['source_soil']}<br>
                - <b>Climate Source:</b> {meta['source_climate']}<br>
                - <b>Limitation Note:</b> {meta['limitations']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info("💡 Tip: You can proceed directly to the Crop Recommendation page to auto-fill these district values.")


# ==============================================================================
# 3. CROP RECOMMENDATION
# ==============================================================================
elif menu == "🌾 Crop Recommendation":
    st.header("🌾 Machine Learning Crop Recommendation (Random Forest)")
    st.write(
        "Trained on 2,200 agro-climatic observations across 22 balanced crop classes. "
        "Provides top crop recommendation, probabilistic alternatives, and Explainable AI feature importance."
    )

    prof = st.session_state.district_profile
    has_district = prof and prof.get("found")

    # Option to autofill from district
    load_col1, load_col2 = st.columns([1.2, 0.8])
    with load_col1:
        if has_district:
            st.success(
                f"Active District Context: **{prof['district']}, {prof['state']}** "
                f"({prof['soil_type']} Soil)"
            )
    with load_col2:
        use_district_vals = st.checkbox("Auto-populate inputs from active district profile", value=True)

    # Resolve default values
    if use_district_vals and has_district:
        s_chem = prof["soil_chemistry"]
        c_norm = prof["climate_normals"]
        def_n = s_chem["nitrogen"]["value"] if s_chem["nitrogen"]["value"] else 60.0
        # Scaled down to typical dataset bounds if large bulk values
        def_n = min(float(def_n), 140.0)
        def_p = float(s_chem["phosphorus"]["value"]) if s_chem["phosphorus"]["value"] else 40.0
        def_k = min(float(s_chem["potassium"]["value"]), 200.0) if s_chem["potassium"]["value"] else 40.0
        def_temp = float(c_norm["mean_temperature_c"]) if c_norm["mean_temperature_c"] else 25.0
        def_hum = float(c_norm["mean_humidity_pct"]) if c_norm["mean_humidity_pct"] else 65.0
        def_ph = float(s_chem["ph"]["value"]) if s_chem["ph"]["value"] else 6.8
        # Use monsoon rainfall or clipped annual
        def_rain = min(float(c_norm["monsoon_rainfall_mm"] or 150.0), 280.0)
        is_district_flag = True
    else:
        def_n, def_p, def_k = 90.0, 42.0, 43.0
        def_temp, def_hum, def_ph, def_rain = 21.0, 82.0, 6.5, 205.0
        is_district_flag = False

    # Input sliders/fields
    st.subheader("Agro-Climatic Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_input = st.number_input("Nitrogen (N, kg/ha)", 0.0, 140.0, float(def_n), step=1.0)
        temp_input = st.number_input("Temperature (°C)", 8.0, 45.0, float(def_temp), step=0.5)
        ph_input = st.number_input("Soil pH", 3.5, 9.9, float(def_ph), step=0.1)
    with c2:
        p_input = st.number_input("Phosphorus (P, kg/ha)", 5.0, 145.0, float(def_p), step=1.0)
        hum_input = st.number_input("Relative Humidity (%)", 14.0, 100.0, float(def_hum), step=1.0)
    with c3:
        k_input = st.number_input("Potassium (K, kg/ha)", 5.0, 205.0, float(def_k), step=1.0)
        rain_input = st.number_input("Rainfall (mm)", 20.0, 300.0, float(def_rain), step=1.0)

    if st.button("Predict Optimal Crop 🌾", type="primary", use_container_width=True):
        with st.spinner("Executing Random Forest inference pipeline..."):
            dist_context = {
                "state": prof["state"] if has_district else "Manual Entry",
                "district": prof["district"] if has_district else "Manual Entry",
            }
            result = predict_crop(
                nitrogen=n_input,
                phosphorus=p_input,
                potassium=k_input,
                temperature=temp_input,
                humidity=hum_input,
                ph=ph_input,
                rainfall=rain_input,
                is_district_estimate=is_district_flag,
                district_info=dist_context,
            )
            st.session_state.recommended_crop_result = result

    # Display Prediction Results
    if st.session_state.recommended_crop_result:
        res = st.session_state.recommended_crop_result
        st.markdown("---")
        st.subheader("Recommendation Diagnostic Summary")

        rcol1, rcol2 = st.columns([1.1, 0.9])
        with rcol1:
            st.markdown(
                f"""
                <div class="farm-card" style="border: 2px solid #2e7d32; background: #f4faf5;">
                    <span class="status-tag status-completed">Primary Advisory</span>
                    <h2 style="color: #1b5e20; margin: 0.3rem 0; font-size: 2.2rem; text-transform: capitalize;">
                        🌾 {res['recommended_crop']}
                    </h2>
                    <p style="font-size: 1.1rem; color: #2e7d32; font-weight: 700;">
                        Confidence: {res['confidence_pct']}%
                    </p>
                    <p>{res['disclaimer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top candidate alternatives
            st.markdown("#### Top Ranked Crop Candidates")
            for idx, cand in enumerate(res["top_candidates"]):
                st.write(
                    f"**{idx + 1}. {cand['crop'].capitalize()}** — "
                    f"Confidence: `{cand['confidence_pct']}%` | "
                    f"Water Demand: `{cand['nutrient_profile'].get('water_demand', 'Moderate')}`"
                )

        with rcol2:
            st.markdown("#### Explainable AI: Feature Importance")
            st.write("Mean Decrease in Impurity (Gini Importance) across the 7 input dimensions:")
            feat_img = generate_feature_importance_plot()
            st.image(feat_img, use_container_width=True)

        # Local Decision Explanation
        st.markdown("#### Agronomic Rationale for This Decision")
        local_pts = explain_local_crop_decision(res["recommended_crop"], res["input_features"])
        for pt in local_pts:
            st.markdown(f"- **{pt['feature']} ({pt['observation']}):** {pt['impact']}")


# ==============================================================================
# 4. FERTILIZER ADVISORY
# ==============================================================================
elif menu == "🧪 Fertilizer Advisory":
    st.header("🧪 Nutrient-Deficit Fertilizer Recommendation Engine (Module D)")
    st.write(
        "Maps crop uptake demands against available soil reserves, computes multi-nutrient deficits, "
        "and prescribes optimal commercial formulations from 7 standardized grades across 5 soil textures."
    )

    prof = st.session_state.district_profile
    has_district = prof and prof.get("found")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        crop_options = [
            "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Millets",
            "Pulses", "Ground Nuts", "Barley", "Oil Seeds", "Tobacco",
            "Apple", "Banana", "Grapes", "Orange", "Papaya", "Watermelon"
        ]
        # Pre-select recommended crop if available
        def_crop_idx = 0
        if st.session_state.recommended_crop_result:
            rec_c = st.session_state.recommended_crop_result["recommended_crop"].capitalize()
            if rec_c in crop_options:
                def_crop_idx = crop_options.index(rec_c)

        selected_crop = st.selectbox("Target Cultivated Crop:", crop_options, index=def_crop_idx)

        default_soil_type = prof.get("soil_type", "Loamy") if has_district else "Loamy"
        if default_soil_type not in SOIL_TYPES:
            default_soil_type = "Loamy"
        soil_type_input = st.selectbox("Dominant Soil Texture:", SOIL_TYPES, index=SOIL_TYPES.index(default_soil_type))

    with fcol2:
        def_fn = float(prof["soil_chemistry"]["nitrogen"]["value"] or 200.0) if has_district else 200.0
        def_fp = float(prof["soil_chemistry"]["phosphorus"]["value"] or 15.0) if has_district else 15.0
        def_fk = float(prof["soil_chemistry"]["potassium"]["value"] or 180.0) if has_district else 180.0

        fert_n = st.number_input("Soil Available Nitrogen (N, kg/ha)", 0.0, 500.0, float(def_fn), step=5.0)
        fert_p = st.number_input("Soil Available Phosphorus (P, kg/ha)", 0.0, 150.0, float(def_fp), step=1.0)
        fert_k = st.number_input("Soil Available Potassium (K, kg/ha)", 0.0, 500.0, float(def_fk), step=5.0)

    if st.button("Generate Fertilizer Advisory 🧪", type="primary", use_container_width=True):
        advisory = recommend_fertilizer(
            crop_name=selected_crop,
            soil_n=fert_n,
            soil_p=fert_p,
            soil_k=fert_k,
            soil_type=soil_type_input,
            is_district_estimate=has_district,
        )

        st.markdown("---")
        st.subheader("Prescribed Fertilizer Formulation & Application Schedule")

        rcol1, rcol2 = st.columns([1.1, 0.9])
        with rcol1:
            st.markdown(
                f"""
                <div class="farm-card" style="border: 2px solid #1b5e20; background: #fbfefb;">
                    <span class="status-tag status-completed">Prescribed Commercial Grade</span>
                    <h2 style="color: #1b5e20; margin: 0.3rem 0; font-size: 2.1rem;">
                        🧪 {advisory['recommended_fertilizer']}
                    </h2>
                    <p style="font-weight: 700; color: #2e7d32; font-size: 1.1rem;">
                        Indicative Dosage: {advisory['indicative_quantity_kg_ha']} kg / hectare 
                        (~{advisory['indicative_quantity_bags_ha']} bags of 50kg)
                    </p>
                    <p><b>Primary Deficit Addressed:</b> {advisory['primary_nutrient_addressed']}</p>
                    <p><b>Active Formula:</b> {advisory.get('fertilizer_composition', 'Standard Grade')}</p>
                    <p>{advisory['rationale']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Application Timing")
            st.info(advisory["application_schedule"])

        with rcol2:
            st.markdown("#### Calculated Nutrient Deficits (kg/ha)")
            defs = advisory["nutrient_deficits_kg_ha"]
            st.markdown(
                f"""
                <div class="farm-card">
                    <p><b>Nitrogen Deficit (ΔN):</b> {defs['N']} kg/ha</p>
                    <p><b>Phosphorus Deficit (ΔP):</b> {defs['P']} kg/ha</p>
                    <p><b>Potassium Deficit (ΔK):</b> {defs['K']} kg/ha</p>
                    <p><b>Soil Retention Factor:</b> {advisory.get('soil_retention_multiplier', 0.8)*100}% ({advisory['soil_type_considered']} soil)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="disclaimer-box">
                {advisory['disclaimer']}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 5. PLANT LEAF DISEASE DIAGNOSIS
# ==============================================================================
elif menu == "🍃 Leaf Disease Diagnosis":
    st.header("🍃 Plant Leaf Pathology Diagnosis & Explainable AI (Grad-CAM)")
    st.write(
        "Implements the 38-class plant pathology system specified in Chapter 4 (Figure 4.1). "
        "Features field-realistic augmentation previews, crop-conditioned candidate filtering, "
        "and architectural integration for the Attention-CNN with Squeeze-and-Excitation (SE) blocks."
    )

    dcol1, dcol2 = st.columns([1.1, 0.9])
    uploaded_file = None

    with dcol1:
        st.subheader("1. Leaf Imagery Upload")
        upload_mode = st.radio("Input Source:", ["Choose from Sample Dataset", "Upload Leaf Photo"], horizontal=True)

        sample_img_path = None
        if upload_mode == "Choose from Sample Dataset":
            sample_files = list(SAMPLE_IMAGES_DIR.glob("*.JPG")) if SAMPLE_IMAGES_DIR.exists() else []
            if sample_files:
                sample_names = [f.name for f in sample_files]
                chosen_sample = st.selectbox("Select Sample Field Image:", sample_names)
                sample_img_path = SAMPLE_IMAGES_DIR / chosen_sample
                uploaded_file = Image.open(sample_img_path)
            else:
                st.warning("No sample images found in data/sample_images.")
        else:
            file_buff = st.file_uploader("Upload Leaf Image (JPEG / PNG):", type=["jpg", "jpeg", "png"])
            if file_buff:
                uploaded_file = Image.open(file_buff)

        if uploaded_file:
            st.image(uploaded_file, caption="Input Leaf Image", width=340)

    with dcol2:
        st.subheader("2. Diagnostic Controls")
        catalog = load_disease_catalog()
        crops_available = ["None (All 14 Crops)"] + sorted(list(set(c["crop"] for c in catalog.values())))
        selected_crop_filter = st.selectbox("Apply Crop Conditioning Filter (Table 4.2):", crops_available)
        effective_filter = None if selected_crop_filter.startswith("None") else selected_crop_filter

        if effective_filter:
            st.caption(f"🛡️ Candidate classes restricted to **{effective_filter}** pathologies only.")

        run_diag = st.button("Run Diagnostic Inference 🍃", type="primary", use_container_width=True)

    if uploaded_file and run_diag:
        st.markdown("---")
        st.subheader("Diagnostic Results & Model Inspection")

        with st.spinner("Preprocessing image and executing neural network inference..."):
            diag_res = predict_disease(
                uploaded_file,
                crop_filter=effective_filter,
                use_baseline_fallback=True
            )

        pcol1, pcol2 = st.columns([1.1, 0.9])
        with pcol1:
            if diag_res.get("success"):
                prim = diag_res["primary_diagnosis"]
                is_healthy = prim["is_healthy"]
                badge_class = "status-completed" if is_healthy else "status-progress"
                st.markdown(
                    f"""
                    <div class="farm-card" style="border: 2px solid {'#2e7d32' if is_healthy else '#d32f2f'};">
                        <span class="status-tag {badge_class}">
                            {'Healthy Foliage' if is_healthy else 'Pathology Detected'}
                        </span>
                        <h2 style="color: {'#1b5e20' if is_healthy else '#c62828'}; margin: 0.3rem 0;">
                            {prim['display_name']}
                        </h2>
                        <p style="font-size: 1.1rem; font-weight: 700; color: #333;">
                            Confidence Score: {prim['confidence_pct']}%
                        </p>
                        <p><b>Model Engine:</b> {diag_res['model_version']}</p>
                        <p><b>Inference Latency:</b> {diag_res['inference_time_ms']} ms ({'Target <65ms Achieved' if diag_res['target_time_met'] else 'Exceeds target'})</p>
                        <p style="font-size: 0.85rem; color: #666;">{diag_res.get('note', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Candidate Top Predictions")
                for cand in diag_res["top_candidates"]:
                    st.write(f"- **{cand['display_name']}**: `{cand['confidence_pct']}%` (Severity Risk: `{cand['severity_risk']}`)")

            else:
                st.warning(f"**Diagnostic Pipeline Status:** {diag_res.get('weights_status')}")
                st.info(diag_res.get("message"))

        with pcol2:
            st.markdown("#### Explainable AI: Grad-CAM Heatmap")
            grad_res = generate_gradcam_explanation(uploaded_file)
            if grad_res.get("available"):
                st.image(grad_res["overlay_image"], caption="Grad-CAM Lesion Heatmap Overlay", use_container_width=True)
            else:
                st.info(f"**Grad-CAM Engine Status:** {grad_res['status']}")
                st.caption(
                    "Grad-CAM gradient hook is tied to layer `last_se_conv` of the Squeeze-and-Excitation CNN. "
                    "Full visual overlays activate upon completion of Week 7 model training."
                )
                st.latex(r"L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)")

    # Field-Realistic Augmentation Visualizer
    if uploaded_file:
        st.markdown("---")
        st.subheader("Field-Realistic Image Augmentation Simulator (Week 5 Benchmark)")
        st.write("Simulates harsh outdoor field distortions: hand motion blur, sunlight shadows, and noise clutter.")
        aug_dict = apply_field_realistic_augmentation(uploaded_file)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.image(aug_dict["motion_blur"], caption="1. Motion Blur (Hand Tremor)", use_container_width=True)
        with a2:
            st.image(aug_dict["shadow_casting"], caption="2. Canopy Shadow Casting", use_container_width=True)
        with a3:
            st.image(aug_dict["illumination_shift"], caption="3. Sunlight Shift", use_container_width=True)
        with a4:
            st.image(aug_dict["background_clutter"], caption="4. Background Clutter", use_container_width=True)

    # SE-CNN Architectural Blueprint
    st.markdown("---")
    st.subheader("Attention-CNN Architectural Specification (Figure 4.1)")
    se_summary = get_model_architecture_summary()
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        st.metric("Total Parameters", f"{se_summary['total_parameters']:,}")
    with scol2:
        st.metric("SE Identity Blocks", f"{se_summary['se_blocks_count']} Blocks")
    with scol3:
        st.metric("Grad-CAM Hook Layer", se_summary["gradcam_target_layer"])

    fig4_path = STATIC_DIR / "fig4_cnn_architecture.png"
    if fig4_path.exists():
        st.image(str(fig4_path), caption="Figure 4.1: Attention-Based CNN with Squeeze-and-Excitation (SE) Identity Blocks", use_container_width=True)


# ==============================================================================
# 6. FACULTY PROGRESS REVIEW
# ==============================================================================
elif menu == "📊 Faculty Progress Review":
    st.header("📊 Faculty Demonstration & Milestone Progress Review")
    st.write(
        f"**Official Progress Audit as of {CURRENT_DEMO_DATE}** | Project Commenced: 18 September 2026. "
        "Strictly reflects verified, testable code deliverables in compliance with academic integrity guidelines."
    )

    # High-level Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current Timeline", f"Week {DEVELOPMENT_WEEK} of {TOTAL_WEEKS}")
    with m2:
        st.metric("Crop Model Test Accuracy", "99.55%", "Verified Stratified Split")
    with m3:
        st.metric("Districts Ingested", "641 Districts", "35 States & UTs")
    with m4:
        st.metric("SE-CNN Architecture", "438,886 Params", "Week 6 Complete")

    st.markdown("---")

    # Detailed Week-by-Week Milestone Matrix
    st.subheader("16-Week Implementation Plan vs Actual Verified Status")
    milestone_data = [
        {"Week": "Week 1", "Milestone": "Project Scope & Literature Review", "Status": "Completed", "Deliverable": "mKisan survey analysis, cross-dataset benchmarks, 4-tier architecture."},
        {"Week": "Week 2", "Milestone": "Regional Soil & Climate Data Ingestion", "Status": "Completed", "Deliverable": "641 IMD district rainfall normals and ICAR SHC Cycle-II benchmarks."},
        {"Week": "Week 3", "Milestone": "Automated Lookup Table Construction", "Status": "Completed", "Deliverable": "data/district_lookup.csv and SQLite tables with rating conversion rules."},
        {"Week": "Week 4", "Milestone": "Crop & Fertilizer Modules", "Status": "Completed", "Deliverable": "Random Forest crop recommender (99.55%) & multi-nutrient deficit fertilizer engine."},
        {"Week": "Week 5", "Milestone": "Disease Dataset Prep & Augmentation", "Status": "Completed", "Deliverable": "38-class catalog (14 crops) and field-realistic augmentations (blur, shadows)."},
        {"Week": "Week 6", "Milestone": "Attention-CNN Architecture Design", "Status": "Completed", "Deliverable": "Keras Attention-CNN with 3 SE blocks (Fig 4.1), Grad-CAM hooks & unified UI."},
        {"Week": "Week 7", "Milestone": "Deep Neural Network Model Training", "Status": "Pending (Next Milestone)", "Deliverable": "Adam optimizer training across 54,305 PlantVillage/PlantDoc images."},
        {"Week": "Week 8", "Milestone": "Cross-Dataset Benchmarking & Crop Conditioning", "Status": "Pending", "Deliverable": "Intra vs cross-dataset testing targeting 82.93% attention benchmark."},
        {"Week": "Week 9", "Milestone": "Explainable AI (Grad-CAM) Full Integration", "Status": "Scaffolded / Pending Weights", "Deliverable": "Grad-CAM activation overlays on final trained SE weights."},
        {"Week": "Week 10", "Milestone": "Backend REST API Deployment", "Status": "Completed (Pre-scheduled)", "Deliverable": "Flask REST API with health, district, crop, fertilizer & disease routes."},
        {"Week": "Week 11", "Milestone": "Unified Farmer Interface Refinement", "Status": "Completed (Pre-scheduled)", "Deliverable": "Responsive green-and-white Streamlit dashboard connecting all modules."},
        {"Week": "Week 12", "Milestone": "End-to-End Pipeline Integration", "Status": "Partially Completed", "Deliverable": "District -> Crop -> Fertilizer -> Pathology pipeline tethering."},
        {"Week": "Week 13", "Milestone": "System Robustness & Latency Stress Testing", "Status": "Pending", "Deliverable": "Concurrent latency benchmarking under noisy field imagery."},
        {"Week": "Week 14", "Milestone": "Extension Worker Validation Trials", "Status": "Pending", "Deliverable": "Field usability trials with agricultural extension functionaries."},
        {"Week": "Week 15", "Milestone": "System Optimization & Containerization", "Status": "Scaffolded", "Deliverable": "Docker containerization and <= 65 ms inference latency optimization."},
        {"Week": "Week 16", "Milestone": "Final Documentation & Viva Defense", "Status": "Pending", "Deliverable": "Final thesis report, code repository archival, and oral defense."},
    ]
    st.dataframe(pd.DataFrame(milestone_data), use_container_width=True)

    st.markdown("---")

    # Evaluation Plan Verification (Table 6.1)
    st.subheader("Evaluation Plan Target Criteria Verification (Table 6.1)")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown(
            """
            <div class="farm-card">
                <h4>Crop Recommendation Verification</h4>
                <p><b>Target:</b> Retain 99.09% on original split; separate district-input reporting.</p>
                <p><b>Empirical Result:</b> Test Accuracy: <b>99.55%</b> | Macro F1: <b>99.55%</b></p>
                <p><b>District Perturbation Accuracy:</b> <b>91.14%</b></p>
                <p style="color: #2e7d32; font-weight: 700;">✅ Target Exceeded</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tcol2:
        st.markdown(
            """
            <div class="farm-card">
                <h4>District Lookup Subsystem Verification</h4>
                <p><b>Target:</b> Complete values for all districts of selected states; handle missing data honestly.</p>
                <p><b>Empirical Result:</b> <b>641 districts</b> across 35 states/UTs ingested.</p>
                <p><b>Data Handling:</b> Unverified soil values flagged as 'Soil Pending' without fake imputation.</p>
                <p style="color: #2e7d32; font-weight: 700;">✅ Target Achieved</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Gantt Chart View
    st.markdown("---")
    st.subheader("16-Week Project Execution Gantt Chart")
    fig6_path = STATIC_DIR / "fig6_gantt.png"
    if fig6_path.exists():
        st.image(str(fig6_path), caption="Figure 6.1: 16-Week Project Execution Schedule (18 September - January 2027)", use_container_width=True)
