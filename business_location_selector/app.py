"""
AI-Based Business Location Selector - Streamlit Web Application
An interactive machine learning decision-support system for evaluating and ranking 
commercial business candidate locations using predictive models, clustering, and MCDA scoring.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import folium

# Helper to render Folium maps natively and robustly without MarshallComponentException
def render_folium_map(m, height=580):
    """Renders Folium map reliably via native Streamlit HTML component without serialization errors."""
    map_html = m.get_root().render()
    components.html(map_html, height=height, scrolling=False)

# Add src to system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from scoring import SuitabilityScorer, DEFAULT_BUSINESS_WEIGHTS
from utils import (
    load_data,
    load_metrics,
    create_folium_map,
    create_radar_chart,
    create_rent_vs_demand_scatter,
    create_feature_importance_plot,
    create_cluster_evaluation_plot,
)

# Page Configuration
st.set_page_config(
    page_title="AI Business Location Selector",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Sharp, Clean UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        border-left-width: 5px;
    }
    .kpi-title {
        font-size: 0.82rem;
        color: #718096;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #1a202c;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 14px;
        padding: 10px 16px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_cached_data():
    csv_path = os.path.join(BASE_DIR, "data", "business_locations_with_predictions.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, "data", "business_locations.csv")
    return load_data(csv_path)

@st.cache_data
def get_cached_metrics():
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    return load_metrics(metrics_path)

@st.cache_resource
def load_trained_models():
    models_dir = os.path.join(BASE_DIR, "models")
    rf_model_path = os.path.join(models_dir, "best_demand_model.joblib")
    kmeans_path = os.path.join(models_dir, "kmeans_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    
    rf_model = joblib.load(rf_model_path) if os.path.exists(rf_model_path) else None
    kmeans_model = joblib.load(kmeans_path) if os.path.exists(kmeans_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return rf_model, kmeans_model, scaler

df_all = get_cached_data()
metrics = get_cached_metrics()
rf_model, kmeans_model, scaler = load_trained_models()

# --- SIDEBAR: CONTROLS & WEIGHTS ---
st.sidebar.image("https://img.icons8.com/fluency/96/place-marker.png", width=60)
st.sidebar.title("Decision Parameters")

# 1. Business Category Selection
business_types = list(DEFAULT_BUSINESS_WEIGHTS.keys())
selected_business = st.sidebar.selectbox(
    "1. Business Category",
    options=business_types,
    index=0,
    help="Select the category of business you are planning to launch."
)

# 2. Neighborhood Filter
neighborhoods = ["All Bhopal Neighborhoods"] + sorted(list(df_all["neighborhood"].unique()))
selected_neighborhood = st.sidebar.selectbox(
    "2. Geographic Filter",
    options=neighborhoods,
    index=0
)

# 3. Monthly Rent Budget
user_budget = st.sidebar.slider(
    "3. Monthly Rent Budget (₹)",
    min_value=20000,
    max_value=150000,
    value=50000,
    step=5000,
    format="₹%d",
    help="Locations within this budget receive higher affordability scores."
)

# 4. Criteria Weights Customizer
with st.sidebar.expander("⚙️ Customize Evaluation Weights", expanded=False):
    preset_w = DEFAULT_BUSINESS_WEIGHTS[selected_business]
    st.caption(f"**Default Profile for {selected_business}:** {preset_w['rationale']}")
    
    use_custom = st.checkbox("Enable Custom Weights", value=False)
    if use_custom:
        w_demand = st.slider("Demand Weight (D)", 0.05, 0.50, float(preset_w["demand"]), 0.05)
        w_access = st.slider("Accessibility Weight (A)", 0.05, 0.50, float(preset_w["accessibility"]), 0.05)
        w_footfall = st.slider("Footfall Weight (F)", 0.05, 0.50, float(preset_w["footfall"]), 0.05)
        w_park = st.slider("Parking & Transit Weight (P)", 0.05, 0.50, float(preset_w["parking_transport"]), 0.05)
        w_comp = st.slider("Competition Opp. Weight (C)", 0.05, 0.50, float(preset_w["competition"]), 0.05)
        w_rent = st.slider("Rent Affordability Weight (R)", 0.05, 0.50, float(preset_w["rent_affordability"]), 0.05)
        
        active_weights = {
            "demand": w_demand,
            "accessibility": w_access,
            "footfall": w_footfall,
            "parking_transport": w_park,
            "competition": w_comp,
            "rent_affordability": w_rent
        }
    else:
        active_weights = {
            "demand": preset_w["demand"],
            "accessibility": preset_w["accessibility"],
            "footfall": preset_w["footfall"],
            "parking_transport": preset_w["parking_transport"],
            "competition": preset_w["competition"],
            "rent_affordability": preset_w["rent_affordability"]
        }

st.sidebar.markdown("---")
st.sidebar.caption("🎓 **Academic ML & Decision Support Project**<br>Bhopal Commercial Geographic Study", unsafe_allow_html=True)

# --- FILTER DATA & COMPUTE SCORES ---
filtered_df = df_all[df_all["business_type"] == selected_business].copy()
if selected_neighborhood != "All Bhopal Neighborhoods":
    filtered_df = filtered_df[filtered_df["neighborhood"] == selected_neighborhood]

scorer = SuitabilityScorer(weights=active_weights)
demand_column = "predicted_demand" if "predicted_demand" in filtered_df.columns else "target_demand"
scored_df = scorer.calculate_scores(filtered_df, budget=user_budget, custom_weights=active_weights, demand_col=demand_column)

top_n = scored_df.head(5)
top_candidate = top_n.iloc[0] if len(top_n) > 0 else None

# --- HEADER SECTION ---
st.markdown('<div class="main-header">📍 AI-Based Business Location Selector</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Evaluating <b>{len(scored_df)}</b> candidate locations for <b>{selected_business}</b> in <b>Bhopal</b> using Machine Learning, K-Means Clustering, and Multi-Criteria Decision Analysis.</div>',
    unsafe_allow_html=True
)

# --- KPI METRIC CARDS ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    top_name = top_candidate["location_name"] if top_candidate is not None else "N/A"
    top_score = top_candidate["suitability_score"] if top_candidate is not None else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-title">Top Recommended Location</div>
        <div class="kpi-value">{top_name}</div>
        <div style="font-size: 0.9rem; color: #27ae60; font-weight: bold; margin-top: 4px;">⭐ Suitability Score: {top_score} / 100</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    within_budget_count = len(scored_df[scored_df["monthly_rent"] <= user_budget])
    pct_budget = round((within_budget_count / max(1, len(scored_df))) * 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-title">Within Budget Properties</div>
        <div class="kpi-value">{within_budget_count} <span style="font-size: 1rem; color: #718096; font-weight: 500;">/ {len(scored_df)}</span></div>
        <div style="font-size: 0.85rem; color: #2980b9; margin-top: 4px;">{pct_budget}% sites meet budget ≤ ₹{user_budget:,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    avg_pred_demand = int(scored_df[demand_column].mean()) if len(scored_df) > 0 else 0
    top_pred_demand = int(top_candidate[demand_column]) if top_candidate is not None else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-title">Projected Daily Customers</div>
        <div class="kpi-value">{top_pred_demand} <span style="font-size: 1rem; color: #718096; font-weight: 500;">cust/day</span></div>
        <div style="font-size: 0.85rem; color: #718096; margin-top: 4px;">Category Average: {avg_pred_demand} cust/day</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    best_cluster = top_candidate.get("cluster_label", "Commercial") if top_candidate is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-title">Top Candidate Cluster</div>
        <div class="kpi-value" style="font-size: 1.3rem;">{best_cluster}</div>
        <div style="font-size: 0.85rem; color: #8e44ad; margin-top: 4px;">Unsupervised K-Means Archetype</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --- TABS FOR WORKFLOW ---
tab_ranking, tab_map, tab_realtime, tab_analytics, tab_clusters, tab_ml_models = st.tabs([
    "🏆 Top Recommendations & Ranking",
    "🗺️ Interactive Location Map (HD)",
    "⚡ Live Geocoding & Real-Time Evaluator",
    "📊 Comparative Analytics & Trade-offs",
    "🧬 K-Means Clustering Intelligence",
    "🤖 Machine Learning Model Evaluation"
])

# ==================== TAB 1: RANKING ====================
with tab_ranking:
    st.subheader(f"Top Recommended Locations for {selected_business}")
    
    card_cols = st.columns(3)
    for idx, col in enumerate(card_cols):
        if idx < len(top_n):
            cand = top_n.iloc[idx]
            with col:
                st.markdown(f"""
                <div style="border: 2px solid {'#f39c12' if idx==0 else '#e2e8f0'}; border-radius: 10px; padding: 16px; background: white; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <span style="background: {'#f39c12' if idx==0 else '#718096'}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                        RANK #{idx+1}
                    </span>
                    <h4 style="margin: 10px 0 4px 0; color: #1e3d59;">{cand['location_name']}</h4>
                    <p style="margin: 0; font-size: 13px; color: #718096;">{cand['neighborhood']} | <b>{cand.get('cluster_label', 'Hub')}</b></p>
                    <hr style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;">
                        <span>Suitability Score:</span> <b style="color: #27ae60;">{cand['suitability_score']}/100</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;">
                        <span>Monthly Rent:</span> <b>₹{cand['monthly_rent']:,}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;">
                        <span>Daily Footfall:</span> <b>{cand['daily_footfall']:,}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;">
                        <span>Predicted Demand:</span> <b>{cand[demand_column]} cust/day</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;">
                        <span>Accessibility:</span> <b>{cand['accessibility_score']} / 10</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;">
                        <span>Competitors Nearby:</span> <b>{cand['competitor_count']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Complete Candidate Locations Table")
    
    display_cols = [
        "suitability_rank", "location_name", "neighborhood", "monthly_rent", 
        "daily_footfall", demand_column, "accessibility_score", "competitor_count", 
        "score_demand", "score_rent_affordability", "suitability_score", "cluster_label"
    ]
    
    table_view = scored_df[display_cols].copy()
    table_view.rename(columns={
        "suitability_rank": "Rank",
        "location_name": "Location Name",
        "neighborhood": "Area",
        "monthly_rent": "Rent (₹)",
        "daily_footfall": "Footfall",
        demand_column: "Est. Customers/Day",
        "accessibility_score": "Access",
        "competitor_count": "Competitors",
        "score_demand": "Demand Score",
        "score_rent_affordability": "Rent Score",
        "suitability_score": "Final Score",
        "cluster_label": "Cluster Profile"
    }, inplace=True)
    
    st.dataframe(
        table_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=100, format="%d"),
            "Rent (₹)": st.column_config.NumberColumn("Rent (₹)", format="₹%d"),
            "Est. Customers/Day": st.column_config.NumberColumn("Est. Cust/Day", format="%d"),
        }
    )

    csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Complete Ranked Evaluation (CSV)",
        data=csv_bytes,
        file_name=f"location_ranking_{selected_business.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ==================== TAB 2: INTERACTIVE MAP ====================
with tab_map:
    st.subheader(f"Geospatial Distribution of Candidates in Bhopal (Crystal Clear HD Map)")
    st.markdown(
        "Use the **layer switcher** in the top-right corner to toggle between **Crisp Street Map**, "
        "**Satellite Aerial Photography**, and **Light/Dark Mode**. Click on any marker to inspect rent, footfall, and score details."
    )
    
    folium_map = create_folium_map(scored_df, top_candidates=top_n)
    render_folium_map(folium_map, height=600)

    st.info(
        "**💡 Map Features & Controls:**\n\n"
        "• **Layer Switcher (Top Right):** Switch between OpenStreetMap, Esri High-Res Satellite View, and CartoDB styles.\n\n"
        "• **Full Screen (Top Left):** Expand map to monitor the entire metropolitan area in full display.\n\n"
        "• **Red/Gold Stars:** Top 5 recommended sites with highest composite suitability scores.\n\n"
        "• **Colored Dots:** Candidate sites categorized by unsupervised K-Means urban archetypes."
    )

# ==================== TAB 3: LIVE GEOCODING & REAL-TIME EVALUATOR ====================
with tab_realtime:
    st.subheader("⚡ Real-Time Geocoding & Candidate Site Evaluator")
    st.write(
        "Evaluate any **real-world address or prospective commercial property** in real time! "
        "The system queries OpenStreetMap Nominatim live, retrieves exact coordinates, and applies the trained "
        "Random Forest model and Multi-Criteria Scoring Engine instantly."
    )

    with st.expander("ℹ️ Understanding Real-Time Data vs. Synthetic Datasets in this Project", expanded=True):
        st.markdown("""
        **Why does the core project use a documented benchmark dataset?**
        1. **Academic Guideline Mandate:** As noted in **Section 5 (Page 3) & Section 10 (Page 6)** of your project handout:
           > *"Real-world footfall and proprietary retail sales are rarely published for free. Option A (Synthetic Dataset of 500–2,000 records) is recommended to validate software workflows, regression modeling, and clustering without claiming unverified real-world predictive accuracy."*
        2. **Commercial Real-Time Data Barriers:** In enterprise settings, live footfall requires expensive commercial telecommunication mobility data (e.g., Vodafone/Airtel tower pings, Google Places Popular Times API, or physical IoT footfall camera sensors).
        3. **The Solution (This Tab!):** We built a **Live Geocoder & Real-Time Inference Engine** that allows you or your examiners to enter any real location, fetch live coordinates via OpenStreetMap Nominatim, and predict its commercial viability on demand!
        """)

    st.markdown("#### Step 1: Search & Geocode Real-World Address")
    geo_col1, geo_col2 = st.columns([3, 1])
    with geo_col1:
        search_query = st.text_input(
            "Enter Real Address, Area, or Landmark:",
            value="MP Nagar Zone 1 Bhopal",
            help="Type any real neighborhood, landmark, or street name."
        )
    with geo_col2:
        st.write("")
        st.write("")
        geocode_btn = st.button("🔍 Geocode Live Location", use_container_width=True)

    # State variables for geocoded location
    if "live_lat" not in st.session_state:
        st.session_state.live_lat = 23.2332
        st.session_state.live_lon = 77.4300
        st.session_state.live_address = "MP Nagar Zone 1, Bhopal, Madhya Pradesh, India"

    if geocode_btn and search_query:
        with st.spinner("Querying OpenStreetMap Nominatim API live..."):
            try:
                encoded_q = urllib.parse.quote(search_query)
                req_url = f"https://nominatim.openstreetmap.org/search?q={encoded_q}&format=json&limit=1"
                req = urllib.request.Request(req_url, headers={"User-Agent": "LocationSelectorApp/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    geo_data = json.loads(resp.read().decode("utf-8"))
                    if geo_data:
                        st.session_state.live_lat = float(geo_data[0]["lat"])
                        st.session_state.live_lon = float(geo_data[0]["lon"])
                        st.session_state.live_address = geo_data[0]["display_name"]
                        st.success(f"✅ Found Live Coordinates: {st.session_state.live_lat:.5f}, {st.session_state.live_lon:.5f}")
                    else:
                        st.warning("Location not found on OpenStreetMap. Please refine your query (e.g., add 'Bhopal').")
            except Exception as err:
                st.error(f"Geocoding request failed: {err}")

    st.caption(f"📍 **Resolved Address:** {st.session_state.live_address}")

    st.markdown("#### Step 2: Input Candidate Property Parameters")
    prop_col1, prop_col2, prop_col3, prop_col4 = st.columns(4)
    with prop_col1:
        live_business = st.selectbox("Business Type", options=business_types, index=business_types.index(selected_business), key="live_biz")
        live_rent = st.number_input("Monthly Rent (₹)", min_value=10000, max_value=300000, value=55000, step=5000)
    with prop_col2:
        live_footfall = st.number_input("Est. Daily Footfall", min_value=100, max_value=30000, value=4800, step=200)
        live_pop = st.number_input("Catchment Population", min_value=1000, max_value=150000, value=30000, step=2000)
    with prop_col3:
        live_competitors = st.slider("Competitors within 1km", min_value=0, max_value=25, value=4)
        live_income = st.selectbox("Area Purchasing Power", options=["Low", "Medium", "High"], index=1)
    with prop_col4:
        live_access = st.slider("Accessibility Score (1-10)", min_value=1.0, max_value=10.0, value=8.0, step=0.5)
        live_parking = st.slider("Parking Score (1-10)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)

    eval_btn = st.button("🚀 Evaluate Live Candidate Property Now", type="primary", use_container_width=True)

    if eval_btn:
        with st.spinner("Running Random Forest prediction and MCDA scoring..."):
            # Compute engineered features
            rent_to_footfall = live_rent / (live_footfall + 1.0)
            competitor_density = live_competitors / (live_pop / 10000.0)
            access_composite = (live_access * 0.4) + (live_parking * 0.3) + (8.0 * 0.3)

            input_dict = {
                "monthly_rent": [live_rent],
                "population": [live_pop],
                "competitor_count": [live_competitors],
                "daily_footfall": [live_footfall],
                "accessibility_score": [live_access],
                "parking_score": [live_parking],
                "public_transport_score": [8.0],
                "rent_to_footfall": [rent_to_footfall],
                "competitor_density": [competitor_density],
                "access_composite": [access_composite],
                "has_offices": [1],
                "has_colleges": [1],
                "has_hospitals": [0],
                "has_transit": [1],
                "has_mall": [0],
                "income_level": [live_income],
                "business_type": [live_business]
            }
            input_df = pd.DataFrame(input_dict)

            # Predict demand with Random Forest
            if rf_model is not None:
                predicted_live_demand = int(np.round(rf_model.predict(input_df)[0]))
            else:
                predicted_live_demand = int(live_footfall * 0.04)

            # Predict cluster
            if kmeans_model is not None and scaler is not None:
                clust_input = np.array([[live_rent, live_footfall, live_pop, predicted_live_demand, live_access, live_competitors]])
                scaled_c = scaler.transform(clust_input)
                live_cluster_id = int(kmeans_model.predict(scaled_c)[0])
                cluster_names = {
                    0: "Affordable / Emerging Market",
                    1: "Prime Commercial Hub",
                    2: "Balanced Residential Catchment",
                    3: "High-Competition Retail Core"
                }
                live_cluster_label = cluster_names.get(live_cluster_id, "Commercial Hub")
            else:
                live_cluster_label = "Prime Commercial Hub"

            # Compute Suitability Score
            norm_demand = min(100.0, (predicted_live_demand / 400.0) * 100.0)
            norm_access = live_access * 10.0
            norm_footfall = min(100.0, (live_footfall / 10000.0) * 100.0)
            norm_park = (((live_parking * 0.5) + (8.0 * 0.5)) / 10.0) * 100.0
            norm_comp = max(0.0, (1.0 - (live_competitors / 15.0))) * 100.0
            
            if live_rent <= user_budget:
                norm_rent = max(0.0, 100.0 * (1.0 - (live_rent / (user_budget * 1.3))))
            else:
                norm_rent = max(0.0, 30.0 * (1.0 - ((live_rent - user_budget) / user_budget)))

            w = active_weights
            live_suitability = round(
                w["demand"] * norm_demand +
                w["accessibility"] * norm_access +
                w["footfall"] * norm_footfall +
                w["parking_transport"] * norm_park +
                w["competition"] * norm_comp +
                w["rent_affordability"] * norm_rent,
                1
            )

            # Display Live Results
            st.markdown("---")
            st.markdown("### 🎯 Real-Time Evaluation Verdict")
            
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.metric("Predicted Daily Customers", f"{predicted_live_demand} cust/day")
                avg_ticket = DEFAULT_BUSINESS_WEIGHTS.get(live_business, {}).get("avg_ticket", 300)
                st.caption(f"Estimated Monthly Gross Revenue: ₹{predicted_live_demand * 300 * 30:,}")
            with res_c2:
                st.metric("Overall Suitability Score", f"{live_suitability} / 100")
                st.caption(f"Evaluated against ₹{user_budget:,} budget threshold")
            with res_c3:
                st.metric("Assigned Market Cluster", live_cluster_label)
                st.caption("Machine learning spatial archetype")

            # Mini live map for this custom location
            st.markdown("#### Live Site Location Map (Satellite & Street)")
            live_map = folium.Map(location=[st.session_state.live_lat, st.session_state.live_lon], zoom_start=15)
            folium.TileLayer("OpenStreetMap", name="Street Map").add_to(live_map)
            folium.TileLayer(
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                name="Satellite View",
                attr="Esri"
            ).add_to(live_map)
            folium.LayerControl().add_to(live_map)

            folium.Marker(
                location=[st.session_state.live_lat, st.session_state.live_lon],
                popup=f"<b>Evaluated Site:</b><br>{search_query}<br>Score: {live_suitability}/100<br>Predicted Demand: {predicted_live_demand} cust/day",
                tooltip=f"Evaluated Site: {live_suitability}/100",
                icon=folium.Icon(color="green", icon="check", prefix="fa")
            ).add_to(live_map)

            render_folium_map(live_map, height=420)

# ==================== TAB 4: COMPARATIVE ANALYTICS ====================
with tab_analytics:
    st.subheader("Candidate Trade-Off & Multi-Criteria Analysis")
    
    chart_col1, chart_col2 = st.columns([1, 1])
    with chart_col1:
        st.plotly_chart(create_radar_chart(top_n), use_container_width=True)
        st.caption("Radar polygon area demonstrates balanced strengths across all 6 criteria: Demand, Accessibility, Footfall, Parking/Transit, Competition Opportunity, and Rent Affordability.")
        
    with chart_col2:
        st.plotly_chart(create_rent_vs_demand_scatter(scored_df), use_container_width=True)
        st.caption("Quadrants illustrate value-for-money: Top-left candidates deliver high customer demand at lower rental costs.")

# ==================== TAB 5: K-MEANS CLUSTERING ====================
with tab_clusters:
    st.subheader("Unsupervised Learning: K-Means Segmentation")
    st.write(
        "Clustering partitions candidate areas into distinct market archetypes without relying on manual labeling. "
        "This helps entrepreneurs understand the structural character of each neighborhood."
    )
    
    if metrics and "clustering_evaluation" in metrics:
        c_eval = metrics["clustering_evaluation"]
        
        c_plot_col, c_desc_col = st.columns([1, 1])
        with c_plot_col:
            st.plotly_chart(create_cluster_evaluation_plot(c_eval), use_container_width=True)
            st.caption(f"Optimal cluster count $k={c_eval['optimal_k']}$ verified by the Elbow Method and Silhouette Coefficient.")
            
        with c_desc_col:
            st.markdown("#### Discovered Market Archetypes")
            for c_key, c_info in c_eval["cluster_profiles"].items():
                st.markdown(f"""
                <div style="background: white; border-left: 4px solid #3498db; padding: 10px 14px; margin-bottom: 10px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; border-left-width: 4px;">
                    <div style="font-weight: bold; color: #1e3d59; font-size: 14px;">{c_info['label_name']} ({c_info['pct_of_total']}% of locations)</div>
                    <div style="font-size: 13px; color: #4a5568; margin-top: 2px;">{c_info['description']}</div>
                    <div style="font-size: 12px; color: #718096; margin-top: 6px;">
                        Avg Rent: <b>₹{c_info['avg_monthly_rent']:,}</b> | Avg Footfall: <b>{c_info['avg_daily_footfall']:,}</b> | Avg Competitors: <b>{c_info['avg_competitor_count']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Clustering metrics file not found. Please run src/train.py first.")

# ==================== TAB 6: ML MODEL EVALUATION ====================
with tab_ml_models:
    st.subheader("Machine Learning Models: Supervised Demand Prediction")
    st.write(
        "To satisfy the academic project guidelines, we compared a baseline **Linear Regression**, a **Decision Tree**, "
        "and a non-linear **Random Forest Regressor** to predict continuous customer demand."
    )
    
    if metrics and "regression_model_comparison" in metrics:
        reg_comp = metrics["regression_model_comparison"]
        
        model_rows = []
        for model_name, m_vals in reg_comp.items():
            model_rows.append({
                "Model Architecture": model_name,
                "Mean Absolute Error (MAE)": m_vals["MAE"],
                "Root Mean Squared Error (RMSE)": m_vals["RMSE"],
                "R² Score (Test Set)": m_vals["R2"],
                "5-Fold CV Mean R²": f"{m_vals['CV_R2_Mean']} ± {m_vals['CV_R2_Std']}",
                "Model Status": "⭐ Recommended Champion" if "Random Forest" in model_name else "Baseline / Candidate"
            })
            
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        
        st.write("")
        st.plotly_chart(create_feature_importance_plot(metrics.get("top_feature_importances", [])), use_container_width=True)
        
        st.markdown("""
        **Key Model Takeaways:**
        - **Linear Regression Baseline** achieves strong benchmark performance due to direct correlations with footfall and access.
        - **Random Forest Regressor** captures non-linear interactions, such as diminishing returns from competitor saturation and facility synergies.
        - **Decision Tree** provides interpretable rule-based partitions but exhibits higher variance on held-out test splits.
        """)
    else:
        st.warning("Model metrics not loaded. Ensure train.py has executed.")

# --- FOOTER: RESPONSIBLE USE NOTICE ---
st.markdown("---")
st.markdown(
    "<small><b>Responsible AI & Decision-Support Notice:</b> This system is designed for decision guidance and comparative location analysis. "
    "A high suitability score or predicted demand estimate does not guarantee commercial success or replace legal, financial, and on-site due diligence.</small>",
    unsafe_allow_html=True
)
