"""
AI-Based Business Location Selector - Visualizations and Utility Functions
Provides crystal-clear, high-definition map generation (Folium) with multiple tile layers
(OpenStreetMap, Satellite, Positron, Dark Mode), layer switcher, and interactive Plotly analytics.
"""

import json
import os
import folium
from folium.plugins import MarkerCluster, Fullscreen
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_data(filepath):
    """Loads dataset from CSV."""
    return pd.read_csv(filepath)

def load_metrics(filepath):
    """Loads model performance metrics JSON."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

def create_folium_map(df, top_candidates=None, center_lat=23.2350, center_lon=77.4200, zoom_start=12):
    """
    Creates a crystal-clear, high-definition Folium map with:
    - High-contrast OpenStreetMap tiles as default (sharp roads, names, landmarks)
    - Satellite imagery layer (Esri World Imagery)
    - Clean Light and Dark mode layers
    - Leaflet Layer Switcher control
    - High-visibility custom SVG star pins for Top 5 candidates
    - On-map visual legend
    """
    # Initialize base map centered on the study area with OpenStreetMap default
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # 1. Additional Free, Non-API-Key Tile Layers (Esri & Open Source)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="🛰️ Satellite Aerial View (Esri)",
        attr="Esri, Maxar, Earthstar Geographics",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        name="🏙️ High-Contrast City Street Map (Esri)",
        attr="Esri, HERE, Garmin, USGS",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        name="🏔️ Topographic & Terrain Map (Esri)",
        attr="Esri, Intermap, NASA, USGS",
        control=True
    ).add_to(m)

    # Fullscreen control
    Fullscreen(position="topleft", title="Full Screen", title_cancel="Exit Full Screen").add_to(m)

    top_ids = set(top_candidates["location_id"]) if top_candidates is not None else set()

    # High-contrast cluster colors
    cluster_color_map = {
        0: "#27ae60", # Affordable/Emerging (Bright Emerald Green)
        1: "#2980b9", # Prime Commercial (Cobalt Blue)
        2: "#8e44ad", # Residential Catchment (Vibrant Purple)
        3: "#d35400", # Retail Core (Deep Orange)
    }

    marker_cluster = MarkerCluster(name="📍 All Candidate Locations", show=True).add_to(m)

    # Add candidate location markers
    for _, row in df.iterrows():
        loc_id = row["location_id"]
        is_top = loc_id in top_ids
        rank = row.get("suitability_rank", "-")
        score = row.get("suitability_score", 0)
        c_id = row.get("cluster_id", 0)
        color = cluster_color_map.get(c_id, "#34495e")

        popup_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; width: 250px; font-size: 13px; line-height: 1.4;">
            <div style="background: {'linear-gradient(135deg, #f39c12, #e67e22)' if is_top else 'linear-gradient(135deg, #1e3d59, #17b978)'}; color: white; padding: 8px 12px; border-radius: 6px 6px 0 0; font-weight: bold; font-size: 14px;">
                {'⭐ RANK #' + str(rank) + ': ' if is_top else ''}{row['location_name']}
            </div>
            <div style="padding: 10px; background: #ffffff; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 6px 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
                <div style="margin-bottom: 4px;"><b>Type:</b> {row['business_type']}</div>
                <div style="margin-bottom: 4px;"><b>Monthly Rent:</b> <span style="color: #c0392b; font-weight: bold;">₹{row['monthly_rent']:,}</span></div>
                <div style="margin-bottom: 4px;"><b>Daily Footfall:</b> {row['daily_footfall']:,} pedestrians</div>
                <div style="margin-bottom: 4px;"><b>Est. Demand:</b> <span style="color: #27ae60; font-weight: bold;">{row.get('predicted_demand', row.get('target_demand'))} cust/day</span></div>
                <div style="margin-bottom: 4px;"><b>Accessibility:</b> {row['accessibility_score']}/10 | <b>Competitors:</b> {row['competitor_count']}</div>
                <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #f0f0f0; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; color: #7f8c8d;"><b>{row.get('cluster_label', 'Hub')}</b></span>
                    <span style="background: #27ae60; color: white; padding: 2px 8px; border-radius: 10px; font-weight: bold; font-size: 12px;">{score} / 100</span>
                </div>
            </div>
        </div>
        """

        if is_top:
            # Highlight top candidates with distinct, prominent red/gold star markers
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"⭐ Rank #{rank}: {row['location_name']} (Score: {score}/100)",
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=7,
                popup=folium.Popup(popup_html, max_width=270),
                tooltip=f"{row['location_name']} | Score: {score}",
                color="#2c3e50",
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
            ).add_to(marker_cluster)

    # Layer Switcher Control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # In-map stylish floating legend
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 25px; 
        left: 25px; 
        width: 230px;
        background-color: rgba(255, 255, 255, 0.95);
        border: 2px solid #ccc;
        border-radius: 8px;
        z-index: 9999;
        font-family: Arial, sans-serif;
        font-size: 12px;
        padding: 10px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    ">
        <b style="font-size: 13px; color: #1e3d59;">🗺️ Location Map Legend</b><br>
        <div style="margin-top: 6px;"><span style="color: #e74c3c; font-size: 15px;">⭐</span> <b>Top 5 Recommendations</b></div>
        <div style="margin-top: 4px;"><span style="display:inline-block; width:12px; height:12px; background:#2980b9; border-radius:50%; margin-right:6px;"></span> Prime Commercial Hub</div>
        <div style="margin-top: 4px;"><span style="display:inline-block; width:12px; height:12px; background:#27ae60; border-radius:50%; margin-right:6px;"></span> Affordable / Emerging</div>
        <div style="margin-top: 4px;"><span style="display:inline-block; width:12px; height:12px; background:#8e44ad; border-radius:50%; margin-right:6px;"></span> Residential Catchment</div>
        <div style="margin-top: 4px;"><span style="display:inline-block; width:12px; height:12px; background:#d35400; border-radius:50%; margin-right:6px;"></span> High-Competition Core</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def create_radar_chart(top_df):
    """Creates a comparative radar chart for top recommended candidate locations."""
    categories = [
        "Demand Indicator",
        "Accessibility",
        "Footfall",
        "Parking & Transit",
        "Competition Opp.",
        "Rent Affordability"
    ]
    
    fig = go.Figure()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (_, row) in enumerate(top_df.head(4).iterrows()):
        values = [
            row.get("score_demand", 70),
            row.get("score_accessibility", 70),
            row.get("score_footfall", 70),
            row.get("score_parking_transport", 70),
            row.get("score_competition", 70),
            row.get("score_rent_affordability", 70),
        ]
        values.append(values[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=f"#{row.get('suitability_rank', idx+1)} {row['location_name'][:18]}",
            line=dict(color=colors[idx % len(colors)], width=2),
            opacity=0.65
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        title=dict(text="<b>Multi-Criteria Radar Comparison (Top Candidates)</b>", font=dict(size=15)),
        margin=dict(l=40, r=40, t=40, b=40),
        height=380
    )
    return fig

def create_rent_vs_demand_scatter(df, highlighted_ids=None):
    """Scatter plot of Monthly Rent vs Predicted Demand with cluster color encoding."""
    fig = px.scatter(
        df,
        x="monthly_rent",
        y="predicted_demand" if "predicted_demand" in df.columns else "target_demand",
        color="cluster_label" if "cluster_label" in df.columns else "business_type",
        size="daily_footfall",
        hover_data=["location_name", "suitability_score", "accessibility_score", "competitor_count"],
        labels={
            "monthly_rent": "Monthly Rent (₹)",
            "predicted_demand": "Predicted Daily Demand (Customers)",
            "target_demand": "Daily Demand (Customers)",
            "cluster_label": "Cluster Group",
            "daily_footfall": "Daily Footfall"
        },
        title="<b>Market Trade-Off: Rent vs. Customer Demand</b> (Bubble size = Footfall)",
        template="plotly_white"
    )
    fig.update_layout(
        height=420,
        margin=dict(l=40, r=40, t=45, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig

def create_feature_importance_plot(feature_importances):
    """Horizontal bar chart showing Random Forest feature importance rankings."""
    feats = [item["feature"].replace("business_type_", "Type: ").replace("_", " ").title() for item in feature_importances[:8]]
    vals = [item["importance"] * 100 for item in feature_importances[:8]]

    feats = feats[::-1]
    vals = vals[::-1]

    fig = go.Figure(go.Bar(
        x=vals,
        y=feats,
        orientation="h",
        marker=dict(color="#3498db", line=dict(color="#2980b9", width=1.5)),
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>Random Forest Feature Importance (% Contribution)</b>",
        xaxis_title="Relative Importance (%)",
        yaxis_title="",
        height=360,
        template="plotly_white",
        margin=dict(l=30, r=40, t=45, b=30)
    )
    return fig

def create_cluster_evaluation_plot(clustering_summary):
    """Elbow curve (Inertia) and Silhouette score plots."""
    k_vals = clustering_summary["k_range"]
    inertias = clustering_summary["inertias"]
    silhouettes = clustering_summary["silhouette_scores"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_vals,
        y=inertias,
        mode="lines+markers",
        name="Inertia (Elbow Method)",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=k_vals,
        y=[s * max(inertias) * 0.8 for s in silhouettes],
        mode="lines+markers",
        name="Silhouette Score (Scaled)",
        line=dict(color="#2ecc71", width=3, dash="dot"),
        marker=dict(size=8),
        hovertext=[f"Silhouette: {s:.3f}" for s in silhouettes]
    ))

    fig.update_layout(
        title="<b>Cluster Optimization: Elbow Method & Silhouette Score</b>",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Sum of Squared Errors (Inertia)",
        height=360,
        template="plotly_white",
        margin=dict(l=30, r=30, t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
