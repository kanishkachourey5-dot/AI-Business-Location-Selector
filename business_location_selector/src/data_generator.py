"""
AI-Based Business Location Selector - Synthetic Dataset Generator
Generates realistic candidate location records based on geospatial, demographic, 
cost, and competition factors as outlined in the project specification.
"""

import json
import os
import numpy as np
import pandas as pd

# Define Bhopal neighborhood centroids with realistic base properties
NEIGHBORHOODS = {
    "MP Nagar Zone 1": {"lat": 23.2332, "lon": 77.4300, "zone_type": "Commercial Hub", "base_rent": 65000, "base_footfall": 5500, "income": "High"},
    "MP Nagar Zone 2": {"lat": 23.2315, "lon": 77.4345, "zone_type": "Commercial Hub", "base_rent": 58000, "base_footfall": 4800, "income": "Medium"},
    "New Market": {"lat": 23.2384, "lon": 77.4018, "zone_type": "Traditional Retail Market", "base_rent": 70000, "base_footfall": 7500, "income": "Medium"},
    "Arera Colony E-3": {"lat": 23.2140, "lon": 77.4270, "zone_type": "Affluent Residential", "base_rent": 55000, "base_footfall": 2800, "income": "High"},
    "Bittan Market": {"lat": 23.2195, "lon": 77.4290, "zone_type": "High-Street Mixed", "base_rent": 60000, "base_footfall": 5200, "income": "High"},
    "TT Nagar": {"lat": 23.2350, "lon": 77.4060, "zone_type": "Commercial/Transit", "base_rent": 50000, "base_footfall": 4500, "income": "Medium"},
    "Kolar Road": {"lat": 23.1790, "lon": 77.4210, "zone_type": "Rapidly Growing Residential", "base_rent": 32000, "base_footfall": 3200, "income": "Medium"},
    "Hoshangabad Road": {"lat": 23.1950, "lon": 77.4480, "zone_type": "Commercial Corridor / Malls", "base_rent": 62000, "base_footfall": 5000, "income": "High"},
    "Indrapuri BHEL": {"lat": 23.2505, "lon": 77.4720, "zone_type": "Student/Industrial Hub", "base_rent": 30000, "base_footfall": 4200, "income": "Medium"},
    "Gulmohar Colony": {"lat": 23.2010, "lon": 77.4320, "zone_type": "Upscale Residential", "base_rent": 48000, "base_footfall": 2500, "income": "High"},
    "Shahpura": {"lat": 23.2080, "lon": 77.4200, "zone_type": "Lakeside Residential/Leisure", "base_rent": 45000, "base_footfall": 3000, "income": "High"},
    "Chunabhatti": {"lat": 23.2030, "lon": 77.4120, "zone_type": "Residential Suburban", "base_rent": 35000, "base_footfall": 2200, "income": "Medium"},
    "Ayodhya Bypass": {"lat": 23.2720, "lon": 77.4580, "zone_type": "Developing Suburban", "base_rent": 22000, "base_footfall": 1800, "income": "Low"},
    "Lalghati": {"lat": 23.2840, "lon": 77.3780, "zone_type": "Airport Corridor / Transit", "base_rent": 34000, "base_footfall": 3100, "income": "Medium"},
    "Bairagarh": {"lat": 23.2790, "lon": 77.3460, "zone_type": "Dense Wholesale Market", "base_rent": 42000, "base_footfall": 6000, "income": "Medium"},
    "Karond": {"lat": 23.2980, "lon": 77.4120, "zone_type": "Wholesale / Low-Cost Suburb", "base_rent": 20000, "base_footfall": 2400, "income": "Low"},
    "10 Number Market": {"lat": 23.2180, "lon": 77.4230, "zone_type": "Food & Retail Street", "base_rent": 68000, "base_footfall": 6200, "income": "High"},
    "Platinum Plaza": {"lat": 23.2380, "lon": 77.3990, "zone_type": "Corporate & Tech Hub", "base_rent": 64000, "base_footfall": 4100, "income": "High"},
    "Malviya Nagar": {"lat": 23.2390, "lon": 77.4110, "zone_type": "Office & Institutional Area", "base_rent": 52000, "base_footfall": 3800, "income": "Medium"},
    "Ashoka Garden": {"lat": 23.2550, "lon": 77.4320, "zone_type": "High Density Residential", "base_rent": 26000, "base_footfall": 2900, "income": "Low"},
}

BUSINESS_PROFILES = {
    "Cafe": {
        "avg_ticket": 260,
        "primary_facilities": ["Offices", "Colleges", "Shopping Mall"],
        "footfall_sensitivity": 0.35,
        "income_sensitivity": 0.25,
        "population_sensitivity": 0.15,
        "optimal_competitors": 4,
    },
    "Pharmacy": {
        "avg_ticket": 550,
        "primary_facilities": ["Hospitals", "Residential", "Transit Hub"],
        "footfall_sensitivity": 0.20,
        "income_sensitivity": 0.10,
        "population_sensitivity": 0.40,
        "optimal_competitors": 2,
    },
    "Grocery Store": {
        "avg_ticket": 750,
        "primary_facilities": ["Residential", "Transit Hub"],
        "footfall_sensitivity": 0.25,
        "income_sensitivity": 0.15,
        "population_sensitivity": 0.35,
        "optimal_competitors": 3,
    },
    "Clothing Store": {
        "avg_ticket": 1400,
        "primary_facilities": ["Shopping Mall", "High Street", "Colleges"],
        "footfall_sensitivity": 0.35,
        "income_sensitivity": 0.30,
        "population_sensitivity": 0.15,
        "optimal_competitors": 6,
    },
    "Quick Service Restaurant": {
        "avg_ticket": 320,
        "primary_facilities": ["Offices", "Colleges", "Transit Hub"],
        "footfall_sensitivity": 0.40,
        "income_sensitivity": 0.15,
        "population_sensitivity": 0.20,
        "optimal_competitors": 5,
    },
    "Co-Working Space": {
        "avg_ticket": 4500,
        "primary_facilities": ["Offices", "Metro Station", "Cafes"],
        "footfall_sensitivity": 0.15,
        "income_sensitivity": 0.35,
        "population_sensitivity": 0.20,
        "optimal_competitors": 2,
    },
}

FACILITY_OPTIONS = [
    "Offices, Colleges",
    "Offices, Transit Hub",
    "Residential, Hospitals",
    "Shopping Mall, High Street",
    "Colleges, Food Court",
    "Metro Station, Offices",
    "Residential, Schools",
    "Hospitals, Transit Hub",
    "Offices, Residential",
    "Transit Hub, Retail Market",
]

def generate_dataset(num_records=1200, random_seed=42):
    """Generates synthetic dataset with documented properties."""
    np.random.seed(random_seed)
    records = []

    neighborhood_names = list(NEIGHBORHOODS.keys())
    business_types = list(BUSINESS_PROFILES.keys())

    for i in range(1, num_records + 1):
        loc_id = f"LOC_{i:04d}"
        neighborhood = np.random.choice(neighborhood_names)
        nh_info = NEIGHBORHOODS[neighborhood]
        b_type = np.random.choice(business_types)
        b_profile = BUSINESS_PROFILES[b_type]

        # Add Gaussian jitter to coordinates
        lat = round(nh_info["lat"] + np.random.normal(0, 0.0035), 6)
        lon = round(nh_info["lon"] + np.random.normal(0, 0.0035), 6)

        # Monthly rent with variance based on zone
        rent_noise = np.random.normal(1.0, 0.18)
        monthly_rent = int(np.clip(nh_info["base_rent"] * rent_noise, 15000, 180000))
        monthly_rent = round(monthly_rent, -2) # Round to nearest 100

        # Catchment population
        pop_base = 35000 if "Residential" in nh_info["zone_type"] else 20000
        population = int(np.clip(pop_base * np.random.lognormal(0, 0.4), 6000, 95000))

        # Daily pedestrian footfall
        footfall_noise = np.random.normal(1.0, 0.22)
        daily_footfall = int(np.clip(nh_info["base_footfall"] * footfall_noise, 300, 15000))

        # Accessibility, parking, public transport scores (1 to 10)
        accessibility_score = round(float(np.clip(np.random.normal(7.2, 1.5), 1.0, 10.0)), 1)
        parking_score = round(float(np.clip(np.random.normal(6.1, 1.8), 1.0, 10.0)), 1)
        public_transport_score = round(float(np.clip(np.random.normal(7.5, 1.4), 1.0, 10.0)), 1)

        # Competitor count (influenced by popularity and market size)
        comp_mean = b_profile["optimal_competitors"] * (daily_footfall / 4000.0)
        competitor_count = int(np.clip(np.random.poisson(max(1, comp_mean)), 0, 22))

        # Income level
        income_choices = ["Low", "Medium", "High"]
        income_probs = (
            [0.05, 0.25, 0.70] if nh_info["income"] == "High"
            else [0.15, 0.65, 0.20] if nh_info["income"] == "Medium"
            else [0.60, 0.35, 0.05]
        )
        income_level = np.random.choice(income_choices, p=income_probs)
        income_multiplier = {"Low": 0.8, "Medium": 1.0, "High": 1.3}[income_level]

        # Nearby facilities
        nearby_facility = np.random.choice(FACILITY_OPTIONS)
        facility_bonus = 1.15 if any(f in nearby_facility for f in b_profile["primary_facilities"]) else 0.95

        # --- Synthetic Ground Truth: Target Demand (Daily Customer Count) ---
        # Demand modeled via non-linear economic factors:
        # Base conversion from footfall
        footfall_demand = (daily_footfall * 0.05) * b_profile["footfall_sensitivity"]
        # Demand from residential catchment population
        population_demand = (population * 0.003) * b_profile["population_sensitivity"]
        # Access factor
        access_factor = ((accessibility_score * 0.5) + (parking_score * 0.25) + (public_transport_score * 0.25)) / 10.0
        
        # Competition effect: up to optimal count it acts as an agglomeration benefit, beyond that it dilutes
        optimal_c = b_profile["optimal_competitors"]
        if competitor_count <= optimal_c:
            competition_multiplier = 0.9 + 0.1 * (competitor_count / max(1, optimal_c))
        else:
            competition_multiplier = 1.0 - 0.035 * (competitor_count - optimal_c)
        competition_multiplier = max(0.55, competition_multiplier)

        raw_daily_demand = (
            (footfall_demand + population_demand)
            * access_factor
            * (income_multiplier ** b_profile["income_sensitivity"])
            * facility_bonus
            * competition_multiplier
            * 2.4
        )

        # Realistic Gaussian noise representing unobserved location variance (decor, staff, weather)
        noise = np.random.normal(1.0, 0.12)
        target_demand = int(np.clip(raw_daily_demand * noise, 15, 850))

        # Monthly sales estimation
        avg_ticket = b_profile["avg_ticket"]
        monthly_sales = int(target_demand * avg_ticket * 30)

        # Business Potential category: ratio of monthly sales to rent, and footfall viability
        revenue_to_rent = monthly_sales / max(1, monthly_rent)
        if revenue_to_rent >= 7.5 and target_demand >= 120:
            potential = "Excellent"
        elif revenue_to_rent >= 5.0 and target_demand >= 75:
            potential = "Good"
        elif revenue_to_rent >= 3.2:
            potential = "Average"
        else:
            potential = "Poor"

        records.append({
            "location_id": loc_id,
            "location_name": f"{neighborhood} #{i%50 + 1}",
            "neighborhood": neighborhood,
            "city": "Bhopal",
            "latitude": lat,
            "longitude": lon,
            "business_type": b_type,
            "monthly_rent": monthly_rent,
            "population": population,
            "competitor_count": competitor_count,
            "daily_footfall": daily_footfall,
            "accessibility_score": accessibility_score,
            "parking_score": parking_score,
            "public_transport_score": public_transport_score,
            "income_level": income_level,
            "nearby_facilities": nearby_facility,
            "target_demand": target_demand,
            "monthly_sales": monthly_sales,
            "business_potential": potential,
        })

    df = pd.DataFrame(records)
    return df

def save_data(df, base_dir):
    """Saves dataset and metadata documentation."""
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "business_locations.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated and saved {len(df)} records to {csv_path}")

    # Dataset Dictionary and Documentation
    dictionary = {
        "dataset_name": "AI-Based Business Location Selector Synthetic Benchmark",
        "study_area": "Bhopal Metropolitan Area, Madhya Pradesh, India",
        "sample_size": len(df),
        "data_integrity_notice": (
            "This is a documented synthetic dataset crafted for academic and prototyping purposes. "
            "It captures multi-factor spatial, financial, and competitive dynamics of commercial retail."
        ),
        "features": {
            "location_id": {"type": "string", "description": "Unique candidate location identifier"},
            "location_name": {"type": "string", "description": "Candidate micro-location name"},
            "neighborhood": {"type": "string", "description": "Bhopal neighborhood/district"},
            "city": {"type": "string", "description": "Urban study center (Bhopal)"},
            "latitude": {"type": "float", "description": "Geographic latitude coordinate (WGS84)"},
            "longitude": {"type": "float", "description": "Geographic longitude coordinate (WGS84)"},
            "business_type": {"type": "categorical", "description": "Type of prospective commercial enterprise (Cafe, Pharmacy, etc.)"},
            "monthly_rent": {"type": "integer", "unit": "INR", "description": "Expected monthly property rental lease expense"},
            "population": {"type": "integer", "description": "Estimated catchment population within 2km radius"},
            "competitor_count": {"type": "integer", "description": "Count of existing competing businesses in immediate vicinity"},
            "daily_footfall": {"type": "integer", "unit": "pedestrians/day", "description": "Estimated average daily pedestrian count"},
            "accessibility_score": {"type": "float", "range": "1.0 - 10.0", "description": "Road connectivity and pedestrian ease"},
            "parking_score": {"type": "float", "range": "1.0 - 10.0", "description": "Availability and convenience of vehicle parking"},
            "public_transport_score": {"type": "float", "range": "1.0 - 10.0", "description": "Proximity to bus stops, metro stations, auto stands"},
            "income_level": {"type": "categorical", "values": ["Low", "Medium", "High"], "description": "Catchment area consumer purchasing power"},
            "nearby_facilities": {"type": "string", "description": "Surrounding generator anchors (Colleges, Offices, Hospitals, etc.)"},
            "target_demand": {"type": "integer", "unit": "customers/day", "description": "Observed/modeled daily customer transaction volume (Target for Regression)"},
            "monthly_sales": {"type": "integer", "unit": "INR/month", "description": "Estimated monthly gross revenue"},
            "business_potential": {"type": "categorical", "values": ["Excellent", "Good", "Average", "Poor"], "description": "Viability classification tier"}
        }
    }

    dict_path = os.path.join(data_dir, "dataset_dictionary.json")
    with open(dict_path, "w") as f:
        json.dump(dictionary, f, indent=4)
    print(f"Saved dataset dictionary to {dict_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_df = generate_dataset(num_records=1200, random_seed=42)
    save_data(dataset_df, project_root)
