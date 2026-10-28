"""
AI-Based Business Location Selector - Suitability Scoring Engine
Implements Multi-Criteria Decision Analysis (MCDA) and normalized weighted scoring.
Formula: S = w_D * D + w_A * A + w_F * F + w_P * P + w_C * C + w_R * R
"""

import numpy as np
import pandas as pd

DEFAULT_BUSINESS_WEIGHTS = {
    "Cafe": {
        "demand": 0.30,
        "accessibility": 0.20,
        "footfall": 0.15,
        "parking_transport": 0.15,
        "competition": 0.10,
        "rent_affordability": 0.10,
        "rationale": "High emphasis on customer demand, accessibility, and pedestrian flow; moderate competition tolerance."
    },
    "Pharmacy": {
        "demand": 0.25,
        "accessibility": 0.25,
        "footfall": 0.10,
        "parking_transport": 0.15,
        "competition": 0.15,
        "rent_affordability": 0.10,
        "rationale": "Prioritizes high accessibility and immediate catchment demand with lower dependence on random footfall."
    },
    "Grocery Store": {
        "demand": 0.25,
        "accessibility": 0.15,
        "footfall": 0.15,
        "parking_transport": 0.20,
        "competition": 0.10,
        "rent_affordability": 0.15,
        "rationale": "Requires strong parking/transit convenience and sustainable rent for high-volume, low-margin products."
    },
    "Clothing Store": {
        "demand": 0.35,
        "accessibility": 0.15,
        "footfall": 0.20,
        "parking_transport": 0.10,
        "competition": 0.10,
        "rent_affordability": 0.10,
        "rationale": "Dependent on high pedestrian volume, purchasing power, and destination shopping footfall."
    },
    "Quick Service Restaurant": {
        "demand": 0.30,
        "accessibility": 0.15,
        "footfall": 0.25,
        "parking_transport": 0.10,
        "competition": 0.10,
        "rent_affordability": 0.10,
        "rationale": "Critical dependence on impulse footfall near colleges, transit, and commercial corridors."
    },
    "Co-Working Space": {
        "demand": 0.25,
        "accessibility": 0.20,
        "footfall": 0.10,
        "parking_transport": 0.20,
        "competition": 0.10,
        "rent_affordability": 0.15,
        "rationale": "Values public transport, parking, high accessibility, and corporate/institutional proximity."
    }
}

class SuitabilityScorer:
    """Computes transparent Multi-Criteria Suitability Scores for candidate locations."""

    def __init__(self, weights=None):
        self.weights = weights or DEFAULT_BUSINESS_WEIGHTS["Cafe"]

    @staticmethod
    def normalize_feature(series, min_val=None, max_val=None, invert=False):
        """Normalizes a series to 0-100 range with clipping."""
        min_v = series.min() if min_val is None else min_val
        max_v = series.max() if max_val is None else max_val
        
        if max_v == min_v:
            norm = pd.Series(50.0, index=series.index)
        else:
            norm = (series - min_v) / (max_v - min_v) * 100.0
            
        norm = norm.clip(0.0, 100.0)
        return (100.0 - norm) if invert else norm

    def calculate_scores(self, df, budget=None, custom_weights=None, demand_col="target_demand"):
        """
        Calculates individual normalized component scores and the final weighted suitability score.
        All sub-scores are bounded [0, 100].
        """
        data = df.copy()
        w = custom_weights or self.weights

        # Normalize weights to ensure sum == 1.0
        total_w = sum(w.values()) if isinstance(w, dict) else 1.0
        norm_w = {k: v / total_w for k, v in w.items() if k != "rationale"}

        # 1. Demand Score (D) [0-100]
        # Evaluated relative to the maximum observed/predicted demand in dataset
        max_demand = max(500, data[demand_col].max())
        data["score_demand"] = self.normalize_feature(data[demand_col], min_val=0, max_val=max_demand)

        # 2. Accessibility Score (A) [0-100] (raw score is 1-10)
        data["score_accessibility"] = (data["accessibility_score"] / 10.0 * 100.0).clip(0, 100)

        # 3. Footfall Score (F) [0-100]
        max_footfall = max(10000, data["daily_footfall"].max())
        data["score_footfall"] = self.normalize_feature(data["daily_footfall"], min_val=0, max_val=max_footfall)

        # 4. Parking & Public Transport Composite (P) [0-100]
        data["score_parking_transport"] = (
            ((data["parking_score"] * 0.5) + (data["public_transport_score"] * 0.5)) / 10.0 * 100.0
        ).clip(0, 100)

        # 5. Competition Opportunity Score (C) [0-100]
        # Inverted: lower competition count yields higher opportunity score
        data["score_competition"] = self.normalize_feature(data["competitor_count"], min_val=0, max_val=15, invert=True)

        # 6. Rent Affordability Score (R) [0-100]
        if budget is not None and budget > 0:
            # Score is 100 when rent is very low compared to budget, decreasing to 30 at budget limit
            # Locations exceeding budget drop steeply
            affordability = data["monthly_rent"].apply(
                lambda r: 100.0 * max(0.0, 1.0 - (r / (budget * 1.3))) if r <= budget else max(0.0, 30.0 * (1.0 - (r - budget) / budget))
            )
            data["score_rent_affordability"] = affordability.clip(0, 100)
        else:
            # Fallback when no budget specified: lower rent scores higher
            data["score_rent_affordability"] = self.normalize_feature(data["monthly_rent"], min_val=15000, max_val=120000, invert=True)

        # Compute Final Weighted Composite Score (S) [0-100]
        data["suitability_score"] = (
            norm_w["demand"] * data["score_demand"] +
            norm_w["accessibility"] * data["score_accessibility"] +
            norm_w["footfall"] * data["score_footfall"] +
            norm_w["parking_transport"] * data["score_parking_transport"] +
            norm_w["competition"] * data["score_competition"] +
            norm_w["rent_affordability"] * data["score_rent_affordability"]
        ).round(1)

        # Rank locations descending
        data["suitability_rank"] = data["suitability_score"].rank(ascending=False, method="min").astype(int)
        
        return data.sort_values(by="suitability_score", ascending=False)
