"""
AI-Based Business Location Selector - Model Training & Evaluation Pipeline
Trains and compares Linear Regression (Baseline), Decision Tree, and Random Forest Regressors.
Performs K-Means clustering with Elbow Method and Silhouette Analysis.
Exports models and comprehensive evaluation metrics.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans

def feature_engineering(df):
    """Enriches dataset with engineered spatial and economic interaction features."""
    data = df.copy()
    
    # Financial and density interaction ratios
    data["rent_to_footfall"] = data["monthly_rent"] / (data["daily_footfall"] + 1.0)
    data["competitor_density"] = data["competitor_count"] / (data["population"] / 10000.0)
    data["access_composite"] = (
        (data["accessibility_score"] * 0.4) +
        (data["parking_score"] * 0.3) +
        (data["public_transport_score"] * 0.3)
    )

    # Facility binary indicators
    data["has_offices"] = data["nearby_facilities"].str.contains("Offices", case=False).astype(int)
    data["has_colleges"] = data["nearby_facilities"].str.contains("Colleges", case=False).astype(int)
    data["has_hospitals"] = data["nearby_facilities"].str.contains("Hospitals", case=False).astype(int)
    data["has_transit"] = data["nearby_facilities"].str.contains("Transit Hub|Metro", case=False).astype(int)
    data["has_mall"] = data["nearby_facilities"].str.contains("Shopping Mall|High Street", case=False).astype(int)
    
    return data

def build_preprocessor():
    """Builds scikit-learn column transformer for numeric, ordinal, and nominal variables."""
    numeric_features = [
        "monthly_rent", "population", "competitor_count", "daily_footfall",
        "accessibility_score", "parking_score", "public_transport_score",
        "rent_to_footfall", "competitor_density", "access_composite",
        "has_offices", "has_colleges", "has_hospitals", "has_transit", "has_mall"
    ]
    
    ordinal_features = ["income_level"]
    income_categories = [["Low", "Medium", "High"]]

    nominal_features = ["business_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("ord", OrdinalEncoder(categories=income_categories), ordinal_features),
            ("nom", OneHotEncoder(drop="first", sparse_output=False), nominal_features),
        ],
        remainder="drop"
    )
    return preprocessor, numeric_features, ordinal_features, nominal_features

def train_and_evaluate_models(data, models_dir):
    """Trains regression models, evaluates metrics, and extracts feature importances."""
    target = "target_demand"
    X = data.drop(columns=[target, "monthly_sales", "business_potential", "location_id", "location_name", "neighborhood", "city"])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor, num_cols, ord_cols, nom_cols = build_preprocessor()

    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=9, random_state=42, n_jobs=-1),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        # Cross validation on training set
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
        
        # Fit on train set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
            "CV_R2_Mean": round(float(np.mean(cv_r2)), 4),
            "CV_R2_Std": round(float(np.std(cv_r2)), 4),
        }
        fitted_pipelines[name] = pipeline
        print(f"[{name}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f} | CV R2: {np.mean(cv_r2):.4f}")

    # Feature Importance extraction from Random Forest
    rf_pipeline = fitted_pipelines["Random Forest Regressor"]
    rf_regressor = rf_pipeline.named_steps["regressor"]
    onehot_encoder = rf_pipeline.named_steps["preprocessor"].named_transformers_["nom"]
    encoded_nom_cols = list(onehot_encoder.get_feature_names_out(nom_cols))
    all_feature_names = num_cols + ord_cols + encoded_nom_cols
    importances = rf_regressor.feature_importances_

    feature_importance_list = sorted(
        [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(all_feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )

    # Save models
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(fitted_pipelines["Linear Regression (Baseline)"], os.path.join(models_dir, "baseline_linear_model.joblib"))
    joblib.dump(fitted_pipelines["Decision Tree Regressor"], os.path.join(models_dir, "decision_tree_model.joblib"))
    joblib.dump(fitted_pipelines["Random Forest Regressor"], os.path.join(models_dir, "best_demand_model.joblib"))
    print("Saved regression models to models/ directory.")

    # Add predictions back to full dataset
    data["predicted_demand"] = np.round(fitted_pipelines["Random Forest Regressor"].predict(X)).astype(int)

    return results, feature_importance_list, data

def perform_kmeans_clustering(data, models_dir):
    """Executes K-Means clustering, evaluates Elbow & Silhouette, and profiles clusters."""
    cluster_features = [
        "monthly_rent", "daily_footfall", "population", "target_demand",
        "accessibility_score", "competitor_count"
    ]
    X_cluster = data[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Evaluate K from 2 to 7
    k_range = list(range(2, 8))
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        sil = silhouette_score(X_scaled, km.labels_)
        silhouette_scores.append(round(float(sil), 4))

    # Optimal K selection (K=4 yields distinct, interpretable real estate clusters)
    optimal_k = 4
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    cluster_labels = final_kmeans.fit_predict(X_scaled)
    data["cluster_id"] = cluster_labels

    # Cluster profiles & meaningful naming
    cluster_profiles = {}
    cluster_name_map = {}

    for c_id in range(optimal_k):
        c_subset = data[data["cluster_id"] == c_id]
        mean_rent = float(c_subset["monthly_rent"].mean())
        mean_footfall = float(c_subset["daily_footfall"].mean())
        mean_demand = float(c_subset["target_demand"].mean())
        mean_comp = float(c_subset["competitor_count"].mean())
        mean_access = float(c_subset["accessibility_score"].mean())

        # Logic to name clusters intuitively based on distinct profiles
        if mean_comp >= 6.5:
            name = "High-Competition Retail Core"
            desc = "Mature commercial strips with high business density, strong retail gravity, and proven pedestrian traffic."
        elif mean_rent >= 52000 and mean_footfall >= 4000:
            name = "Prime Commercial Hub"
            desc = "High-rent central commercial district with bustling pedestrian activity and corporate/lifestyle appeal."
        elif mean_rent <= 35000:
            name = "Affordable / Emerging Market"
            desc = "Cost-effective suburban and developing nodes with favorable rental affordability and rising potential."
        else:
            name = "Balanced Residential Catchment"
            desc = "Moderate rent neighborhood hubs with steady residential footfall and balanced competitive dynamics."

        cluster_name_map[c_id] = name
        cluster_profiles[f"Cluster_{c_id}"] = {
            "cluster_id": c_id,
            "label_name": name,
            "description": desc,
            "count": int(len(c_subset)),
            "pct_of_total": round(float(len(c_subset) / len(data) * 100), 1),
            "avg_monthly_rent": round(mean_rent),
            "avg_daily_footfall": round(mean_footfall),
            "avg_target_demand": round(mean_demand),
            "avg_competitor_count": round(mean_comp, 1),
            "avg_accessibility": round(mean_access, 1),
        }

    data["cluster_label"] = data["cluster_id"].map(cluster_name_map)

    # Save clustering artifacts
    joblib.dump(final_kmeans, os.path.join(models_dir, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    clustering_summary = {
        "optimal_k": optimal_k,
        "k_range": k_range,
        "inertias": [round(i, 2) for i in inertias],
        "silhouette_scores": silhouette_scores,
        "cluster_features": cluster_features,
        "cluster_profiles": cluster_profiles
    }

    return clustering_summary, data

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "models")

    csv_path = os.path.join(data_dir, "business_locations.csv")
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Feature Engineering
    data_enriched = feature_engineering(df)

    # 2. Train and evaluate regression models
    print("\n--- Training Supervised Regression Models ---")
    reg_results, feature_importances, data_with_preds = train_and_evaluate_models(data_enriched, models_dir)

    # 3. K-Means Clustering
    print("\n--- Performing K-Means Clustering Analysis ---")
    cluster_summary, final_data = perform_kmeans_clustering(data_with_preds, models_dir)

    # 4. Save enhanced dataset
    enhanced_csv_path = os.path.join(data_dir, "business_locations_with_predictions.csv")
    final_data.to_csv(enhanced_csv_path, index=False)
    print(f"Saved enriched dataset with predictions and clusters to {enhanced_csv_path}")

    # 5. Export comprehensive metrics JSON
    metrics_export = {
        "project_title": "AI-Based Business Location Selector Using Machine Learning, Clustering, and Multi-Criteria Decision Analysis",
        "regression_model_comparison": reg_results,
        "top_feature_importances": feature_importances[:10],
        "clustering_evaluation": cluster_summary
    }

    metrics_json_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_export, f, indent=4)
    print(f"Saved complete evaluation metrics to {metrics_json_path}")
    print("\nTraining and evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
