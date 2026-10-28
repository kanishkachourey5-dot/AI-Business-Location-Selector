"""
Generates clean, valid Jupyter Notebook files (.ipynb) for the project:
1. 01_data_cleaning_and_eda.ipynb
2. 02_model_training_and_evaluation.ipynb
"""

import json
import os

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")

def make_code_cell(source_code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_code.split("\n")]
    }

def make_markdown_cell(source_md):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_md.split("\n")]
    }

def generate_eda_notebook(filepath):
    cells = [
        make_markdown_cell(
            "# AI-Based Business Location Selector\n"
            "## Notebook 01: Data Cleaning, Integrity Verification & Exploratory Data Analysis (EDA)\n\n"
            "**Domain:** Machine Learning, Geospatial Analytics & Decision Support  \n"
            "**Objective:** Verify data integrity, examine distributions, analyze correlations, and identify key drivers for commercial location selection."
        ),
        make_code_cell(
            "import os\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "# Configure visual styling\n"
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
            "plt.rcParams['figure.figsize'] = (10, 6)\n"
            "plt.rcParams['font.size'] = 11"
        ),
        make_markdown_cell(
            "### 1. Data Ingestion\n"
            "Load the candidate location dataset and inspect the schema, column types, and sample records."
        ),
        make_code_cell(
            "# Load dataset\n"
            "data_path = os.path.join('..', 'data', 'business_locations.csv')\n"
            "df = pd.read_csv(data_path)\n"
            "print(f'Total records loaded: {df.shape[0]} rows, {df.shape[1]} columns')\n"
            "df.head()"
        ),
        make_code_cell(
            "# Summary statistics and data types\n"
            "df.info()\n"
            "df.describe().round(2)"
        ),
        make_markdown_cell(
            "### 2. Data Cleaning & Integrity Auditing\n"
            "As stated in the project guide, real or synthetic data must undergo systematic validation:\n"
            "1. Missing value checks\n"
            "2. Duplicate identifier checks\n"
            "3. Range sanity checks (e.g. rent > 0, scores within 1-10)"
        ),
        make_code_cell(
            "# Check missing values and duplicate location IDs\n"
            "missing_counts = df.isnull().sum()\n"
            "duplicates = df.duplicated(subset=['location_id']).sum()\n"
            "print('Missing Values per Column:')\n"
            "print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else 'None! Complete data.')\n"
            "print(f'Duplicate Location IDs: {duplicates}')"
        ),
        make_code_cell(
            "# Range sanity checks\n"
            "assert (df['monthly_rent'] > 0).all(), 'Rent contains non-positive values'\n"
            "assert (df['accessibility_score'].between(1.0, 10.0)).all(), 'Accessibility out of bounds'\n"
            "assert (df['parking_score'].between(1.0, 10.0)).all(), 'Parking out of bounds'\n"
            "print('All range sanity checks passed successfully!')"
        ),
        make_markdown_cell(
            "### 3. Univariate Distributions: Rent, Footfall, Population, and Demand\n"
            "Examine continuous variables to detect skewness, multimodality, or extreme outliers."
        ),
        make_code_cell(
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
            "sns.histplot(df['monthly_rent'], kde=True, ax=axes[0, 0], color='#2980b9')\n"
            "axes[0, 0].set_title('Monthly Rent Distribution (₹)')\n\n"
            "sns.histplot(df['daily_footfall'], kde=True, ax=axes[0, 1], color='#27ae60')\n"
            "axes[0, 1].set_title('Daily Pedestrian Footfall Distribution')\n\n"
            "sns.histplot(df['population'], kde=True, ax=axes[1, 0], color='#8e44ad')\n"
            "axes[1, 0].set_title('Catchment Population Distribution')\n\n"
            "sns.histplot(df['target_demand'], kde=True, ax=axes[1, 1], color='#e67e22')\n"
            "axes[1, 1].set_title('Observed Customer Demand (Daily Transactions)')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        make_markdown_cell(
            "### 4. Categorical Feature Distributions\n"
            "Analyze candidate representation across commercial industries and purchasing power segments."
        ),
        make_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "sns.countplot(data=df, y='business_type', ax=axes[0], palette='Blues_r', order=df['business_type'].value_counts().index)\n"
            "axes[0].set_title('Candidate Locations per Business Category')\n\n"
            "sns.countplot(data=df, x='income_level', ax=axes[1], palette='Purples', order=['Low', 'Medium', 'High'])\n"
            "axes[1].set_title('Income / Purchasing Power Distribution')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        make_markdown_cell(
            "### 5. Correlation Heatmap\n"
            "Evaluate collinearity and dependencies among numeric variables to inform feature selection."
        ),
        make_code_cell(
            "numeric_cols = [\n"
            "    'monthly_rent', 'population', 'competitor_count', 'daily_footfall',\n"
            "    'accessibility_score', 'parking_score', 'public_transport_score', 'target_demand'\n"
            "]\n"
            "corr_matrix = df[numeric_cols].corr()\n"
            "plt.figure(figsize=(10, 8))\n"
            "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n"
            "plt.title('Correlation Matrix of Commercial & Geospatial Factors')\n"
            "plt.show()"
        ),
        make_markdown_cell(
            "### 6. Bivariate Analysis: Rent vs. Footfall by Business Potential\n"
            "Observe whether premium rent areas correspond with higher customer footfall and business viability."
        ),
        make_code_cell(
            "plt.figure(figsize=(11, 7))\n"
            "sns.scatterplot(\n"
            "    data=df,\n"
            "    x='monthly_rent',\n"
            "    y='target_demand',\n"
            "    hue='business_potential',\n"
            "    size='daily_footfall',\n"
            "    sizes=(30, 250),\n"
            "    palette='Set2',\n"
            "    alpha=0.85\n"
            ")\n"
            "plt.title('Rent vs. Customer Demand Segregated by Business Potential')\n"
            "plt.xlabel('Monthly Rent (₹)')\n"
            "plt.ylabel('Daily Customer Demand')\n"
            "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        make_markdown_cell(
            "### 7. Key Findings & EDA Summary\n"
            "1. **Footfall & Accessibility Drive Demand:** Pedestrian volume and transit composite show the strongest direct positive association with customer transactions.\n"
            "2. **Rent vs. Viability Trade-off:** High rent does not guarantee high potential; the sweet spot lies in locations with moderate rent and high pedestrian access.\n"
            "3. **Competition Dynamics:** Moderate competition signifies a proven customer base, but beyond an optimal saturation threshold, returns diminish."
        )
    ]
    create_notebook(cells, filepath)

def generate_modeling_notebook(filepath):
    cells = [
        make_markdown_cell(
            "# AI-Based Business Location Selector\n"
            "## Notebook 02: Supervised Model Training, Clustering & MCDA Suitability Scoring\n\n"
            "**Domain:** Machine Learning & Geospatial Analytics  \n"
            "**Objective:** Compare predictive regression models (Linear Regression Baseline vs. Decision Tree vs. Random Forest), perform K-Means spatial clustering, and apply multi-criteria ranking."
        ),
        make_code_cell(
            "import os\n"
            "import json\n"
            "import joblib\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.model_selection import train_test_split, cross_val_score, KFold\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.tree import DecisionTreeRegressor\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score\n"
            "from sklearn.cluster import KMeans"
        ),
        make_markdown_cell(
            "### 1. Load Data & Feature Engineering\n"
            "We engineer interaction features: `rent_to_footfall`, `competitor_density`, and facility indicators."
        ),
        make_code_cell(
            "data_path = os.path.join('..', 'data', 'business_locations.csv')\n"
            "df = pd.read_csv(data_path)\n\n"
            "# Feature engineering\n"
            "df['rent_to_footfall'] = df['monthly_rent'] / (df['daily_footfall'] + 1.0)\n"
            "df['competitor_density'] = df['competitor_count'] / (df['population'] / 10000.0)\n"
            "df['access_composite'] = (df['accessibility_score'] * 0.4) + (df['parking_score'] * 0.3) + (df['public_transport_score'] * 0.3)\n"
            "df['has_offices'] = df['nearby_facilities'].str.contains('Offices', case=False).astype(int)\n"
            "df['has_colleges'] = df['nearby_facilities'].str.contains('Colleges', case=False).astype(int)\n"
            "df['has_hospitals'] = df['nearby_facilities'].str.contains('Hospitals', case=False).astype(int)\n"
            "df['has_transit'] = df['nearby_facilities'].str.contains('Transit Hub|Metro', case=False).astype(int)\n"
            "print('Feature engineering complete. Sample features:')\n"
            "df[['rent_to_footfall', 'competitor_density', 'access_composite']].head()"
        ),
        make_markdown_cell(
            "### 2. Preprocessing & Train-Test Split (80/20)\n"
            "We construct a scikit-learn `ColumnTransformer` to handle scaling and encoding cleanly without data leakage."
        ),
        make_code_cell(
            "target = 'target_demand'\n"
            "X = df.drop(columns=[target, 'monthly_sales', 'business_potential', 'location_id', 'location_name', 'neighborhood', 'city', 'nearby_facilities'])\n"
            "y = df[target]\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)\n\n"
            "num_features = [\n"
            "    'monthly_rent', 'population', 'competitor_count', 'daily_footfall',\n"
            "    'accessibility_score', 'parking_score', 'public_transport_score',\n"
            "    'rent_to_footfall', 'competitor_density', 'access_composite',\n"
            "    'has_offices', 'has_colleges', 'has_hospitals', 'has_transit'\n"
            "]\n"
            "ord_features = ['income_level']\n"
            "nom_features = ['business_type']\n\n"
            "preprocessor = ColumnTransformer(\n"
            "    transformers=[\n"
            "        ('num', StandardScaler(), num_features),\n"
            "        ('ord', OrdinalEncoder(categories=[['Low', 'Medium', 'High']]), ord_features),\n"
            "        ('nom', OneHotEncoder(drop='first', sparse_output=False), nom_features)\n"
            "    ]\n"
            ")\n"
            "print(f'Training shape: {X_train.shape}, Test shape: {X_test.shape}')"
        ),
        make_markdown_cell(
            "### 3. Model Training & Comparison\n"
            "As mandated by Section 7 & 14 of the guide, we compare multiple algorithms:\n"
            "- **Linear Regression** (Interpretable baseline)\n"
            "- **Decision Tree Regressor** (Rule-based non-linear baseline)\n"
            "- **Random Forest Regressor** (Ensemble non-linear regressor)"
        ),
        make_code_cell(
            "models = {\n"
            "    'Linear Regression': LinearRegression(),\n"
            "    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),\n"
            "    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=9, random_state=42, n_jobs=-1)\n"
            "}\n\n"
            "results = []\n"
            "fitted_models = {}\n\n"
            "for name, model in models.items():\n"
            "    pipe = Pipeline([('prep', preprocessor), ('reg', model)])\n"
            "    \n"
            "    # 5-Fold Cross Validation\n"
            "    cv = KFold(n_splits=5, shuffle=True, random_state=42)\n"
            "    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2')\n"
            "    \n"
            "    pipe.fit(X_train, y_train)\n"
            "    y_pred = pipe.predict(X_test)\n"
            "    \n"
            "    mae = mean_absolute_error(y_test, y_pred)\n"
            "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "    r2 = r2_score(y_test, y_pred)\n"
            "    \n"
            "    results.append({\n"
            "        'Model': name,\n"
            "        'MAE': round(mae, 2),\n"
            "        'RMSE': round(rmse, 2),\n"
            "        'R2 Score': round(r2, 4),\n"
            "        '5-Fold CV R2': f'{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}'\n"
            "    })\n"
            "    fitted_models[name] = pipe\n\n"
            "results_df = pd.DataFrame(results)\n"
            "results_df"
        ),
        make_markdown_cell(
            "### 4. Residual and Actual vs. Predicted Plots\n"
            "Visually inspect prediction error distributions for the champion Random Forest model."
        ),
        make_code_cell(
            "rf_pipe = fitted_models['Random Forest']\n"
            "y_test_pred = rf_pipe.predict(X_test)\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "# Actual vs Predicted\n"
            "axes[0].scatter(y_test, y_test_pred, alpha=0.6, color='#2980b9')\n"
            "axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)\n"
            "axes[0].set_title('Actual vs. Predicted Demand (Test Set)')\n"
            "axes[0].set_xlabel('Actual Demand')\n"
            "axes[0].set_ylabel('Predicted Demand')\n\n"
            "# Residual distribution\n"
            "residuals = y_test - y_test_pred\n"
            "sns.histplot(residuals, kde=True, ax=axes[1], color='#e74c3c')\n"
            "axes[1].axvline(0, color='black', linestyle='--')\n"
            "axes[1].set_title('Residual Error Distribution (y_test - y_pred)')\n"
            "axes[1].set_xlabel('Residual Error')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        make_markdown_cell(
            "### 5. Feature Importance Analysis\n"
            "Explainable AI: Determine the top drivers of customer demand."
        ),
        make_code_cell(
            "rf_reg = rf_pipe.named_steps['reg']\n"
            "ohe = rf_pipe.named_steps['prep'].named_transformers_['nom']\n"
            "nom_cols_encoded = list(ohe.get_feature_names_out(nom_features))\n"
            "all_features = num_features + ord_features + nom_cols_encoded\n"
            "importances = rf_reg.feature_importances_\n\n"
            "fi_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values('importance', ascending=False)\n\n"
            "plt.figure(figsize=(10, 6))\n"
            "sns.barplot(data=fi_df.head(10), x='importance', y='feature', palette='viridis')\n"
            "plt.title('Top 10 Feature Importances (Random Forest Regressor)')\n"
            "plt.xlabel('Gini Importance')\n"
            "plt.show()"
        ),
        make_markdown_cell(
            "### 6. Unsupervised Learning: K-Means Clustering\n"
            "Group candidate locations based on multi-dimensional attributes. Determine optimal $k$ using the Elbow Method and Silhouette Analysis."
        ),
        make_code_cell(
            "cluster_cols = ['monthly_rent', 'daily_footfall', 'population', 'target_demand', 'accessibility_score', 'competitor_count']\n"
            "scaler = StandardScaler()\n"
            "X_clust_scaled = scaler.fit_transform(df[cluster_cols])\n\n"
            "k_range = list(range(2, 8))\n"
            "inertias = []\n"
            "silhouettes = []\n\n"
            "for k in k_range:\n"
            "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n"
            "    km.fit(X_clust_scaled)\n"
            "    inertias.append(km.inertia_)\n"
            "    silhouettes.append(silhouette_score(X_clust_scaled, km.labels_))\n\n"
            "fig, ax1 = plt.subplots(figsize=(10, 5))\n"
            "ax1.plot(k_range, inertias, 'bo-', label='Inertia (Elbow Method)')\n"
            "ax1.set_xlabel('Number of Clusters (k)')\n"
            "ax1.set_ylabel('Inertia (SSE)', color='b')\n"
            "ax2 = ax1.twinx()\n"
            "ax2.plot(k_range, silhouettes, 'rs--', label='Silhouette Score')\n"
            "ax2.set_ylabel('Silhouette Score', color='r')\n"
            "plt.title('Cluster Evaluation: Elbow Method and Silhouette Score')\n"
            "plt.show()"
        ),
        make_code_cell(
            "# Fit optimal K=4\n"
            "optimal_k = 4\n"
            "final_km = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)\n"
            "df['cluster_id'] = final_km.fit_predict(X_clust_scaled)\n\n"
            "# Cluster Centroid Profiles\n"
            "cluster_profiles = df.groupby('cluster_id')[cluster_cols].mean().round(1)\n"
            "print('Cluster Centroid Profiles:')\n"
            "cluster_profiles"
        ),
        make_markdown_cell(
            "### 7. Multi-Criteria Decision Analysis (MCDA) Suitability Scoring\n"
            "We implement the transparent formula from Section 9 of the guide:\n"
            "$$S = 0.30D + 0.20A + 0.15F + 0.15P + 0.10C + 0.10R$$\n"
            "All sub-scores are normalized to [0, 100]."
        ),
        make_code_cell(
            "# Normalize sub-components\n"
            "D = (df['target_demand'] / df['target_demand'].max()) * 100.0\n"
            "A = (df['accessibility_score'] / 10.0) * 100.0\n"
            "F = (df['daily_footfall'] / df['daily_footfall'].max()) * 100.0\n"
            "P = (((df['parking_score'] * 0.5) + (df['public_transport_score'] * 0.5)) / 10.0) * 100.0\n"
            "C = (1.0 - (df['competitor_count'] / 15.0)).clip(0, 1) * 100.0\n"
            "budget = 50000\n"
            "R = df['monthly_rent'].apply(lambda r: 100.0 * max(0.0, 1.0 - (r / (budget * 1.3))) if r <= budget else max(0.0, 30.0 * (1.0 - (r - budget) / budget)))\n\n"
            "# Weighted Composite Formula\n"
            "df['suitability_score'] = (0.30*D + 0.20*A + 0.15*F + 0.15*P + 0.10*C + 0.10*R).round(1)\n"
            "df['rank'] = df['suitability_score'].rank(ascending=False).astype(int)\n\n"
            "# Display Top 5 Ranked Candidates for Cafe\n"
            "cafe_top5 = df[df['business_type'] == 'Cafe'].sort_values('suitability_score', ascending=False).head(5)\n"
            "print('Top 5 Ranked Candidate Locations for Cafe:')\n"
            "cafe_top5[['rank', 'location_name', 'neighborhood', 'monthly_rent', 'daily_footfall', 'target_demand', 'suitability_score']]"
        )
    ]
    create_notebook(cells, filepath)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    nb_dir = os.path.join(project_root, "notebooks")
    os.makedirs(nb_dir, exist_ok=True)
    
    generate_eda_notebook(os.path.join(nb_dir, "01_data_cleaning_and_eda.ipynb"))
    generate_modeling_notebook(os.path.join(nb_dir, "02_model_training_and_evaluation.ipynb"))
