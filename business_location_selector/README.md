# 📍 AI-Based Business Location Selector Using Machine Learning, Clustering, and Multi-Criteria Decision Analysis

A complete, production-grade Machine Learning and Geospatial Decision-Support System developed in accordance with the **Student Project Guide**.

---

## 🌟 Overview & Highlights

Choosing a business location is difficult because high population does not always mean high sales, low rent may come with low customer demand, and competition can indicate either market saturation or proven demand. 

This project solves this multi-dimensional challenge by combining:
1. **Supervised Regression Modeling:** Comparing a **Linear Regression Baseline**, **Decision Tree**, and **Random Forest Regressor** to predict expected daily customer transactions and sales.
2. **Unsupervised K-Means Clustering:** Segmenting commercial properties into 4 distinct urban market archetypes (*Prime Commercial Hub*, *Affordable / Emerging Market*, *High-Competition Retail Core*, and *Balanced Residential Catchment*) validated using the **Elbow Method** and **Silhouette Analysis**.
3. **Multi-Criteria Suitability Scoring (MCDA):** A transparent, stakeholder-weighted formula:
   $$S = 0.30D + 0.20A + 0.15F + 0.15P + 0.10C + 0.10R$$
   incorporating budget constraints, accessibility, footfall, and competitor opportunity.
4. **Interactive Streamlit Web Dashboard:** Featuring an interactive **Folium geospatial map**, Plotly radar trade-off diagrams, rent vs. demand scatter plots, model explainability, and one-click CSV export.

---

## 📁 Project Directory Structure

```
business_location_selector/
├── data/
│   ├── business_locations.csv                     # Documented benchmark dataset (1,200 locations)
│   ├── business_locations_with_predictions.csv    # Enriched dataset with model predictions & clusters
│   └── dataset_dictionary.json                    # Feature definitions, schema & units
├── models/
│   ├── best_demand_model.joblib                   # Trained Random Forest Regressor Pipeline
│   ├── baseline_linear_model.joblib               # Trained Linear Regression Pipeline
│   ├── decision_tree_model.joblib                 # Trained Decision Tree Regressor Pipeline
│   ├── kmeans_model.joblib                        # Trained K-Means Clustering Model (k=4)
│   ├── scaler.joblib                              # StandardScaler for spatial clustering
│   └── model_metrics.json                         # Full evaluation metrics (MAE, RMSE, R2, Silhouette)
├── notebooks/
│   ├── 01_data_cleaning_and_eda.ipynb             # Exploratory Data Analysis & Integrity checks
│   └── 02_model_training_and_evaluation.ipynb     # Model training, comparison & clustering
├── src/
│   ├── data_generator.py                          # Synthetic dataset generator for Bhopal study area
│   ├── train.py                                   # End-to-end model training & evaluation pipeline
│   ├── scoring.py                                 # Multi-Criteria Suitability Scoring Engine
│   ├── utils.py                                   # Folium map & Plotly visualization utilities
│   └── generate_notebooks.py                      # Jupyter notebook builder
├── docs/
│   ├── PROJECT_REPORT.md                          # Comprehensive formal academic report
│   └── VIVA_QUESTIONS_AND_ANSWERS.md              # Detailed answers to all 20 Viva questions
├── app.py                                         # Full interactive Streamlit web dashboard
├── requirements.txt                               # Project dependencies
└── README.md                                      # Project guide & instructions
```

---

## 🚀 Quickstart & Setup Instructions

### 1. Prerequisites
Ensure Python 3.9+ is installed on your machine.

### 2. Install Required Libraries
Open your terminal in the `business_location_selector` directory and run:
```bash
python -m pip install -r requirements.txt
```

### 3. Generate Data & Train Models (Optional - Already Pre-built)
If you wish to re-generate the dataset or re-train models from scratch:
```bash
# Generate synthetic dataset of 1,200 records
python src/data_generator.py

# Train Linear Regression, Decision Tree, Random Forest & K-Means
python src/train.py
```

### 4. Launch the Interactive Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Results

### Supervised Demand Regression:
| Model Architecture | MAE (Cust/Day) | RMSE | Test $R^2$ | 5-Fold Cross-Validation $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 21.06 | 28.31 | 0.7658 | 0.7668 ± 0.0252 |
| **Decision Tree Regressor** | 27.40 | 38.65 | 0.5634 | 0.5258 ± 0.0929 |
| **Random Forest Regressor (Champion)** | 21.09 | 29.99 | 0.7371 | **0.7510 ± 0.0282** |

### Unsupervised K-Means ($k=4$):
- **Silhouette Score:** `0.2043`
- **Discovered Archetypes:**
  - `Prime Commercial Hub`: High rent, high pedestrian traffic, corporate appeal.
  - `Affordable / Emerging Market`: Low rent, high potential for bootstrap startups.
  - `Balanced Residential Catchment`: Moderate rent, loyal neighborhood customer base.
  - `High-Competition Retail Core`: High customer gravity with aggressive competition.

---

## 🎓 Academic Deliverables Included

- 📘 **Formal Project Report:** [`docs/PROJECT_REPORT.md`](file:///c:/Users/acer/OneDrive/Desktop/AI%20Video%20Automation/business_location_selector/docs/PROJECT_REPORT.md)
- 🎯 **Viva Voce Prep (20 Questions & Answers):** [`docs/VIVA_QUESTIONS_AND_ANSWERS.md`](file:///c:/Users/acer/OneDrive/Desktop/AI%20Video%20Automation/business_location_selector/docs/VIVA_QUESTIONS_AND_ANSWERS.md)
- 📓 **Jupyter Notebooks:**
  - Data Cleaning & EDA: [`notebooks/01_data_cleaning_and_eda.ipynb`](file:///c:/Users/acer/OneDrive/Desktop/AI%20Video%20Automation/business_location_selector/notebooks/01_data_cleaning_and_eda.ipynb)
  - Model Training & Evaluation: [`notebooks/02_model_training_and_evaluation.ipynb`](file:///c:/Users/acer/OneDrive/Desktop/AI%20Video%20Automation/business_location_selector/notebooks/02_model_training_and_evaluation.ipynb)
- 💾 **Documented Dataset:** [`data/business_locations.csv`](file:///c:/Users/acer/OneDrive/Desktop/AI%20Video%20Automation/business_location_selector/data/business_locations.csv) and [`data/dataset_dictionary.json`](file:///c:/Users/acer/OneDrive/Desktop/AI%20Video%20Automation/business_location_selector/data/dataset_dictionary.json)

---

## ⚖️ Responsible Use Notice
This application is designed as a **decision-support tool**. High suitability scores or predicted customer counts reflect model heuristics and synthetic training patterns; they do not guarantee commercial success or substitute for professional financial, legal, and on-site due diligence.
