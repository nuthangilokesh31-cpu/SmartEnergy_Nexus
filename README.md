# ⚡ SmartEnergy Nexus

## Intelligent Energy Consumption Forecasting for Smart Buildings

SmartEnergy Nexus is an end-to-end AI-powered energy intelligence platform designed to forecast electricity demand, analyze prediction behavior, identify peak-demand periods, explain model decisions, and evaluate energy optimization scenarios.

The system combines **time-series feature engineering, machine learning, explainable AI, peak-demand analysis, and prescriptive optimization** into a single interactive Streamlit dashboard.

### 🚀 Live Demo

**Streamlit Cloud:**
https://smartenergynexus-otchpgfrgq5kdqlxzmrxtw.streamlit.app/

**GitHub Repository:**
https://github.com/nuthangilokesh31-cpu/SmartEnergy_Nexus

---

## 🎯 Project Objective

The objective of SmartEnergy Nexus is to predict future electricity consumption at **30-minute intervals** and provide actionable intelligence for smart-building energy management.

The platform is designed to answer four important questions:

1. **How much energy will be consumed?**
2. **When is demand likely to become high or very high?**
3. **Why did the forecasting model make a particular prediction?**
4. **What energy-management scenarios can be evaluated using the forecast?**

---

## 🏗️ System Architecture

```text
SGSC Energy Dataset
        │
        ▼
Data Inspection & Profiling
        │
        ▼
Time-Series Validation
        │
        ▼
Aggregate Energy Dataset
        │
        ▼
Feature Engineering
        │
        ├── Calendar Features
        ├── Cyclical Features
        ├── Lag Features
        ├── Rolling Statistics
        └── Change Features
        │
        ▼
Chronological Train/Test Split
        │
        ▼
Model Comparison
        │
        ├── Persistence
        ├── Seasonal Naive
        ├── Linear Regression
        └── Random Forest
        │
        ▼
Model Validation & Error Analysis
        │
        ▼
SHAP Explainability
        │
        ▼
Peak Demand Risk Analysis
        │
        ▼
Energy Optimization Scenarios
        │
        ▼
Model Packaging & Compression
        │
        ▼
Streamlit Dashboard
        │
        ▼
Cloud Deployment
```

---

## 📊 Dataset

The project uses the **SGSC clustered residential electricity load dataset**.

The original dataset contains:

* **6,031 customer load series**
* **17,520 half-hourly timestamps**
* **30-minute resolution**
* **One complete year of data**
* **2013 calendar year**

The customer-level load series were aggregated to construct a system-level electricity demand signal for forecasting.

---

## 🔍 Data Quality Validation

The time-series validation pipeline verified:

| Validation           |     Result |
| -------------------- | ---------: |
| Expected records     |     17,520 |
| Actual records       |     17,520 |
| Invalid timestamps   |          0 |
| Duplicate timestamps |          0 |
| Missing time gaps    |          0 |
| Interval             | 30 minutes |

The resulting dataset passed the time-series validation checks.

---

## ⚙️ Feature Engineering

The forecasting dataset contains engineered temporal and historical-demand features.

### Calendar Features

* Hour
* Minute
* Day of week
* Day of month
* Month
* Day of year
* Week of year
* Weekend indicator
* Month-start indicator
* Month-end indicator

### Cyclical Features

Cyclical transformations were created for temporal variables so that periodic relationships can be represented numerically.

### Historical Load Features

The model uses historical demand information through:

* Lag features
* Rolling statistics
* Recent-load relationships
* Historical demand patterns

Two change features were retained for analysis but excluded from model inputs to reduce leakage risk:

```text
load_change_30min
load_change_1hour
```

---

## 🤖 Machine Learning

Multiple forecasting approaches were evaluated:

* Persistence baseline
* Seasonal Naive baseline
* Linear Regression
* Random Forest Regressor

The final deployment model is:

```text
RandomForestRegressor
```

The model uses **29 forecasting features**.

---

## 📈 Model Performance

The deployed forecasting system reports:

| Metric              |                  Result |
| ------------------- | ----------------------: |
| Forecast resolution |              30 minutes |
| Forecast MAE        |                   58.91 |
| Forecast R²         |                  0.9933 |
| Model               | Random Forest Regressor |
| Model features      |                      29 |

The model was additionally evaluated using representative periods from the held-out test set.

### Sample Period Testing

Nine representative test periods were evaluated.

Results:

* **9/9 samples within 5% error**
* **9/9 samples within 10% error**
* Average percentage error: **1.7170%**
* Maximum percentage error: **4.8156%**

These sample tests complement the aggregate evaluation metrics and demonstrate model behavior across different demand conditions.

---

## 🧪 Deployment Validation

Before deployment, the serialized model was independently validated.

Validation confirmed:

```text
Model type:
RandomForestRegressor

Expected features:
29

Test observations:
3,437

Prediction test:
PASSED
```

The model successfully generated predictions after being reloaded from the deployment artifact.

---

## 🗜️ Model Compression

The original deployment model was approximately:

```text
103.11 MiB
```

The compressed deployment artifact was reduced to approximately:

```text
32.17 MiB
```

This represents a size reduction of approximately:

```text
68.80%
```

Prediction equivalence was also tested after reloading the compressed model.

Maximum observed prediction difference:

```text
0.000000000002
```

This confirmed that compression preserved prediction behavior to extremely high numerical precision.

---

## 🧠 Explainable AI

SmartEnergy Nexus incorporates SHAP-based explainability to help understand the contribution of input features to model predictions.

The dashboard provides:

* Feature importance
* Global model explanation
* Local prediction explanation
* SHAP visualizations

This makes the forecasting system more interpretable than a black-box prediction-only application.

---

## ⚠️ Peak Demand Intelligence

The platform analyzes electricity demand to identify elevated-demand periods.

Demand conditions are classified into categories such as:

* Normal Demand
* High Demand
* Very High Demand

Peak-demand analysis can help building operators identify periods where energy-management strategies may be particularly relevant.

---

## ⚡ Energy Optimization

The project extends forecasting into a prescriptive analytics layer.

Optimization analysis evaluates modeled energy-management scenarios and produces:

* Optimization scenarios
* Optimization recommendations
* Energy optimization insights
* Peak-demand analysis

The objective is to connect **forecasting intelligence with operational decision support**.

---

## 🖥️ Interactive Dashboard

The Streamlit application contains the following modules:

### Executive Dashboard

Provides a high-level overview of:

* Customer load series
* Forecast resolution
* Forecast MAE
* Forecast R²
* Actual vs predicted demand
* Model status

### Energy Forecast

Provides forecasting-related analysis and predicted energy demand.

### Peak Demand

Provides peak-demand and demand-risk analysis.

### Explainable AI

Provides SHAP-based model explanations.

### Optimization

Provides modeled energy optimization scenarios and recommendations.

### Model Performance

Provides model evaluation and forecasting performance information.

### Data Quality

Provides information about dataset and time-series validation.

---

## 📁 Project Structure

```text
SmartEnergy_Nexus/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── processed/
│       ├── aggregate_energy.csv
│       ├── forecasting_features.csv
│       ├── train.csv
│       ├── test.csv
│       ├── test_predictions.csv
│       ├── model_comparison.csv
│       ├── stability_validation.csv
│       ├── forecast_error_analysis.csv
│       ├── shap_feature_importance.csv
│       ├── shap_local_explanation.csv
│       ├── peak_demand_analysis.csv
│       ├── optimization_scenarios.csv
│       ├── optimization_recommendations.csv
│       └── ...
│
├── models/
│   ├── best_forecasting_model_compressed.joblib
│   └── model_metadata.json
│
├── reports/
│   └── figures/
│       ├── actual_vs_predicted.png
│       ├── forecast_error_over_time.png
│       ├── hourly_forecast_mae.png
│       ├── optimization_scenarios.png
│       ├── peak_demand_risk.png
│       ├── shap_feature_importance.png
│       └── shap_summary.png
│
└── src/
    ├── 01_inspect_data.py
    ├── 02_profile_full_data.py
    ├── 03_analyze_load.py
    ├── 04_validate_time_series.py
    ├── 05_create_forecasting_dataset.py
    ├── 06_feature_engineering.py
    ├── 07_split_data.py
    ├── 08_train_models.py
    ├── 09_validate_stability.py
    ├── 10_error_analysis.py
    ├── 11_shap_explainability.py
    ├── 12_energy_optimization.py
    ├── 13_package_model.py
    ├── 14_validate_deployment_model.py
    ├── 15_test_sample_periods.py
    ├── 16_compress_model.py
    └── config.py
```

---

## 🛠️ Technologies Used

### Programming

* Python 3.13

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Random Forest Regression
* Linear Regression
* Persistence forecasting
* Seasonal Naive forecasting

### Explainability

* SHAP

### Visualization

* Plotly
* Matplotlib

### Deployment

* Streamlit
* Streamlit Cloud
* Joblib

### Version Control

* Git
* GitHub

---

## ▶️ Running Locally

Clone the repository:

```bash
git clone https://github.com/nuthangilokesh31-cpu/SmartEnergy_Nexus.git
cd SmartEnergy_Nexus
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will be available locally through the Streamlit URL shown in the terminal.

---

## 🔬 Reproducibility Pipeline

The numbered scripts represent the main project workflow:

```text
01 → Data inspection
02 → Dataset profiling
03 → Load analysis
04 → Time-series validation
05 → Forecasting dataset creation
06 → Feature engineering
07 → Train/test split
08 → Model training
09 → Stability validation
10 → Error analysis
11 → SHAP explainability
12 → Energy optimization
13 → Model packaging
14 → Deployment validation
15 → Sample period testing
16 → Model compression
```

---

## 📌 Key Project Highlights

* 6,031 customer electricity load series
* 17,520 half-hourly observations
* 30-minute forecasting resolution
* 29 model input features
* Chronological train/test methodology
* Multiple baseline models
* Random Forest forecasting
* Stability validation
* Forecast error analysis
* SHAP explainability
* Peak-demand intelligence
* Energy optimization scenarios
* Deployment model validation
* Model compression
* Interactive Streamlit dashboard
* Cloud deployment

---

## 🔮 Future Enhancements

Potential future improvements include:

* Real-time smart-meter integration
* Weather and temperature features
* Building occupancy information
* Multi-building forecasting
* Probabilistic forecasting
* Automated model retraining
* Online drift monitoring
* Carbon-emission forecasting
* Renewable-energy integration
* Battery-storage optimization
* Real-time alerting
* Advanced demand-response strategies

---

## 👨‍💻 Project

**SmartEnergy Nexus**

An AI-powered energy forecasting, explainability, and optimization platform for smart-building energy intelligence.

Built with Python, Machine Learning, Explainable AI, and Streamlit.
