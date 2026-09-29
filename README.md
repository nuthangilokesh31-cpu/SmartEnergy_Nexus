# ⚡ SmartEnergy Nexus

### Intelligent Energy Consumption Forecasting for Smart Buildings

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit)](https://streamlit.io/)
[![Scikit--learn](https://img.shields.io/badge/ML-Scikit--learn-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://shap.readthedocs.io/)

> An end-to-end AI-powered energy intelligence platform for forecasting electricity demand, analyzing peak-demand behavior, explaining machine-learning predictions, and evaluating energy optimization scenarios.

## 🚀 Live Demo

**Streamlit Cloud:**
https://smartenergynexus.streamlit.app/

The deployed application provides an interactive dashboard for exploring forecasting performance, peak-demand risk, explainable AI insights, optimization scenarios, and data-quality validation.

---

## 🎯 Project Overview

SmartEnergy Nexus addresses the problem of **electricity demand forecasting for smart buildings**.

The system processes high-resolution electricity consumption data and transforms it into a complete machine-learning workflow:

**Raw Energy Data → Data Validation → Aggregation → Feature Engineering → Time-Based Split → Model Training → Stability Validation → Error Analysis → Explainable AI → Peak-Demand Analysis → Energy Optimization → Deployment**

Unlike a simple forecasting model, the project combines predictive, diagnostic, explainable, and prescriptive analytics in one application.

### Core objectives

* Forecast future electricity demand at 30-minute resolution.
* Capture temporal and seasonal consumption patterns.
* Evaluate multiple forecasting approaches.
* Validate model stability using time-based validation.
* Analyze forecast errors and bias.
* Explain model behavior using SHAP.
* Identify peak-demand periods and associated risks.
* Evaluate modeled energy optimization scenarios.
* Deploy the final system as an interactive Streamlit application.

---

## ✨ Key Features

### 📈 Energy Consumption Forecasting

The forecasting pipeline uses engineered temporal and historical features including:

* Hour and minute
* Day of week
* Day of month
* Month
* Day of year
* Week of year
* Weekend indicators
* Month-start/month-end indicators
* Cyclical time representations
* Lagged demand features
* Rolling statistics
* Historical change features

The final model is packaged for deployment using a compressed Joblib artifact.

### 🔍 Explainable AI

The project uses **SHAP-based explainability** to investigate:

* Global feature importance
* Local prediction explanations
* Features contributing to individual forecasts
* Model behavior across the forecasting dataset

This makes the forecasting system more interpretable than a prediction-only pipeline.

### ⚡ Peak-Demand Analysis

The system analyzes:

* High-demand periods
* Peak-demand behavior
* Forecast error during different periods
* Hourly forecasting performance
* Potential peak-risk periods

### 🔧 Energy Optimization

The platform evaluates modeled optimization scenarios and produces structured recommendations based on forecasted demand and peak-demand behavior.

### 🧪 Model Validation

The workflow includes:

* Chronological train/test splitting
* Multiple forecasting baselines
* Time-based stability validation
* Forecast error analysis
* Sample-period testing
* Deployment-model validation

---

## 📊 Dataset

The project uses the **SGSC clustered residential electricity load dataset**.

### Dataset characteristics

| Property             |      Value |
| -------------------- | ---------: |
| Customer load series |      6,031 |
| Time resolution      | 30 minutes |
| Time period          |       2013 |
| Total timestamps     |     17,520 |
| Raw dataset columns  |      6,032 |
| Missing timestamps   |          0 |
| Duplicate timestamps |          0 |
| Time-series gaps     |          0 |

The raw dataset contains thousands of individual customer load series. The project aggregates these series into a single electricity-demand signal for the forecasting workflow.

The raw dataset is intentionally not included in the repository because of its large size.

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────────┐
                 │   SGSC Energy Dataset   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Data Inspection &     │
                 │   Quality Validation    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Aggregate Energy Load   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Feature Engineering     │
                 │ Calendar + Lag +        │
                 │ Rolling Features        │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Chronological Train/Test│
                 │ Split                   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Model Training &         │
                 │ Baseline Comparison      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Stability Validation    │
                 │ + Error Analysis        │
                 └────────────┬────────────┘
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
      ┌──────────────────┐        ┌──────────────────┐
      │ SHAP Explainable │        │ Peak-Demand &    │
      │ AI               │        │ Optimization     │
      └────────┬─────────┘        └────────┬─────────┘
               │                           │
               └─────────────┬─────────────┘
                             ▼
                 ┌─────────────────────────┐
                 │   Streamlit Dashboard   │
                 └─────────────────────────┘
```

---

## 🔬 Machine Learning Approach

The project compares several forecasting approaches.

### Baselines

* Persistence forecasting using the previous observation.
* Seasonal-naive forecasting using the corresponding previous-day interval.

### Machine-learning models

* Linear Regression
* Random Forest Regressor

The model-development workflow evaluates performance using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R²** — Coefficient of Determination
* **MAPE** — Mean Absolute Percentage Error

The final deployment model is stored as:

```text
models/best_forecasting_model_compressed.joblib
```

---

## 📈 Model Performance

The forecasting workflow produced the following representative evaluation results on the held-out test period:

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Persistence       | 143.43 | 185.00 | 0.9715 |
| Seasonal Naive    | 375.70 | 662.43 | 0.6349 |
| Linear Regression |  65.20 |  85.84 | 0.9939 |

The final deployed application reports an overall forecast evaluation of approximately:

* **MAE:** 58.91
* **RMSE:** 89.86
* **MAPE:** 1.73%
* **R²:** approximately 0.993

These metrics should be interpreted in the context of the SGSC dataset, the defined aggregation procedure, feature construction, and chronological evaluation period.

---

## 🧠 Explainable AI

SHAP is used to move beyond:

> "What did the model predict?"

toward:

> "Which features influenced the prediction?"

The application provides both global and local explanations.

### Global explanation

Identifies features that have the greatest influence across the evaluated forecasting dataset.

### Local explanation

Examines the contribution of individual features for selected predictions.

This supports model transparency and helps investigate unexpected forecasts.

---

## ⚡ Peak-Demand Intelligence

Peak-demand analysis examines periods where electricity consumption reaches elevated levels.

The system provides:

* Peak-demand analysis
* Hourly error analysis
* Weekday error analysis
* Forecast-error analysis
* Peak-risk visualization
* Optimization scenarios

This connects forecasting with operational energy-management decisions.

---

## 🔧 Optimization Layer

SmartEnergy Nexus includes a modeled optimization layer that evaluates different energy-management scenarios.

The optimization workflow generates:

* Optimization scenarios
* Recommendations
* Energy-optimization insights
* Peak-demand analysis

The optimization results are presented as **modeled scenarios**, not claims of measured savings from a live building.

---

## 🧪 Validation & Reliability

The project includes multiple validation stages.

### Time-series validation

The source time series was checked for:

* Invalid timestamps
* Duplicate timestamps
* Missing intervals
* Unexpected interval lengths

Validation confirmed a continuous 30-minute time series for the analyzed 2013 period.

### Chronological split

The forecasting dataset uses an **80/20 chronological train/test split** rather than random shuffling.

This prevents future observations from being randomly mixed into the training data.

### Stability validation

The model was evaluated across multiple chronological validation folds to examine performance stability over time.

### Deployment validation

The final packaged model was separately validated before deployment.

---

## 🖥️ Streamlit Dashboard

The deployed application contains seven major sections:

1. **Executive Dashboard**
2. **Energy Forecast**
3. **Peak Demand**
4. **Explainable AI**
5. **Optimization**
6. **Model Performance**
7. **Data Quality**

The dashboard combines model outputs, evaluation metrics, visual analysis, and optimization results into a single interface.

---

## 🛠️ Technology Stack

| Category            | Technologies              |
| ------------------- | ------------------------- |
| Language            | Python                    |
| Data Processing     | Pandas, NumPy             |
| Machine Learning    | Scikit-learn              |
| Explainable AI      | SHAP                      |
| Visualization       | Plotly / Matplotlib       |
| Dashboard           | Streamlit                 |
| Model Serialization | Joblib                    |
| Version Control     | Git + GitHub              |
| Deployment          | Streamlit Community Cloud |

---

## 📁 Project Structure

```text
SmartEnergy_Nexus/
│
├── app.py
├── requirements.txt
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
│       └── optimization_*.csv
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

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/nuthangilokesh31-cpu/SmartEnergy_Nexus.git
cd SmartEnergy_Nexus
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud** and connected to the GitHub repository.

### Deployment configuration

```text
Repository: nuthangilokesh31-cpu/SmartEnergy_Nexus
Branch: main
Main file: app.py
```

### Live application

https://smartenergynexus.streamlit.app/

---

## 📌 Key Results

The completed project demonstrates an end-to-end machine-learning workflow covering:

* Large-scale energy-data processing
* Time-series validation
* Feature engineering
* Chronological model evaluation
* Forecasting
* Stability testing
* Error analysis
* Explainable AI
* Peak-demand intelligence
* Energy optimization scenarios
* Model packaging
* Cloud deployment

The project therefore goes beyond a standalone ML notebook and demonstrates how a forecasting model can be integrated into an interactive decision-support application.

---

## 🔮 Future Improvements

Potential extensions include:

* Multi-step forecasting
* Weather-data integration
* Real-time smart-meter/IoT ingestion
* Building-level personalization
* Online model retraining
* Probabilistic forecasting and prediction intervals
* Automated anomaly detection
* Carbon-emission estimation
* Real-time demand-response integration
* Containerized deployment
* Model monitoring and drift detection

---

## 👨‍💻 Author

**Nuthangi Lokesh**

B.Tech — Artificial Intelligence & Machine Learning

GitHub:
https://github.com/nuthangilokesh31-cpu

---

## 📄 Project Note

This project is developed as an academic/placement-oriented AI/ML project demonstrating an end-to-end energy forecasting and intelligence workflow.

The optimization outputs represent modeled scenarios based on the available dataset and should not be interpreted as measured operational savings from a live building.
