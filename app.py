from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# SMARTENERGY NEXUS
# AI-POWERED ENERGY FORECASTING & INTELLIGENCE PLATFORM
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_forecasting_model_compressed.joblib"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_predictions.csv"
)

MODEL_COMPARISON_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_comparison.csv"
)

ERROR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_error_analysis.csv"
)

SHAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shap_feature_importance.csv"
)

OPTIMIZATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "optimization_scenarios.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartEnergy Nexus",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metadata():
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_FILE)


@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_FILE)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_FILE)


@st.cache_data
def load_error_analysis():
    return pd.read_csv(ERROR_FILE)


@st.cache_data
def load_shap():
    return pd.read_csv(SHAP_FILE)


@st.cache_data
def load_optimization():
    return pd.read_csv(OPTIMIZATION_FILE)


model = load_model()
metadata = load_metadata()
test = load_test_data()
predictions = load_predictions()
model_comparison = load_model_comparison()
error_analysis = load_error_analysis()
shap_data = load_shap()
optimization = load_optimization()


# ============================================================
# PREPARE DATA
# ============================================================

features = metadata["features"]["columns"]

if "timestamp" in test.columns:
    test["timestamp"] = pd.to_datetime(test["timestamp"])

if "timestamp" in predictions.columns:
    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"]
    )

if "timestamp" in error_analysis.columns:
    error_analysis["timestamp"] = pd.to_datetime(
        error_analysis["timestamp"]
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ SmartEnergy Nexus")

st.sidebar.markdown(
    """
**AI-Powered Predictive, Explainable
& Prescriptive Energy Intelligence**
"""
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Energy Forecast",
        "Peak Demand",
        "Explainable AI",
        "Optimization",
        "Model Performance",
        "Data Quality",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Dataset: SGSC clustered residential electricity load profiles"
)

st.sidebar.caption(
    "Resolution: 30-minute intervals"
)

st.sidebar.caption(
    "Year: 2013"
)


# ============================================================
# HEADER
# ============================================================

st.title("⚡ SmartEnergy Nexus")

st.markdown(
    """
### Intelligent Energy Consumption Forecasting
Predict demand, understand model decisions, identify peak-risk
periods, and evaluate modeled energy optimization scenarios.
"""
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header("Executive Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customer Load Series",
        "6,031",
    )

    col2.metric(
        "Forecast Resolution",
        "30 min",
    )

    col3.metric(
        "Forecast MAE",
        "58.91",
    )

    col4.metric(
        "Forecast R²",
        "0.9933",
    )

    st.divider()

    left, right = st.columns([2, 1])

    with left:

        st.subheader("Actual vs Predicted Energy Demand")

        if (
            "actual_load" in predictions.columns
            and "predicted_load" in predictions.columns
        ):

            chart_data = predictions.copy()

            fig = px.line(
                chart_data,
                x="timestamp",
                y=[
                    "actual_load",
                    "predicted_load",
                ],
                labels={
                    "value": "Load",
                    "timestamp": "Time",
                    "variable": "Series",
                },
            )

            fig.update_layout(
                height=450,
                legend_title_text="",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    with right:

        st.subheader("Model Status")

        st.success("Forecasting model loaded")

        st.info(
            f"Model: {type(model).__name__}"
        )

        st.info(
            f"Features: {len(features)}"
        )

        st.info(
            "Validation: 4 chronological folds"
        )

        st.warning(
            "Optimization values are modeled scenarios, "
            "not measured savings."
        )

    st.divider()

    st.subheader("Energy Intelligence Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Test Load",
        f"{predictions['actual_load'].mean():,.2f}",
    )

    c2.metric(
        "Observed Test Peak",
        f"{predictions['actual_load'].max():,.2f}",
    )

    c3.metric(
        "90th Percentile",
        f"{predictions['actual_load'].quantile(0.90):,.2f}",
    )


# ============================================================
# ENERGY FORECAST
# ============================================================

elif page == "Energy Forecast":

    st.header("🔮 Energy Forecast")

    st.write(
        "Select a historical test-period observation and "
        "run the saved Random Forest forecasting model."
    )

    selected_index = st.slider(
        "Test observation",
        min_value=0,
        max_value=len(test) - 1,
        value=0,
    )

    selected_row = test.iloc[[selected_index]]

    X_selected = selected_row[features]

    forecast = model.predict(X_selected)[0]

    timestamp = selected_row["timestamp"].iloc[0]

    actual = (
        selected_row["total_load"].iloc[0]
        if "total_load" in selected_row.columns
        else None
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Timestamp",
        str(timestamp),
    )

    c2.metric(
        "Forecast Load",
        f"{forecast:,.2f}",
    )

    if actual is not None:

        c3.metric(
            "Actual Load",
            f"{actual:,.2f}",
            delta=f"{forecast - actual:,.2f}",
        )

    st.divider()

    st.subheader("Forecast Context")

    context_columns = [
        "hour",
        "day_of_week",
        "month",
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_48",
        "rolling_mean_48",
    ]

    available_context = [
        column
        for column in context_columns
        if column in selected_row.columns
    ]

    st.dataframe(
        selected_row[
            available_context
        ].T.rename(
            columns={
                selected_row.index[0]: "Value"
            }
        ),
        width="stretch",
    )

    st.caption(
        "This forecast uses the same 29-feature schema "
        "used during model training."
    )


# ============================================================
# PEAK DEMAND
# ============================================================

elif page == "Peak Demand":

    st.header("🚨 Peak Demand Intelligence")

    actual_peak = predictions["actual_load"].max()

    peak_row = predictions.loc[
        predictions["actual_load"].idxmax()
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Observed Test Peak",
        f"{actual_peak:,.2f}",
    )

    c2.metric(
        "Peak Timestamp",
        str(peak_row["timestamp"]),
    )

    evening = predictions[
        predictions["timestamp"].dt.hour.between(
            17,
            20,
        )
    ]

    c3.metric(
        "Evening Average",
        f"{evening['actual_load'].mean():,.2f}",
    )

    st.divider()

    st.subheader("Demand Profile")

    hourly = (
        predictions.assign(
            hour=predictions["timestamp"].dt.hour
        )
        .groupby("hour")["actual_load"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        hourly,
        x="hour",
        y="actual_load",
        labels={
            "hour": "Hour of Day",
            "actual_load": "Average Load",
        },
    )

    fig.update_layout(height=450)

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.subheader("Peak Risk Interpretation")

    st.warning(
        """
The model's error analysis shows that evening demand is
the most difficult forecasting period. This makes the
17:00–20:59 window particularly important for peak-risk
monitoring and energy-management decisions.
"""
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "Explainable AI":

    st.header("🧠 Explainable AI")

    st.write(
        "SHAP-based feature importance explains which input "
        "features most influenced the Random Forest predictions."
    )

    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=min(20, len(shap_data)),
        value=10,
    )

    display_shap = shap_data.head(top_n).copy()

    fig = px.bar(
        display_shap.sort_values(
            "mean_abs_shap"
        ),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        labels={
            "mean_abs_shap": "Mean |SHAP value|",
            "feature": "Feature",
        },
    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.subheader("Interpretation")

    st.info(
        """
The immediate previous load (`lag_1`) is the dominant
predictive feature. Time-of-day features and short-term
historical lags provide additional forecasting context.

SHAP explains model behavior; it does not establish
causal relationships.
"""
    )

    st.dataframe(
        shap_data.head(20),
        width="stretch",
    )


# ============================================================
# OPTIMIZATION
# ============================================================

elif page == "Optimization":

    st.header("⚙️ Energy Optimization Intelligence")

    st.warning(
        """
All optimization values shown here are hypothetical
modeled scenarios. They are not measured operational
savings and should not be interpreted as guaranteed
reductions.
"""
    )

    if not optimization.empty:

        st.subheader("Modeled Peak-Reduction Scenarios")

        scenario_columns = [
            column
            for column in [
                "scenario",
                "peak_reduction",
                "average_high_demand_reduction",
            ]
            if column in optimization.columns
        ]

        st.dataframe(
            optimization[scenario_columns],
            width="stretch",
        )

    st.divider()

    st.subheader("Optimization Principle")

    st.markdown(
        """
The platform evaluates hypothetical actions such as:

- reducing flexible demand during high-demand periods
- shifting flexible loads away from peak intervals
- prioritizing evening peak-risk periods
- estimating the effect of different reduction assumptions

The current implementation is a **decision-support simulator**;
it does not directly control building equipment.
"""
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("📊 Model Performance")

    st.subheader("Model Comparison")

    st.dataframe(
        model_comparison,
        width="stretch",
    )

    st.divider()

    st.subheader("Selected Model")

    st.success(
        "Random Forest was selected as the primary forecasting model "
        "because it achieved the lowest test-period MAE."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Test MAE",
        "58.9054",
    )

    c2.metric(
        "Test RMSE",
        "89.8589",
    )

    c3.metric(
        "Test R²",
        "0.9933",
    )

    st.divider()

    st.subheader("High-Demand Forecast Risk")

    st.info(
        """
Forecast errors increase substantially during high-demand
periods. Therefore, overall accuracy should not be the only
criterion used for operational decision-making.
"""
    )

    if not error_analysis.empty:

        st.dataframe(
            error_analysis,
            width="stretch",
        )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.header("✅ Data Quality & Governance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Raw Time Records",
        "17,520",
    )

    c2.metric(
        "Customer Series",
        "6,031",
    )

    c3.metric(
        "Missing Timestamps",
        "0",
    )

    c4.metric(
        "Duplicate Timestamps",
        "0",
    )

    st.divider()

    st.subheader("Dataset Coverage")

    st.write(
        """
The selected SGSC release contains 30-minute electricity
load observations for 6,031 residential customer load
series across calendar year 2013.
"""
    )

    st.subheader("Model Governance")

    st.markdown(
        """
**Forecasting model:** Random Forest Regressor

**Feature count:** 29

**Validation:** chronological time-series validation

**Explainability:** SHAP TreeExplainer

**Optimization:** hypothetical scenario modeling

**Deployment status:** local Streamlit application

**Important limitation:** modeled optimization results
must not be presented as measured energy savings.
"""
    )

    st.divider()

    st.caption(
        "SmartEnergy Nexus — Intelligent Energy Consumption "
        "Forecasting for Smart Buildings"
    )




