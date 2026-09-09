from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 70)
print("SMARTENERGY NEXUS")
print("ENERGY OPTIMIZATION INSIGHTS")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TEST_PREDICTIONS_FILE = PROCESSED_DIR / "test_predictions.csv"
ERROR_FILE = PROCESSED_DIR / "forecast_error_analysis.csv"
HOURLY_ERROR_FILE = PROCESSED_DIR / "hourly_error_analysis.csv"

OUTPUT_INSIGHTS = PROCESSED_DIR / "energy_optimization_insights.csv"
OUTPUT_PEAK = PROCESSED_DIR / "peak_demand_analysis.csv"
OUTPUT_SCENARIOS = PROCESSED_DIR / "optimization_scenarios.csv"
OUTPUT_RECOMMENDATIONS = PROCESSED_DIR / "optimization_recommendations.csv"

FIGURE_PEAK = FIGURE_DIR / "peak_demand_risk.png"
FIGURE_SCENARIOS = FIGURE_DIR / "optimization_scenarios.png"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
print("\nLoading forecast predictions...")

if not TEST_PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {TEST_PREDICTIONS_FILE}"
    )

pred = pd.read_csv(TEST_PREDICTIONS_FILE)

print(f"Prediction rows: {len(pred):,}")
print(f"Prediction columns: {len(pred.columns)}")

# Normalize timestamp
if "timestamp" not in pred.columns:
    raise ValueError("test_predictions.csv must contain a timestamp column.")

pred["timestamp"] = pd.to_datetime(pred["timestamp"])

# ---------------------------------------------------------
# IDENTIFY ACTUAL / PREDICTED COLUMNS
# ---------------------------------------------------------
actual_candidates = [
    "actual",
    "actual_load",
    "total_load",
    "y_true",
]

predicted_candidates = [
    "predicted",
    "predicted_load",
    "prediction",
    "y_pred",
]

actual_col = next(
    (c for c in actual_candidates if c in pred.columns),
    None
)

predicted_col = next(
    (c for c in predicted_candidates if c in pred.columns),
    None
)

if actual_col is None:
    raise ValueError(
        f"Could not identify actual-load column. "
        f"Available columns: {list(pred.columns)}"
    )

if predicted_col is None:
    raise ValueError(
        f"Could not identify prediction column. "
        f"Available columns: {list(pred.columns)}"
    )

print(f"Actual column    : {actual_col}")
print(f"Prediction column: {predicted_col}")

pred["actual_load"] = pd.to_numeric(pred[actual_col])
pred["predicted_load"] = pd.to_numeric(pred[predicted_col])

# ---------------------------------------------------------
# BASIC FORECAST / DEMAND FEATURES
# ---------------------------------------------------------
pred["hour"] = pred["timestamp"].dt.hour
pred["minute"] = pred["timestamp"].dt.minute
pred["day_of_week"] = pred["timestamp"].dt.day_name()
pred["is_weekend"] = pred["timestamp"].dt.dayofweek >= 5

pred["forecast_error"] = (
    pred["actual_load"] - pred["predicted_load"]
)

pred["absolute_error"] = pred["forecast_error"].abs()

# ---------------------------------------------------------
# DEMAND THRESHOLDS
# ---------------------------------------------------------
p90 = pred["actual_load"].quantile(0.90)
p95 = pred["actual_load"].quantile(0.95)

peak_actual = pred["actual_load"].max()
peak_timestamp = pred.loc[
    pred["actual_load"].idxmax(), "timestamp"
]

average_load = pred["actual_load"].mean()

print("\n" + "=" * 70)
print("DEMAND PROFILE")
print("=" * 70)

print(f"Average test-period load : {average_load:,.4f}")
print(f"90th percentile          : {p90:,.4f}")
print(f"95th percentile          : {p95:,.4f}")
print(f"Observed test peak       : {peak_actual:,.4f}")
print(f"Peak timestamp           : {peak_timestamp}")

# ---------------------------------------------------------
# PEAK DEMAND ANALYSIS
# ---------------------------------------------------------
pred["demand_class"] = np.select(
    [
        pred["actual_load"] >= p95,
        pred["actual_load"] >= p90,
    ],
    [
        "Very High Demand",
        "High Demand",
    ],
    default="Normal Demand"
)

peak_analysis = (
    pred.groupby("demand_class")
    .agg(
        observations=("actual_load", "size"),
        average_actual_load=("actual_load", "mean"),
        maximum_actual_load=("actual_load", "max"),
        average_prediction_error=("forecast_error", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
    )
    .reset_index()
)

print("\nPeak demand classification:")
print(peak_analysis.to_string(index=False))

peak_analysis.to_csv(
    OUTPUT_PEAK,
    index=False
)

# ---------------------------------------------------------
# EVENING RISK
# ---------------------------------------------------------
evening_hours = [17, 18, 19, 20]

evening = pred[pred["hour"].isin(evening_hours)].copy()

if len(evening) > 0:
    evening_average = evening["actual_load"].mean()
    evening_peak = evening["actual_load"].max()
    evening_mae = evening["absolute_error"].mean()
else:
    evening_average = np.nan
    evening_peak = np.nan
    evening_mae = np.nan

print("\nEvening peak window: 17:00-20:59")
print(f"Evening average load : {evening_average:,.4f}")
print(f"Evening maximum load : {evening_peak:,.4f}")
print(f"Evening MAE          : {evening_mae:,.4f}")

# ---------------------------------------------------------
# HYPOTHETICAL PEAK REDUCTION SCENARIOS
# ---------------------------------------------------------
# These are modeled scenarios only.
# They do NOT claim that the building actually achieved
# these reductions.
# ---------------------------------------------------------
scenario_reductions = [0.05, 0.10, 0.15]

scenario_rows = []

peak_mask = pred["actual_load"] >= p90

peak_load = pred.loc[peak_mask, "actual_load"]

for reduction in scenario_reductions:

    reduced_peak = peak_load * (1 - reduction)

    original_peak = peak_load.max()
    scenario_peak = reduced_peak.max()

    peak_reduction_absolute = (
        original_peak - scenario_peak
    )

    average_peak_load = peak_load.mean()
    scenario_average_peak = reduced_peak.mean()

    scenario_rows.append({
        "scenario": f"{int(reduction * 100)}% peak reduction",
        "reduction_fraction": reduction,
        "peak_observations": len(peak_load),
        "baseline_peak_load": original_peak,
        "modeled_peak_load": scenario_peak,
        "modeled_peak_reduction": peak_reduction_absolute,
        "baseline_average_high_demand": average_peak_load,
        "modeled_average_high_demand": scenario_average_peak,
        "modeled_average_reduction": (
            average_peak_load - scenario_average_peak
        ),
        "interpretation": (
            "Hypothetical scenario assuming flexible loads "
            "can be reduced or shifted during high-demand periods."
        )
    })

scenario_df = pd.DataFrame(scenario_rows)

print("\n" + "=" * 70)
print("HYPOTHETICAL PEAK REDUCTION SCENARIOS")
print("=" * 70)

print(
    scenario_df[
        [
            "scenario",
            "baseline_peak_load",
            "modeled_peak_load",
            "modeled_peak_reduction",
            "modeled_average_reduction",
        ]
    ].to_string(index=False)
)

scenario_df.to_csv(
    OUTPUT_SCENARIOS,
    index=False
)

# ---------------------------------------------------------
# LOAD-SHIFTING SCENARIO
# ---------------------------------------------------------
# Shift 10% of high-demand load away from the high-demand
# window. Total modeled energy is assumed unchanged.
# This is a scenario, not an observed intervention.
# ---------------------------------------------------------
shift_fraction = 0.10

high_demand_load = pred.loc[
    peak_mask, "actual_load"
]

shifted_amount = high_demand_load * shift_fraction

load_shift_total = shifted_amount.sum()
load_shift_peak_reduction = shifted_amount.max()

load_shift_summary = {
    "scenario": "10% high-demand load shift",
    "high_demand_observations": len(high_demand_load),
    "shift_fraction": shift_fraction,
    "modeled_load_shift_sum": load_shift_total,
    "maximum_single_interval_shift": load_shift_peak_reduction,
    "energy_conservation_assumption": True,
    "interpretation": (
        "Models shifting 10% of load during high-demand intervals "
        "to lower-demand periods. Total modeled energy is unchanged; "
        "only the timing of flexible demand is changed."
    )
}

print("\n" + "=" * 70)
print("LOAD-SHIFTING SCENARIO")
print("=" * 70)

for key, value in load_shift_summary.items():
    print(f"{key}: {value}")

# ---------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------
recommendations = []

recommendations.append({
    "priority": 1,
    "recommendation": "Activate peak-demand early warning",
    "trigger": f"Forecasted load approaches the 90th percentile ({p90:,.2f})",
    "action": (
        "Notify the operator before a high-demand period so "
        "flexible loads can be reviewed."
    ),
    "evidence": (
        "High-demand periods have materially higher forecasting error "
        "than normal-demand periods."
    ),
    "status": "Decision support only"
})

recommendations.append({
    "priority": 2,
    "recommendation": "Review flexible loads during evening peak",
    "trigger": "17:00-20:59",
    "action": (
        "Consider shifting non-critical flexible demand toward "
        "lower-demand periods."
    ),
    "evidence": (
        f"Observed forecast MAE in the evening window: "
        f"{evening_mae:,.2f}"
    ),
    "status": "Modeled recommendation"
})

recommendations.append({
    "priority": 3,
    "recommendation": "Prioritize high-demand intervals for optimization",
    "trigger": f"Actual demand >= P90 ({p90:,.2f})",
    "action": (
        "Evaluate HVAC, charging, storage, or other flexible loads "
        "for potential scheduling changes."
    ),
    "evidence": (
        "The high-demand regime represents the most operationally "
        "important forecast-risk period."
    ),
    "status": "Decision support only"
})

recommendations.append({
    "priority": 4,
    "recommendation": "Use scenario simulation before intervention",
    "trigger": "Potential peak-management action",
    "action": (
        "Compare 5%, 10%, and 15% hypothetical demand-reduction "
        "scenarios before selecting an operational strategy."
    ),
    "evidence": (
        "Scenario results quantify potential peak exposure without "
        "claiming observed savings."
    ),
    "status": "What-if analysis"
})

recommendations.append({
    "priority": 5,
    "recommendation": "Monitor forecast uncertainty around peaks",
    "trigger": "High predicted demand or rapid load change",
    "action": (
        "Use the forecast as decision support and combine it with "
        "real-time operational information before acting."
    ),
    "evidence": (
        "Forecast error is substantially larger during high-demand "
        "periods than normal-demand periods."
    ),
    "status": "Decision support only"
})

recommendations_df = pd.DataFrame(recommendations)

recommendations_df.to_csv(
    OUTPUT_RECOMMENDATIONS,
    index=False
)

# ---------------------------------------------------------
# CONSOLIDATED INSIGHTS
# ---------------------------------------------------------
insights = [
    {
        "insight_type": "Peak Demand",
        "metric": "Observed test-period peak",
        "value": peak_actual,
        "unit": "load units",
        "interpretation": (
            f"Maximum observed test-period demand occurred at "
            f"{peak_timestamp}."
        )
    },
    {
        "insight_type": "Peak Demand",
        "metric": "90th percentile threshold",
        "value": p90,
        "unit": "load units",
        "interpretation": (
            "Demand above this threshold is classified as high-demand "
            "for optimization analysis."
        )
    },
    {
        "insight_type": "Peak Demand",
        "metric": "95th percentile threshold",
        "value": p95,
        "unit": "load units",
        "interpretation": (
            "Demand above this threshold is classified as very high "
            "demand."
        )
    },
    {
        "insight_type": "Forecast Risk",
        "metric": "Evening MAE",
        "value": evening_mae,
        "unit": "load units",
        "interpretation": (
            "Evening periods are a priority area for forecast-aware "
            "peak management."
        )
    },
    {
        "insight_type": "Optimization",
        "metric": "Modeled 10% peak reduction",
        "value": scenario_df.loc[
            scenario_df["reduction_fraction"] == 0.10,
            "modeled_peak_reduction"
        ].iloc[0],
        "unit": "load units",
        "interpretation": (
            "Hypothetical reduction in the maximum high-demand load "
            "if a 10% reduction were achievable."
        )
    },
    {
        "insight_type": "Optimization",
        "metric": "Modeled high-demand load shift",
        "value": load_shift_total,
        "unit": "load units",
        "interpretation": (
            "Hypothetical quantity shifted from high-demand intervals "
            "under the 10% load-shifting scenario."
        )
    },
]

insights_df = pd.DataFrame(insights)

insights_df.to_csv(
    OUTPUT_INSIGHTS,
    index=False
)

# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # Peak demand risk
    plt.figure(figsize=(12, 5))
    plt.plot(
        pred["timestamp"],
        pred["actual_load"],
        label="Actual Load",
        linewidth=1
    )
    plt.axhline(
        p90,
        linestyle="--",
        label="90th Percentile"
    )
    plt.axhline(
        p95,
        linestyle="--",
        label="95th Percentile"
    )
    plt.title("Peak Demand Risk Analysis")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PEAK, dpi=150)
    plt.close()

    # Optimization scenarios
    plt.figure(figsize=(9, 5))
    plt.bar(
        scenario_df["scenario"],
        scenario_df["modeled_peak_load"]
    )
    plt.axhline(
        peak_actual,
        linestyle="--",
        label="Baseline Peak"
    )
    plt.title("Hypothetical Peak Reduction Scenarios")
    plt.xlabel("Scenario")
    plt.ylabel("Modeled Peak Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_SCENARIOS, dpi=150)
    plt.close()

    print("\nFigures created successfully.")

except Exception as exc:
    print(f"\nVisualization warning: {exc}")

# ---------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ENERGY OPTIMIZATION ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved files:")

print(f"\n  {OUTPUT_INSIGHTS}")
print(f"\n  {OUTPUT_PEAK}")
print(f"\n  {OUTPUT_SCENARIOS}")
print(f"\n  {OUTPUT_RECOMMENDATIONS}")

print("\nSaved figures:")

print(f"\n  {FIGURE_PEAK}")
print(f"\n  {FIGURE_SCENARIOS}")

print("\nIMPORTANT:")
print(
    "All optimization reductions are hypothetical modeled scenarios. "
    "They are NOT measured energy savings."
)

print("\n" + "=" * 70)
