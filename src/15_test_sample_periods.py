from pathlib import Path
import pandas as pd
import numpy as np


print("=" * 70)
print("SMARTENERGY NEXUS")
print("STEP 15 - SAMPLE PERIOD PREDICTION TESTING")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sample_period_tests.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sample_period_test_summary.csv"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading test predictions...")

df = pd.read_csv(PREDICTIONS_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

required_columns = [
    "timestamp",
    "actual_load",
    "predicted_load",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing required columns: {missing_columns}"
    )


df = df.sort_values("timestamp").reset_index(drop=True)


print(f"Rows available: {len(df):,}")
print(f"Start: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")


# ============================================================
# CREATE ERROR METRICS
# ============================================================

df["error"] = (
    df["actual_load"] - df["predicted_load"]
)

df["absolute_error"] = (
    df["error"].abs()
)

df["percentage_error"] = (
    df["absolute_error"]
    / df["actual_load"].replace(0, np.nan)
    * 100
)


# ============================================================
# DEMAND CLASSIFICATION
# ============================================================

p90 = df["actual_load"].quantile(0.90)
p95 = df["actual_load"].quantile(0.95)


def classify_demand(load):
    if load >= p95:
        return "Very High Demand"
    elif load >= p90:
        return "High Demand"
    else:
        return "Normal Demand"


df["demand_class"] = df["actual_load"].apply(
    classify_demand
)


# ============================================================
# SELECT REPRESENTATIVE SAMPLE PERIODS
# ============================================================

samples = []


def add_nearest_sample(label, target_time=None, target_value=None):

    if target_time is not None:

        differences = (
            (df["timestamp"] - target_time)
            .abs()
        )

    elif target_value is not None:

        differences = (
            (df["actual_load"] - target_value)
            .abs()
        )

    else:
        return

    index = differences.idxmin()

    row = df.loc[index]

    samples.append(
        {
            "sample": label,
            "timestamp": row["timestamp"],
            "actual_load": row["actual_load"],
            "predicted_load": row["predicted_load"],
            "absolute_error": row["absolute_error"],
            "percentage_error": row["percentage_error"],
            "demand_class": row["demand_class"],
            "hour": row["timestamp"].hour,
            "day_of_week": row["timestamp"].day_name(),
        }
    )


# 1. First available test observation
add_nearest_sample(
    "Test Period Start",
    target_time=df["timestamp"].min(),
)


# 2. Middle of test period
middle_time = (
    df["timestamp"].min()
    + (
        df["timestamp"].max()
        - df["timestamp"].min()
    ) / 2
)

add_nearest_sample(
    "Test Period Middle",
    target_time=middle_time,
)


# 3. Final test observation
add_nearest_sample(
    "Test Period End",
    target_time=df["timestamp"].max(),
)


# 4. Lowest-demand observation
add_nearest_sample(
    "Lowest Demand",
    target_value=df["actual_load"].min(),
)


# 5. Median-demand observation
add_nearest_sample(
    "Median Demand",
    target_value=df["actual_load"].median(),
)


# 6. 90th-percentile demand
add_nearest_sample(
    "High Demand P90",
    target_value=p90,
)


# 7. 95th-percentile demand
add_nearest_sample(
    "Very High Demand P95",
    target_value=p95,
)


# 8. Maximum observed demand
add_nearest_sample(
    "Observed Peak",
    target_value=df["actual_load"].max(),
)


# 9. Morning period
morning = df[
    df["timestamp"].dt.hour.between(6, 10)
]

if not morning.empty:

    index = morning["actual_load"].idxmax()

    row = df.loc[index]

    samples.append(
        {
            "sample": "Morning Peak",
            "timestamp": row["timestamp"],
            "actual_load": row["actual_load"],
            "predicted_load": row["predicted_load"],
            "absolute_error": row["absolute_error"],
            "percentage_error": row["percentage_error"],
            "demand_class": row["demand_class"],
            "hour": row["timestamp"].hour,
            "day_of_week": row["timestamp"].day_name(),
        }
    )


# 10. Evening period
evening = df[
    df["timestamp"].dt.hour.between(17, 20)
]

if not evening.empty:

    index = evening["actual_load"].idxmax()

    row = df.loc[index]

    samples.append(
        {
            "sample": "Evening Peak",
            "timestamp": row["timestamp"],
            "actual_load": row["actual_load"],
            "predicted_load": row["predicted_load"],
            "absolute_error": row["absolute_error"],
            "percentage_error": row["percentage_error"],
            "demand_class": row["demand_class"],
            "hour": row["timestamp"].hour,
            "day_of_week": row["timestamp"].day_name(),
        }
    )


# ============================================================
# CREATE SAMPLE REPORT
# ============================================================

sample_results = pd.DataFrame(samples)

sample_results = (
    sample_results
    .drop_duplicates(subset=["timestamp"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# SUMMARY METRICS
# ============================================================

summary = pd.DataFrame(
    [
        {
            "sample_count": len(sample_results),
            "average_absolute_error":
                sample_results["absolute_error"].mean(),
            "maximum_absolute_error":
                sample_results["absolute_error"].max(),
            "average_percentage_error":
                sample_results["percentage_error"].mean(),
            "maximum_percentage_error":
                sample_results["percentage_error"].max(),
            "samples_within_5_percent":
                (
                    sample_results["percentage_error"] <= 5
                ).sum(),
            "samples_within_10_percent":
                (
                    sample_results["percentage_error"] <= 10
                ).sum(),
        }
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

sample_results.to_csv(
    OUTPUT_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nRepresentative prediction tests:")
print("-" * 70)

for _, row in sample_results.iterrows():

    print(
        f"\n{row['sample']}"
    )

    print(
        f"  Timestamp       : {row['timestamp']}"
    )

    print(
        f"  Actual load     : "
        f"{row['actual_load']:,.4f}"
    )

    print(
        f"  Predicted load  : "
        f"{row['predicted_load']:,.4f}"
    )

    print(
        f"  Absolute error  : "
        f"{row['absolute_error']:,.4f}"
    )

    print(
        f"  Percentage error: "
        f"{row['percentage_error']:.4f}%"
    )

    print(
        f"  Demand class    : "
        f"{row['demand_class']}"
    )


print("\n" + "=" * 70)
print("SAMPLE TEST SUMMARY")
print("=" * 70)

print(
    f"Samples tested: "
    f"{len(sample_results)}"
)

print(
    f"Average absolute error: "
    f"{summary['average_absolute_error'].iloc[0]:,.4f}"
)

print(
    f"Maximum absolute error: "
    f"{summary['maximum_absolute_error'].iloc[0]:,.4f}"
)

print(
    f"Average percentage error: "
    f"{summary['average_percentage_error'].iloc[0]:.4f}%"
)

print(
    f"Maximum percentage error: "
    f"{summary['maximum_percentage_error'].iloc[0]:.4f}%"
)

print(
    f"Samples within 5% error: "
    f"{summary['samples_within_5_percent'].iloc[0]}"
    f"/{len(sample_results)}"
)

print(
    f"Samples within 10% error: "
    f"{summary['samples_within_10_percent'].iloc[0]}"
    f"/{len(sample_results)}"
)


print("\nOutput files:")
print(
    f"  {OUTPUT_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)

print("\n" + "=" * 70)
print("STEP 15 SAMPLE TESTING COMPLETE")
print("=" * 70)
