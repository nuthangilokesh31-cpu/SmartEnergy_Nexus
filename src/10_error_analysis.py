from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, FIGURE_DIR


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("FORECAST ERROR ANALYSIS")
    print("=" * 70)

    prediction_file = (
        PROCESSED_DATA_DIR /
        "test_predictions.csv"
    )

    if not prediction_file.exists():
        print("\nERROR: test_predictions.csv not found.")
        return

    print("\nLoading predictions...")

    df = pd.read_csv(
        prediction_file,
        parse_dates=["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Error calculations
    # ---------------------------------------------------------

    df["error"] = (
        df["actual_load"] -
        df["predicted_load"]
    )

    df["absolute_error"] = (
        df["error"].abs()
    )

    df["squared_error"] = (
        df["error"] ** 2
    )

    # Avoid division problems
    df["absolute_percentage_error"] = (
        df["absolute_error"] /
        df["actual_load"].replace(0, np.nan)
        * 100
    )

    # Time features
    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.day_name()
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    mae = df["absolute_error"].mean()

    rmse = np.sqrt(
        df["squared_error"].mean()
    )

    mape = (
        df["absolute_percentage_error"]
        .mean()
    )

    bias = df["error"].mean()

    print("\n" + "=" * 70)
    print("OVERALL ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"\nMean Absolute Error : {mae:,.4f}"
    )

    print(
        f"Root Mean Squared Error: {rmse:,.4f}"
    )

    print(
        f"Mean Absolute Percentage Error: {mape:,.4f}%"
    )

    print(
        f"Forecast Bias: {bias:,.4f}"
    )

    if bias > 0:
        print(
            "\nInterpretation: the model tends to UNDER-predict."
        )
    elif bias < 0:
        print(
            "\nInterpretation: the model tends to OVER-predict."
        )
    else:
        print(
            "\nInterpretation: no average directional bias."
        )

    # ---------------------------------------------------------
    # Hourly errors
    # ---------------------------------------------------------

    hourly_error = (
        df.groupby("hour")
        .agg(
            mean_actual=("actual_load", "mean"),
            mean_predicted=("predicted_load", "mean"),
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda x: np.sqrt(x.mean()))
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("ERROR BY HOUR")
    print("=" * 70)

    print(
        hourly_error.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    # ---------------------------------------------------------
    # Day-of-week errors
    # ---------------------------------------------------------

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_error = (
        df.groupby("day_of_week")
        .agg(
            mean_actual=("actual_load", "mean"),
            mean_predicted=("predicted_load", "mean"),
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda x: np.sqrt(x.mean()))
        )
        .reindex(weekday_order)
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("ERROR BY DAY OF WEEK")
    print("=" * 70)

    print(
        weekday_error.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    # ---------------------------------------------------------
    # Highest error observations
    # ---------------------------------------------------------

    top_errors = (
        df.nlargest(
            20,
            "absolute_error"
        )[
            [
                "timestamp",
                "actual_load",
                "predicted_load",
                "error",
                "absolute_error"
            ]
        ]
    )

    print("\n" + "=" * 70)
    print("TOP 20 FORECAST ERRORS")
    print("=" * 70)

    print(
        top_errors.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    # ---------------------------------------------------------
    # High-demand error analysis
    # ---------------------------------------------------------

    peak_threshold = (
        df["actual_load"]
        .quantile(0.90)
    )

    peak_periods = df[
        df["actual_load"] >= peak_threshold
    ]

    normal_periods = df[
        df["actual_load"] < peak_threshold
    ]

    peak_mae = (
        peak_periods["absolute_error"]
        .mean()
    )

    normal_mae = (
        normal_periods["absolute_error"]
        .mean()
    )

    print("\n" + "=" * 70)
    print("HIGH-DEMAND ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"\n90th percentile demand threshold:"
        f" {peak_threshold:,.4f}"
    )

    print(
        f"\nHigh-demand observations:"
        f" {len(peak_periods):,}"
    )

    print(
        f"High-demand MAE:"
        f" {peak_mae:,.4f}"
    )

    print(
        f"\nNormal-demand observations:"
        f" {len(normal_periods):,}"
    )

    print(
        f"Normal-demand MAE:"
        f" {normal_mae:,.4f}"
    )

    # ---------------------------------------------------------
    # Save detailed errors
    # ---------------------------------------------------------

    error_file = (
        PROCESSED_DATA_DIR /
        "forecast_error_analysis.csv"
    )

    df.to_csv(
        error_file,
        index=False
    )

    hourly_file = (
        PROCESSED_DATA_DIR /
        "hourly_error_analysis.csv"
    )

    hourly_error.to_csv(
        hourly_file,
        index=False
    )

    weekday_file = (
        PROCESSED_DATA_DIR /
        "weekday_error_analysis.csv"
    )

    weekday_error.to_csv(
        weekday_file,
        index=False
    )

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Actual vs predicted
    plt.figure(figsize=(14, 6))

    plt.plot(
        df["timestamp"],
        df["actual_load"],
        label="Actual"
    )

    plt.plot(
        df["timestamp"],
        df["predicted_load"],
        label="Predicted"
    )

    plt.title(
        "Actual vs Predicted Energy Demand"
    )

    plt.xlabel("Time")
    plt.ylabel("Energy Demand")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR /
        "actual_vs_predicted.png",
        dpi=150
    )

    plt.close()

    # Error over time
    plt.figure(figsize=(14, 6))

    plt.plot(
        df["timestamp"],
        df["error"]
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.title(
        "Forecast Error Over Time"
    )

    plt.xlabel("Time")
    plt.ylabel("Actual - Predicted")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR /
        "forecast_error_over_time.png",
        dpi=150
    )

    plt.close()

    # Hourly MAE
    plt.figure(figsize=(12, 6))

    plt.bar(
        hourly_error["hour"],
        hourly_error["MAE"]
    )

    plt.title(
        "Forecast MAE by Hour"
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("MAE")

    plt.xticks(
        range(24)
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR /
        "hourly_forecast_mae.png",
        dpi=150
    )

    plt.close()

    print("\nSaved analysis files:")

    print(
        f"\n  {error_file}"
    )

    print(
        f"\n  {hourly_file}"
    )

    print(
        f"\n  {weekday_file}"
    )

    print("\nSaved figures:")

    print(
        f"\n  {FIGURE_DIR / 'actual_vs_predicted.png'}"
    )

    print(
        f"\n  {FIGURE_DIR / 'forecast_error_over_time.png'}"
    )

    print(
        f"\n  {FIGURE_DIR / 'hourly_forecast_mae.png'}"
    )

    print("\n" + "=" * 70)
    print("FORECAST ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
