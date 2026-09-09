from pathlib import Path
import sys

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("FEATURE ENGINEERING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Input / output
    # ---------------------------------------------------------

    input_file = (
        PROCESSED_DATA_DIR /
        "aggregate_energy.csv"
    )

    output_file = (
        PROCESSED_DATA_DIR /
        "forecasting_features.csv"
    )

    if not input_file.exists():

        print("\nERROR: Processed dataset not found.")
        print(input_file)
        return

    print("\nInput:")
    print(input_file)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        input_file,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"\nRows loaded: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Calendar features
    # ---------------------------------------------------------

    print("\nCreating calendar features...")

    df["hour"] = df["timestamp"].dt.hour

    df["minute"] = df["timestamp"].dt.minute

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["timestamp"].dt.day
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["day_of_year"] = (
        df["timestamp"].dt.dayofyear
    )

    df["week_of_year"] = (
        df["timestamp"].dt.isocalendar().week.astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_month_start"] = (
        df["timestamp"].dt.is_month_start
    ).astype(int)

    df["is_month_end"] = (
        df["timestamp"].dt.is_month_end
    ).astype(int)

    # ---------------------------------------------------------
    # Cyclical time features
    # ---------------------------------------------------------

    print("Creating cyclical time features...")

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # ---------------------------------------------------------
    # Lag features
    # ---------------------------------------------------------

    print("Creating lag features...")

    # Previous 30 minutes
    df["lag_1"] = (
        df["total_load"].shift(1)
    )

    # Previous hour
    df["lag_2"] = (
        df["total_load"].shift(2)
    )

    # Previous 2 hours
    df["lag_4"] = (
        df["total_load"].shift(4)
    )

    # Previous 6 hours
    df["lag_12"] = (
        df["total_load"].shift(12)
    )

    # Previous 12 hours
    df["lag_24"] = (
        df["total_load"].shift(24)
    )

    # Previous day
    df["lag_48"] = (
        df["total_load"].shift(48)
    )

    # Previous 2 days
    df["lag_96"] = (
        df["total_load"].shift(96)
    )

    # Previous week
    df["lag_336"] = (
        df["total_load"].shift(336)
    )

    # ---------------------------------------------------------
    # Rolling features
    # ---------------------------------------------------------

    print("Creating rolling statistics...")

    # Shift first so the current target is never included.
    shifted_load = df["total_load"].shift(1)

    df["rolling_mean_2"] = (
        shifted_load
        .rolling(window=2)
        .mean()
    )

    df["rolling_mean_4"] = (
        shifted_load
        .rolling(window=4)
        .mean()
    )

    df["rolling_mean_12"] = (
        shifted_load
        .rolling(window=12)
        .mean()
    )

    df["rolling_mean_48"] = (
        shifted_load
        .rolling(window=48)
        .mean()
    )

    df["rolling_std_48"] = (
        shifted_load
        .rolling(window=48)
        .std()
    )

    # ---------------------------------------------------------
    # Recent load change
    # ---------------------------------------------------------

    print("Creating change features...")

    df["load_change_30min"] = (
        df["total_load"] -
        df["lag_1"]
    )

    df["load_change_1hour"] = (
        df["total_load"] -
        df["lag_2"]
    )

    # ---------------------------------------------------------
    # Remove rows with insufficient history
    # ---------------------------------------------------------

    print("\nRemoving rows without sufficient history...")

    before = len(df)

    df = df.dropna().reset_index(
        drop=True
    )

    removed = before - len(df)

    print(
        f"Rows removed: {removed:,}"
    )

    print(
        f"Rows remaining: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Check for remaining missing values
    # ---------------------------------------------------------

    remaining_missing = (
        df.isna().sum().sum()
    )

    print(
        f"\nRemaining missing values: "
        f"{remaining_missing:,}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print("\nSaving feature dataset...")

    df.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nRows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    print(
        "\nFeatures created:"
    )

    for column in df.columns:
        print(f"  - {column}")

    print("\nSaved to:")
    print(output_file)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()