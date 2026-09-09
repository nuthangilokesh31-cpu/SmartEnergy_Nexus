from pathlib import Path
import sys

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SGSC_FILE, PROCESSED_DATA_DIR


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("FORECASTING DATASET CREATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------

    if not SGSC_FILE.exists():
        print("\nERROR: Raw dataset was not found.")
        print(SGSC_FILE)
        return

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DATA_DIR /
        "aggregate_energy.csv"
    )

    print("\nInput dataset:")
    print(SGSC_FILE)

    print("\nOutput dataset:")
    print(output_file)

    # ---------------------------------------------------------
    # Load raw dataset
    # ---------------------------------------------------------

    print("\nLoading raw SGSC dataset...")

    df = pd.read_csv(
        SGSC_FILE
    )

    print(
        f"Rows loaded    : {len(df):,}"
    )

    print(
        f"Columns loaded : {len(df.columns):,}"
    )

    # ---------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------

    timestamp_column = df.columns[0]

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Customer load columns
    # ---------------------------------------------------------

    load_columns = df.columns[1:]

    print(
        f"\nCustomer series : {len(load_columns):,}"
    )

    # ---------------------------------------------------------
    # Calculate aggregate demand
    # ---------------------------------------------------------

    print(
        "\nCalculating aggregate demand..."
    )

    aggregate_load = df[load_columns].sum(
        axis=1
    )

    # ---------------------------------------------------------
    # Create processed dataset
    # ---------------------------------------------------------

    processed = pd.DataFrame(
        {
            "timestamp": df[timestamp_column],
            "total_load": aggregate_load
        }
    )

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    processed = processed.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Data-quality checks
    # ---------------------------------------------------------

    print(
        "\nRunning data-quality checks..."
    )

    missing_timestamps = (
        processed["timestamp"]
        .isna()
        .sum()
    )

    missing_load = (
        processed["total_load"]
        .isna()
        .sum()
    )

    duplicate_timestamps = (
        processed["timestamp"]
        .duplicated()
        .sum()
    )

    negative_load = (
        processed["total_load"] < 0
    ).sum()

    print(
        f"Missing timestamps : {missing_timestamps:,}"
    )

    print(
        f"Missing load       : {missing_load:,}"
    )

    print(
        f"Duplicate timestamps: "
        f"{duplicate_timestamps:,}"
    )

    print(
        f"Negative load values: "
        f"{negative_load:,}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print(
        "\nSaving processed dataset..."
    )

    processed.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FORECASTING DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nRows          : {len(processed):,}"
    )

    print(
        f"Columns       : {len(processed.columns):,}"
    )

    print(
        f"Start         : "
        f"{processed['timestamp'].min()}"
    )

    print(
        f"End           : "
        f"{processed['timestamp'].max()}"
    )

    print(
        f"Average load  : "
        f"{processed['total_load'].mean():,.4f}"
    )

    print(
        f"Minimum load  : "
        f"{processed['total_load'].min():,.4f}"
    )

    print(
        f"Maximum load  : "
        f"{processed['total_load'].max():,.4f}"
    )

    print(
        f"\nSaved to:"
    )

    print(output_file)

    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()