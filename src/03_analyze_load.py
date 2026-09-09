from pathlib import Path
import sys

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SGSC_FILE


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("LOAD ANALYSIS")
    print("=" * 70)

    print("\nLoading SGSC dataset...")

    df = pd.read_csv(SGSC_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    # ---------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------

    timestamp_column = df.columns[0]

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column]
    )

    df = df.set_index(timestamp_column)

    # ---------------------------------------------------------
    # Customer/load columns
    # ---------------------------------------------------------

    load_columns = df.columns

    print("\nNumber of customer load series:")
    print(f"{len(load_columns):,}")

    # ---------------------------------------------------------
    # Overall statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL LOAD STATISTICS")
    print("=" * 70)

    total_load = df[load_columns].sum(axis=1)

    average_load = df[load_columns].mean(axis=1)

    minimum_load = df[load_columns].min(axis=1)

    maximum_load = df[load_columns].max(axis=1)

    std_load = df[load_columns].std(axis=1)

    print(f"\nAverage total load : {total_load.mean():,.4f}")
    print(f"Minimum total load : {total_load.min():,.4f}")
    print(f"Maximum total load : {total_load.max():,.4f}")
    print(f"Std total load     : {total_load.std():,.4f}")

    # ---------------------------------------------------------
    # Peak demand
    # ---------------------------------------------------------

    peak_timestamp = total_load.idxmax()
    peak_value = total_load.max()

    print("\n" + "=" * 70)
    print("PEAK LOAD")
    print("=" * 70)

    print(f"\nPeak timestamp : {peak_timestamp}")
    print(f"Peak load      : {peak_value:,.4f}")

    # ---------------------------------------------------------
    # Lowest demand
    # ---------------------------------------------------------

    minimum_timestamp = total_load.idxmin()
    minimum_value = total_load.min()

    print("\n" + "=" * 70)
    print("LOWEST LOAD")
    print("=" * 70)

    print(f"\nLowest timestamp : {minimum_timestamp}")
    print(f"Lowest load      : {minimum_value:,.4f}")

    # ---------------------------------------------------------
    # Daily statistics
    # ---------------------------------------------------------

    daily_load = total_load.resample("D").sum()

    print("\n" + "=" * 70)
    print("DAILY ENERGY SUMMARY")
    print("=" * 70)

    print(f"\nNumber of days : {len(daily_load):,}")
    print(f"Average daily value : {daily_load.mean():,.4f}")
    print(f"Maximum daily value : {daily_load.max():,.4f}")
    print(f"Minimum daily value : {daily_load.min():,.4f}")

    # ---------------------------------------------------------
    # Hourly pattern
    # ---------------------------------------------------------

    hourly_profile = total_load.groupby(
        total_load.index.hour
    ).mean()

    highest_hour = hourly_profile.idxmax()
    highest_hour_value = hourly_profile.max()

    lowest_hour = hourly_profile.idxmin()
    lowest_hour_value = hourly_profile.min()

    print("\n" + "=" * 70)
    print("HOURLY CONSUMPTION PATTERN")
    print("=" * 70)

    print(
        f"\nHighest average hour : "
        f"{highest_hour:02d}:00"
    )

    print(
        f"Highest hourly load  : "
        f"{highest_hour_value:,.4f}"
    )

    print(
        f"\nLowest average hour  : "
        f"{lowest_hour:02d}:00"
    )

    print(
        f"Lowest hourly load   : "
        f"{lowest_hour_value:,.4f}"
    )

    # ---------------------------------------------------------
    # Weekday/weekend pattern
    # ---------------------------------------------------------

    weekday_profile = total_load.groupby(
        total_load.index.dayofweek
    ).mean()

    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    print("\n" + "=" * 70)
    print("DAY-OF-WEEK PATTERN")
    print("=" * 70)

    for day_number, value in weekday_profile.items():

        print(
            f"{weekday_names[day_number]:>10}: "
            f"{value:,.4f}"
        )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()