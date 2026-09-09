from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SGSC_FILE


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("TIME SERIES VALIDATION")
    print("=" * 70)

    print("\nLoading timestamps...")

    df = pd.read_csv(
        SGSC_FILE,
        usecols=[0]
    )

    timestamp_column = df.columns[0]

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    timestamps = df[timestamp_column]

    print(f"\nTotal timestamps : {len(timestamps):,}")

    # ---------------------------------------------------------
    # Invalid timestamps
    # ---------------------------------------------------------

    invalid_count = timestamps.isna().sum()

    print(f"Invalid timestamps: {invalid_count:,}")

    # ---------------------------------------------------------
    # Duplicate timestamps
    # ---------------------------------------------------------

    duplicate_count = timestamps.duplicated().sum()

    print(f"Duplicate timestamps: {duplicate_count:,}")

    # ---------------------------------------------------------
    # Sort timestamps
    # ---------------------------------------------------------

    timestamps = timestamps.sort_values()

    # ---------------------------------------------------------
    # Calculate time differences
    # ---------------------------------------------------------

    time_difference = timestamps.diff().dropna()

    print("\n" + "=" * 70)
    print("TIME INTERVAL ANALYSIS")
    print("=" * 70)

    print(
        f"\nMost common interval:"
    )

    print(
        time_difference
        .value_counts()
        .head(10)
        .to_string()
    )

    # ---------------------------------------------------------
    # Check 30-minute intervals
    # ---------------------------------------------------------

    expected_interval = pd.Timedelta(minutes=30)

    incorrect_intervals = (
        time_difference != expected_interval
    ).sum()

    print(
        f"\nExpected interval      : "
        f"{expected_interval}"
    )

    print(
        f"Incorrect intervals     : "
        f"{incorrect_intervals:,}"
    )

    # ---------------------------------------------------------
    # Find gaps
    # ---------------------------------------------------------

    gaps = time_difference[
        time_difference > expected_interval
    ]

    print("\n" + "=" * 70)
    print("TIME GAPS")
    print("=" * 70)

    print(f"\nNumber of gaps: {len(gaps):,}")

    if len(gaps) > 0:

        print("\nLargest gaps:")

        print(
            gaps
            .sort_values(ascending=False)
            .head(10)
            .to_string()
        )

    # ---------------------------------------------------------
    # Check expected number of records
    # ---------------------------------------------------------

    expected_records = 365 * 48

    print("\n" + "=" * 70)
    print("EXPECTED RECORD COUNT")
    print("=" * 70)

    print(
        f"\nExpected records for 2013 : "
        f"{expected_records:,}"
    )

    print(
        f"Actual records            : "
        f"{len(timestamps):,}"
    )

    if len(timestamps) == expected_records:

        print(
            "\nRecord count status       : "
            "CORRECT"
        )

    else:

        print(
            "\nRecord count status       : "
            "DIFFERENT FROM EXPECTED"
        )

    # ---------------------------------------------------------
    # Final assessment
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TIME SERIES VALIDATION RESULT")
    print("=" * 70)

    if (
        invalid_count == 0
        and duplicate_count == 0
        and incorrect_intervals == 0
        and len(timestamps) == expected_records
    ):

        print(
            "\nSTATUS: VALID 30-MINUTE TIME SERIES"
        )

    else:

        print(
            "\nSTATUS: TIME SERIES REQUIRES CLEANING"
        )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()