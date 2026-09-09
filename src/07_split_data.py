from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("TIME-BASED TRAIN / TEST SPLIT")
    print("=" * 70)

    input_file = (
        PROCESSED_DATA_DIR /
        "forecasting_features.csv"
    )

    train_file = (
        PROCESSED_DATA_DIR /
        "train.csv"
    )

    test_file = (
        PROCESSED_DATA_DIR /
        "test.csv"
    )

    if not input_file.exists():
        print("\nERROR: Feature dataset not found.")
        print(input_file)
        return

    print("\nLoading feature dataset...")

    df = pd.read_csv(
        input_file,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(f"Total rows: {len(df):,}")

    # ---------------------------------------------------------
    # Remove leakage-prone analysis-only features
    # ---------------------------------------------------------

    analysis_only_features = [
        "load_change_30min",
        "load_change_1hour"
    ]

    df_model = df.drop(
        columns=analysis_only_features
    )

    print("\nExcluded from model inputs:")
    for column in analysis_only_features:
        print(f"  - {column}")

    # ---------------------------------------------------------
    # Chronological 80 / 20 split
    # ---------------------------------------------------------

    split_index = int(
        len(df_model) * 0.80
    )

    train = df_model.iloc[
        :split_index
    ].copy()

    test = df_model.iloc[
        split_index:
    ].copy()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    train.to_csv(
        train_file,
        index=False
    )

    test.to_csv(
        test_file,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print(f"\nTotal rows : {len(df_model):,}")
    print(f"Train rows : {len(train):,}")
    print(f"Test rows  : {len(test):,}")

    print("\nTraining period:")
    print(f"  Start: {train['timestamp'].min()}")
    print(f"  End  : {train['timestamp'].max()}")

    print("\nTesting period:")
    print(f"  Start: {test['timestamp'].min()}")
    print(f"  End  : {test['timestamp'].max()}")

    print("\nTraining proportion:")
    print(
        f"  {len(train) / len(df_model) * 100:.2f}%"
    )

    print("\nTesting proportion:")
    print(
        f"  {len(test) / len(df_model) * 100:.2f}%"
    )

    print("\nSaved files:")

    print("  Train:")
    print(train_file)

    print("\n  Test:")
    print(test_file)

    print("\n" + "=" * 70)
    print("TIME-BASED SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
