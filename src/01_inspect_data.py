from pathlib import Path
import sys

import pandas as pd


# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SGSC_FILE


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("SGSC DATASET INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    print("\n[1] Checking dataset...")

    if not SGSC_FILE.exists():

        print("ERROR: Dataset was not found.")
        print(f"Expected location:\n{SGSC_FILE}")

        return

    print("Dataset found.")
    print(f"Path: {SGSC_FILE}")

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    size_mb = SGSC_FILE.stat().st_size / (1024 * 1024)

    print(f"\nFile size: {size_mb:,.2f} MB")

    # --------------------------------------------------------
    # Read small sample
    # --------------------------------------------------------

    print("\n[2] Reading first 1,000 rows...")

    df = pd.read_csv(
        SGSC_FILE,
        nrows=1000
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET STRUCTURE")
    print("=" * 70)

    print(f"Rows inspected   : {len(df):,}")
    print(f"Columns detected : {len(df.columns):,}")

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COLUMN NAMES")
    print("=" * 70)

    for number, column in enumerate(df.columns, start=1):

        print(f"{number:>5}. {column}")

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    print(df.dtypes.to_string())

    # --------------------------------------------------------
    # First rows
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST 5 ROWS")
    print("=" * 70)

    print(df.head().to_string())

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:

        print("No missing values in the first 1,000 rows.")

    else:

        print(missing.to_string())

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()