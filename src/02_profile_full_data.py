from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SGSC_FILE, CHUNK_SIZE


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("FULL SGSC DATASET PROFILE")
    print("=" * 70)

    if not SGSC_FILE.exists():
        print("\nERROR: Dataset was not found.")
        print(f"Expected location:\n{SGSC_FILE}")
        return

    print(f"\nDataset: {SGSC_FILE}")

    file_size_mb = SGSC_FILE.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:,.2f} MB")

    print("\nReading dataset in chunks...")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")

    total_rows = 0
    total_missing = 0
    first_timestamp = None
    last_timestamp = None

    column_names = None

    try:

        reader = pd.read_csv(
            SGSC_FILE,
            chunksize=CHUNK_SIZE
        )

        for chunk_number, chunk in enumerate(reader, start=1):

            if column_names is None:
                column_names = list(chunk.columns)

            total_rows += len(chunk)

            missing_count = int(chunk.isna().sum().sum())
            total_missing += missing_count

            # First column is expected to contain timestamps
            timestamp_column = chunk.columns[0]

            timestamps = pd.to_datetime(
                chunk[timestamp_column],
                errors="coerce"
            )

            valid_timestamps = timestamps.dropna()

            if len(valid_timestamps) > 0:

                chunk_first = valid_timestamps.min()
                chunk_last = valid_timestamps.max()

                if first_timestamp is None:
                    first_timestamp = chunk_first
                else:
                    first_timestamp = min(
                        first_timestamp,
                        chunk_first
                    )

                if last_timestamp is None:
                    last_timestamp = chunk_last
                else:
                    last_timestamp = max(
                        last_timestamp,
                        chunk_last
                    )

            print(
                f"Processed chunk {chunk_number:>4} | "
                f"Rows processed: {total_rows:>10,} | "
                f"Missing values: {total_missing:>10,}"
            )

    except Exception as error:

        print("\nERROR while reading dataset:")
        print(error)
        return

    print("\n" + "=" * 70)
    print("FULL DATASET PROFILE")
    print("=" * 70)

    print(f"\nTotal rows              : {total_rows:,}")
    print(f"Total columns           : {len(column_names):,}")
    print(f"Total missing values    : {total_missing:,}")

    print(f"\nFirst timestamp         : {first_timestamp}")
    print(f"Last timestamp          : {last_timestamp}")

    if total_missing == 0:
        print("\nMissing-data status      : NO MISSING VALUES DETECTED")
    else:
        print("\nMissing-data status      : MISSING VALUES DETECTED")

    print("\n" + "=" * 70)
    print("COLUMN SUMMARY")
    print("=" * 70)

    print(f"\nFirst column:")
    print(f"  {column_names[0]}")

    print("\nFirst 10 load columns:")

    for column in column_names[1:11]:
        print(f"  {column}")

    print("\n" + "=" * 70)
    print("PROFILE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()