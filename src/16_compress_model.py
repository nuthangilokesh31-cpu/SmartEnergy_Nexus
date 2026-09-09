from pathlib import Path
import joblib
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_forecasting_model.joblib"
)

COMPRESSED_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_forecasting_model_compressed.joblib"
)


print("=" * 70)
print("SMARTENERGY NEXUS")
print("MODEL COMPRESSION FOR DEPLOYMENT")
print("=" * 70)

print("\nLoading validated model...")

model = joblib.load(MODEL_FILE)

original_size = os.path.getsize(MODEL_FILE)

print(
    f"Original model size: "
    f"{original_size / (1024 * 1024):.2f} MiB"
)

print("\nCompressing model...")

joblib.dump(
    model,
    COMPRESSED_FILE,
    compress=3,
)

compressed_size = os.path.getsize(COMPRESSED_FILE)

print(
    f"Compressed model size: "
    f"{compressed_size / (1024 * 1024):.2f} MiB"
)

reduction = (
    1
    - compressed_size / original_size
) * 100

print(
    f"Size reduction: "
    f"{reduction:.2f}%"
)

print("\nReloading compressed model...")

compressed_model = joblib.load(
    COMPRESSED_FILE
)

print(
    f"Reloaded model type: "
    f"{type(compressed_model).__name__}"
)

print("\nTesting prediction equivalence...")

import pandas as pd

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)

import json

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

features = metadata["features"]["columns"]

test = pd.read_csv(TEST_FILE)

X_sample = test[features].head(100)

original_predictions = model.predict(
    X_sample
)

compressed_predictions = (
    compressed_model.predict(X_sample)
)

maximum_difference = abs(
    original_predictions
    - compressed_predictions
).max()

print(
    f"Maximum prediction difference: "
    f"{maximum_difference:.12f}"
)

if maximum_difference > 1e-9:
    raise RuntimeError(
        "Compressed model predictions differ "
        "from the original model."
    )

print("\n" + "=" * 70)
print("MODEL COMPRESSION VALIDATION PASSED")
print("=" * 70)
