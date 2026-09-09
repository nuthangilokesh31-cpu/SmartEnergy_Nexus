from pathlib import Path
import json
import joblib
import pandas as pd

print("=" * 70)
print("SMARTENERGY NEXUS")
print("DEPLOYMENT MODEL VALIDATION")
print("=" * 70)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_forecasting_model.joblib"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

print("\nLoading model...")
model = joblib.load(MODEL_FILE)

print(f"Model type: {type(model).__name__}")

print("\nLoading metadata...")
with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

print(
    f"Metadata model type: "
    f"{metadata['model']['model_type']}"
)

features = metadata["features"]["columns"]

print(f"Expected feature count: {len(features)}")

print("\nLoading test data...")
test = pd.read_csv(TEST_FILE)

X_test = test[features]

print(f"Test rows: {len(X_test):,}")
print(f"Test features: {X_test.shape[1]}")

print("\nRunning prediction test...")

sample = X_test.head(10)

predictions = model.predict(sample)

print("\nSample predictions:")
for i, prediction in enumerate(predictions, start=1):
    print(
        f"  Sample {i:02d}: "
        f"{prediction:,.4f}"
    )

if len(predictions) != 10:
    raise RuntimeError(
        "Prediction count does not match sample count."
    )

if not all(
    pd.notna(predictions)
):
    raise RuntimeError(
        "Prediction contains missing values."
    )

print("\n" + "=" * 70)
print("DEPLOYMENT MODEL VALIDATION PASSED")
print("=" * 70)
