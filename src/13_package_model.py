from pathlib import Path
import json
import sys
import platform
from datetime import datetime

import joblib
import pandas as pd
import sklearn
import numpy as np

print("=" * 70)
print("SMARTENERGY NEXUS")
print("MODEL PACKAGING & METADATA")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_FILE = MODEL_DIR / "best_forecasting_model.joblib"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
print("\nLoading trained model...")

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)

print(f"Model type: {type(model).__name__}")

# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------
comparison_file = PROCESSED_DIR / "model_comparison.csv"
stability_file = PROCESSED_DIR / "stability_validation.csv"

comparison = pd.read_csv(comparison_file)
stability = pd.read_csv(stability_file)

# ---------------------------------------------------------
# TEST METRICS
# ---------------------------------------------------------
best_row = comparison[
    comparison["model"].str.contains(
        "Random Forest",
        case=False,
        na=False
    )
]

if best_row.empty:
    raise ValueError(
        "Random Forest results not found in model_comparison.csv"
    )

best_row = best_row.iloc[0]

# ---------------------------------------------------------
# RANDOM FOREST FOLD RESULTS
# ---------------------------------------------------------
rf_stability = stability[
    stability["model"].str.contains(
        "Random Forest",
        case=False,
        na=False
    )
].copy()

if rf_stability.empty:
    raise ValueError(
        "Random Forest stability results not found."
    )

print("\nRandom Forest validation folds:")
print(
    rf_stability[
        ["fold", "MAE", "RMSE", "R2"]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# CALCULATE STABILITY STATISTICS
# ---------------------------------------------------------
mean_mae = rf_stability["MAE"].mean()
std_mae = rf_stability["MAE"].std(ddof=0)

mean_rmse = rf_stability["RMSE"].mean()
std_rmse = rf_stability["RMSE"].std(ddof=0)

mean_r2 = rf_stability["R2"].mean()
std_r2 = rf_stability["R2"].std(ddof=0)

mae_cv_percent = (
    (std_mae / mean_mae) * 100
    if mean_mae != 0
    else 0
)

fold_count = len(rf_stability)

# ---------------------------------------------------------
# FEATURE INFORMATION
# ---------------------------------------------------------
train_file = PROCESSED_DIR / "train.csv"

if not train_file.exists():
    raise FileNotFoundError(
        f"Training data not found: {train_file}"
    )

train = pd.read_csv(train_file)

excluded_columns = {
    "timestamp",
    "total_load",
    "load_change_30min",
    "load_change_1hour",
}

feature_columns = [
    column
    for column in train.columns
    if column not in excluded_columns
]

print(f"\nFeature count: {len(feature_columns)}")

# ---------------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------------
model_parameters = {}

if hasattr(model, "get_params"):
    model_parameters = model.get_params()

# ---------------------------------------------------------
# DATA PERIODS
# ---------------------------------------------------------
train["timestamp"] = pd.to_datetime(train["timestamp"])

test_file = PROCESSED_DIR / "test.csv"

if not test_file.exists():
    raise FileNotFoundError(
        f"Test data not found: {test_file}"
    )

test = pd.read_csv(test_file)
test["timestamp"] = pd.to_datetime(test["timestamp"])

# ---------------------------------------------------------
# METADATA
# ---------------------------------------------------------
metadata = {
    "project": {
        "name": "SmartEnergy Nexus",
        "description": (
            "AI-powered predictive and explainable energy "
            "intelligence platform for smart-building energy forecasting."
        ),
        "workflow_step": 13,
        "artifact_type": "forecasting_model"
    },

    "model": {
        "filename": MODEL_FILE.name,
        "model_type": type(model).__name__,
        "selection_criterion": (
            "Lowest mean MAE across chronological validation windows"
        ),
        "parameters": model_parameters
    },

    "features": {
        "count": len(feature_columns),
        "columns": feature_columns,
        "excluded_from_model": [
            "timestamp",
            "total_load",
            "load_change_30min",
            "load_change_1hour"
        ],
        "leakage_note": (
            "load_change_30min and load_change_1hour were excluded "
            "because they use the current target value."
        )
    },

    "training_data": {
        "rows": len(train),
        "start": str(train["timestamp"].min()),
        "end": str(train["timestamp"].max())
    },

    "testing_data": {
        "rows": len(test),
        "start": str(test["timestamp"].min()),
        "end": str(test["timestamp"].max())
    },

    "test_metrics": {
        "MAE": float(best_row["MAE"]),
        "RMSE": float(best_row["RMSE"]),
        "R2": float(best_row["R2"])
    },

    "time_series_validation": {
        "folds": fold_count,
        "mean_MAE": float(mean_mae),
        "std_MAE": float(std_mae),
        "mean_RMSE": float(mean_rmse),
        "std_RMSE": float(std_rmse),
        "mean_R2": float(mean_r2),
        "std_R2": float(std_r2),
        "MAE_CV_percent": float(mae_cv_percent)
    },

    "dataset": {
        "name": (
            "Clustered residential electricity load profiles "
            "from Smart Grid Smart City dataset"
        ),
        "source": "Mendeley Data",
        "interval": "30 minutes",
        "customer_series": 6031,
        "calendar_year": 2013,
        "target": (
            "Aggregate electricity demand across all "
            "6031 residential customer load series"
        )
    },

    "explainability": {
        "method": "SHAP",
        "explainer": "TreeExplainer",
        "sample_size": 500,
        "global_output": "shap_feature_importance.csv",
        "local_output": "shap_local_explanation.csv"
    },

    "optimization": {
        "method": "scenario_based",
        "warning": (
            "Optimization scenarios are hypothetical modeled estimates "
            "and are not measured energy savings."
        )
    },

    "environment": {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__
    },

    "created_at": datetime.now().isoformat()
}

# ---------------------------------------------------------
# SAVE METADATA
# ---------------------------------------------------------
with open(
    MODEL_METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=4
    )

# ---------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL PACKAGE")
print("=" * 70)

print("\nModel file:")
print(f"  {MODEL_FILE}")

print("\nMetadata file:")
print(f"  {MODEL_METADATA_FILE}")

print("\nModel type:")
print(f"  {type(model).__name__}")

print("\nFeatures:")
print(f"  {len(feature_columns)}")

print("\nTest metrics:")
print(f"  MAE : {best_row['MAE']:.4f}")
print(f"  RMSE: {best_row['RMSE']:.4f}")
print(f"  R2  : {best_row['R2']:.4f}")

print("\nTime-series validation:")
print(f"  Folds    : {fold_count}")
print(f"  Mean MAE : {mean_mae:.4f}")
print(f"  Std MAE  : {std_mae:.4f}")
print(f"  Mean RMSE: {mean_rmse:.4f}")
print(f"  Std RMSE : {std_rmse:.4f}")
print(f"  Mean R2  : {mean_r2:.4f}")
print(f"  Std R2   : {std_r2:.4f}")
print(f"  MAE CV % : {mae_cv_percent:.4f}")

print("\nMODEL PACKAGING COMPLETE")

print("\n" + "=" * 70)
