from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib
import shap

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODEL_DIR, FIGURE_DIR


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("EXPLAINABLE AI - SHAP ANALYSIS")
    print("=" * 70)

    model_file = (
        MODEL_DIR /
        "best_forecasting_model.joblib"
    )

    test_file = (
        PROCESSED_DATA_DIR /
        "test.csv"
    )

    if not model_file.exists():
        print("\nERROR: Forecasting model not found.")
        print(model_file)
        return

    if not test_file.exists():
        print("\nERROR: Test dataset not found.")
        print(test_file)
        return

    print("\nLoading trained model...")

    model = joblib.load(
        model_file
    )

    print(
        f"Model type: {type(model).__name__}"
    )

    print("\nLoading test data...")

    test = pd.read_csv(
        test_file,
        parse_dates=["timestamp"]
    )

    # ---------------------------------------------------------
    # Prepare features
    # ---------------------------------------------------------

    target = "total_load"

    excluded = [
        "timestamp",
        target
    ]

    feature_columns = [
        column
        for column in test.columns
        if column not in excluded
    ]

    X_test = test[
        feature_columns
    ]

    print(
        f"\nTest rows: {len(X_test):,}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # SHAP sample
    # ---------------------------------------------------------

    # Use a representative sample to keep SHAP computation
    # efficient while retaining the actual test distribution.

    sample_size = min(
        500,
        len(X_test)
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    print(
        f"\nSHAP sample size: {len(X_sample):,}"
    )

    # ---------------------------------------------------------
    # Tree SHAP
    # ---------------------------------------------------------

    print("\nCreating SHAP explainer...")

    explainer = shap.TreeExplainer(
        model
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X_sample
    )

    # ---------------------------------------------------------
    # Global feature importance
    # ---------------------------------------------------------

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap
    })

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}"
        )
    )

    # ---------------------------------------------------------
    # Save importance
    # ---------------------------------------------------------

    importance_file = (
        PROCESSED_DATA_DIR /
        "shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_file,
        index=False
    )

    # ---------------------------------------------------------
    # SHAP summary plot
    # ---------------------------------------------------------

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nCreating SHAP summary plot...")

    plt.figure(
        figsize=(12, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    summary_plot = (
        FIGURE_DIR /
        "shap_summary.png"
    )

    plt.savefig(
        summary_plot,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # SHAP bar plot
    # ---------------------------------------------------------

    print("Creating SHAP importance bar chart...")

    plt.figure(
        figsize=(12, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    bar_plot = (
        FIGURE_DIR /
        "shap_feature_importance.png"
    )

    plt.savefig(
        bar_plot,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # Local explanation for a representative observation
    # ---------------------------------------------------------

    local_index = 0

    local_timestamp = test.iloc[
        X_sample.index[local_index]
    ]["timestamp"]

    local_values = shap_values[
        local_index
    ]

    local_explanation = pd.DataFrame({
        "feature": feature_columns,
        "feature_value": X_sample.iloc[
            local_index
        ].values,
        "shap_value": local_values
    })

    local_explanation[
        "absolute_shap"
    ] = local_explanation[
        "shap_value"
    ].abs()

    local_explanation = (
        local_explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("LOCAL SHAP EXPLANATION")
    print("=" * 70)

    print(
        f"\nTimestamp: {local_timestamp}"
    )

    print(
        "\nTop 15 contributing features:"
    )

    print(
        local_explanation.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}"
        )
    )

    local_file = (
        PROCESSED_DATA_DIR /
        "shap_local_explanation.csv"
    )

    local_explanation.to_csv(
        local_file,
        index=False
    )

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nSaved files:")

    print(
        f"\n  {importance_file}"
    )

    print(
        f"\n  {local_file}"
    )

    print("\nSaved figures:")

    print(
        f"\n  {summary_plot}"
    )

    print(
        f"\n  {bar_plot}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
