from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("FORECAST STABILITY VALIDATION")
    print("=" * 70)

    input_file = (
        PROCESSED_DATA_DIR /
        "forecasting_features.csv"
    )

    output_file = (
        PROCESSED_DATA_DIR /
        "stability_validation.csv"
    )

    if not input_file.exists():
        print("\nERROR: forecasting_features.csv not found.")
        return

    print("\nLoading forecasting dataset...")

    df = pd.read_csv(
        input_file,
        parse_dates=["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Remove leakage-prone analysis features
    # ---------------------------------------------------------

    df = df.drop(
        columns=[
            "load_change_30min",
            "load_change_1hour"
        ]
    )

    target = "total_load"

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "timestamp",
            target
        ]
    ]

    X = df[feature_columns]
    y = df[target]

    print(f"\nRows: {len(df):,}")
    print(f"Features: {len(feature_columns)}")

    # ---------------------------------------------------------
    # Time-series cross-validation
    # ---------------------------------------------------------

    n_splits = 4

    test_size = len(df) // 10

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size
    )

    results = []

    print("\n" + "=" * 70)
    print("TIME-SERIES VALIDATION")
    print("=" * 70)

    print(
        f"\nNumber of folds : {n_splits}"
    )

    print(
        f"Test observations per fold: {test_size:,}"
    )

    for fold, (train_index, test_index) in enumerate(
        tscv.split(X),
        start=1
    ):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        train_start = df.iloc[
            train_index[0]
        ]["timestamp"]

        train_end = df.iloc[
            train_index[-1]
        ]["timestamp"]

        test_start = df.iloc[
            test_index[0]
        ]["timestamp"]

        test_end = df.iloc[
            test_index[-1]
        ]["timestamp"]

        print("\n" + "-" * 70)
        print(f"FOLD {fold}")
        print("-" * 70)

        print(
            f"Training: {train_start} → {train_end}"
        )

        print(
            f"Testing : {test_start} → {test_end}"
        )

        print(
            f"Train rows: {len(train_index):,}"
        )

        print(
            f"Test rows : {len(test_index):,}"
        )

        # -----------------------------------------------------
        # Linear Regression
        # -----------------------------------------------------

        linear_model = LinearRegression()

        linear_model.fit(
            X_train,
            y_train
        )

        linear_prediction = linear_model.predict(
            X_test
        )

        mae, rmse, r2 = calculate_metrics(
            y_test,
            linear_prediction
        )

        results.append({
            "fold": fold,
            "model": "Linear Regression",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "test_start": test_start,
            "test_end": test_end
        })

        print(
            f"\nLinear Regression"
            f"\n  MAE : {mae:,.4f}"
            f"\n  RMSE: {rmse:,.4f}"
            f"\n  R2  : {r2:,.4f}"
        )

        # -----------------------------------------------------
        # Random Forest
        # -----------------------------------------------------

        random_forest = RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        print("\nTraining Random Forest...")

        random_forest.fit(
            X_train,
            y_train
        )

        rf_prediction = random_forest.predict(
            X_test
        )

        mae, rmse, r2 = calculate_metrics(
            y_test,
            rf_prediction
        )

        results.append({
            "fold": fold,
            "model": "Random Forest",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "test_start": test_start,
            "test_end": test_end
        })

        print(
            f"\nRandom Forest"
            f"\n  MAE : {mae:,.4f}"
            f"\n  RMSE: {rmse:,.4f}"
            f"\n  R2  : {r2:,.4f}"
        )

        # -----------------------------------------------------
        # Gradient Boosting
        # -----------------------------------------------------

        gradient_model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        )

        print("\nTraining Gradient Boosting...")

        gradient_model.fit(
            X_train,
            y_train
        )

        gradient_prediction = (
            gradient_model.predict(X_test)
        )

        mae, rmse, r2 = calculate_metrics(
            y_test,
            gradient_prediction
        )

        results.append({
            "fold": fold,
            "model": "Histogram Gradient Boosting",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "test_start": test_start,
            "test_end": test_end
        })

        print(
            f"\nHistogram Gradient Boosting"
            f"\n  MAE : {mae:,.4f}"
            f"\n  RMSE: {rmse:,.4f}"
            f"\n  R2  : {r2:,.4f}"
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Stability summary
    # ---------------------------------------------------------

    summary = (
        results_df
        .groupby("model")
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std")
        )
        .reset_index()
    )

    summary["MAE_CV_percent"] = (
        summary["std_MAE"] /
        summary["mean_MAE"] *
        100
    )

    summary = summary.sort_values(
        "mean_MAE"
    )

    print("\n" + "=" * 70)
    print("FORECAST STABILITY SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    # ---------------------------------------------------------
    # Best stable model
    # ---------------------------------------------------------

    best_model = summary.iloc[0]

    print("\n" + "=" * 70)
    print("STABILITY RESULT")
    print("=" * 70)

    print(
        f"\nBest model by average MAE:"
    )

    print(
        f"  {best_model['model']}"
    )

    print(
        f"\nAverage MAE:"
    )

    print(
        f"  {best_model['mean_MAE']:,.4f}"
    )

    print(
        f"\nMAE standard deviation:"
    )

    print(
        f"  {best_model['std_MAE']:,.4f}"
    )

    print(
        f"\nMAE coefficient of variation:"
    )

    print(
        f"  {best_model['MAE_CV_percent']:,.2f}%"
    )

    print(
        f"\nAverage RMSE:"
    )

    print(
        f"  {best_model['mean_RMSE']:,.4f}"
    )

    print(
        f"\nAverage R2:"
    )

    print(
        f"  {best_model['mean_R2']:,.4f}"
    )

    print("\nSaved detailed validation results:")
    print(output_file)

    print("\n" + "=" * 70)
    print("FORECAST STABILITY VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
