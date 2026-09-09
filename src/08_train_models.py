from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_STATE


def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("-" * 50)
    print(f"MAE  : {mae:,.4f}")
    print(f"RMSE : {rmse:,.4f}")
    print(f"R2   : {r2:,.4f}")

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def main():

    print("=" * 70)
    print("SMARTENERGY NEXUS")
    print("FORECASTING MODEL TRAINING")
    print("=" * 70)

    train_file = PROCESSED_DATA_DIR / "train.csv"
    test_file = PROCESSED_DATA_DIR / "test.csv"

    if not train_file.exists():
        print("\nERROR: train.csv not found.")
        return

    if not test_file.exists():
        print("\nERROR: test.csv not found.")
        return

    print("\nLoading training data...")
    train = pd.read_csv(
        train_file,
        parse_dates=["timestamp"]
    )

    print("Loading testing data...")
    test = pd.read_csv(
        test_file,
        parse_dates=["timestamp"]
    )

    print(f"\nTraining rows: {len(train):,}")
    print(f"Testing rows : {len(test):,}")

    target = "total_load"

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------

    excluded = [
        "timestamp",
        target
    ]

    feature_columns = [
        column
        for column in train.columns
        if column not in excluded
    ]

    X_train = train[feature_columns]
    y_train = train[target]

    X_test = test[feature_columns]
    y_test = test[target]

    print(f"\nNumber of model features: {len(feature_columns)}")

    # ---------------------------------------------------------
    # Baseline 1: Persistence
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE MODELS")
    print("=" * 70)

    persistence_prediction = test["lag_1"]

    baseline_results = []

    baseline_results.append(
        evaluate_model(
            "Persistence (lag_1)",
            y_test,
            persistence_prediction
        )
    )

    # ---------------------------------------------------------
    # Baseline 2: Same time previous day
    # ---------------------------------------------------------

    seasonal_prediction = test["lag_48"]

    baseline_results.append(
        evaluate_model(
            "Seasonal Naive (lag_48)",
            y_test,
            seasonal_prediction
        )
    )

    # ---------------------------------------------------------
    # Linear Regression
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL 1: LINEAR REGRESSION")
    print("=" * 70)

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_prediction = linear_model.predict(
        X_test
    )

    linear_result = evaluate_model(
        "Linear Regression",
        y_test,
        linear_prediction
    )

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL 2: RANDOM FOREST")
    print("=" * 70)

    random_forest = RandomForestRegressor(
        n_estimators=200,
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

    rf_result = evaluate_model(
        "Random Forest",
        y_test,
        rf_prediction
    )

    # ---------------------------------------------------------
    # HistGradientBoosting
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL 3: HISTOGRAM GRADIENT BOOSTING")
    print("=" * 70)

    gradient_model = HistGradientBoostingRegressor(
        max_iter=300,
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

    gradient_prediction = gradient_model.predict(
        X_test
    )

    gradient_result = evaluate_model(
        "Histogram Gradient Boosting",
        y_test,
        gradient_prediction
    )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    results = baseline_results + [
        linear_result,
        rf_result,
        gradient_result
    ]

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "MAE"
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Select best ML model
    # ---------------------------------------------------------

    ml_results = results_df[
        ~results_df["model"].isin([
            "Persistence (lag_1)",
            "Seasonal Naive (lag_48)"
        ])
    ]

    best_model_name = ml_results.iloc[0]["model"]

    print("\nBest machine-learning model:")
    print(f"  {best_model_name}")

    if best_model_name == "Linear Regression":

        best_model = linear_model
        best_prediction = linear_prediction

    elif best_model_name == "Random Forest":

        best_model = random_forest
        best_prediction = rf_prediction

    else:

        best_model = gradient_model
        best_prediction = gradient_prediction

    # ---------------------------------------------------------
    # Save best model
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_file = (
        MODEL_DIR /
        "best_forecasting_model.joblib"
    )

    joblib.dump(
        best_model,
        model_file
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    predictions = pd.DataFrame({
        "timestamp": test["timestamp"],
        "actual_load": y_test,
        "predicted_load": best_prediction
    })

    prediction_file = (
        PROCESSED_DATA_DIR /
        "test_predictions.csv"
    )

    predictions.to_csv(
        prediction_file,
        index=False
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics_file = (
        PROCESSED_DATA_DIR /
        "model_comparison.csv"
    )

    results_df.to_csv(
        metrics_file,
        index=False
    )

    print("\nSaved:")
    print(f"  Model      : {model_file}")
    print(f"  Predictions: {prediction_file}")
    print(f"  Metrics    : {metrics_file}")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
