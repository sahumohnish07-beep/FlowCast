"""Baseline models for FlowCast traffic volume prediction.

Implements Linear Regression (Ridge), Random Forest, and XGBoost regressors
for hourly traffic volume forecasting evaluated on a strictly chronological test set.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys

# Ensure FlowCast root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger("flowcast.models.baseline")

# Default predictive features (strictly historical and contemporary covariates)
FEATURE_COLUMNS = [
    # Temporal & calendar features
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
    "is_morning_rush",
    "is_evening_rush",
    "is_rush_hour",
    # Cyclical encodings
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    # Meteorological covariates
    "temp_c",
    "clouds_all",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "is_raining",
    "is_snowing",
    "is_freezing",
    "weather_severity_index",
    # Time-series lags (strictly past observations)
    "traffic_lag_1h",
    "traffic_lag_2h",
    "traffic_lag_3h",
    "traffic_lag_24h",
    "traffic_lag_168h",
    # Rolling window summary statistics
    "traffic_rolling_mean_3h",
    "traffic_rolling_mean_6h",
    "traffic_rolling_mean_24h",
    "traffic_rolling_std_24h",
]

TARGET_COLUMN = "traffic_volume"


def load_datasets(
    processed_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets from parquet.

    Args:
        processed_dir: Path to directory containing processed splits.

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    proc = Path(processed_dir).resolve()
    train_path = proc / "train.parquet"
    val_path = proc / "val.parquet"
    test_path = proc / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Missing parquet splits in {proc}. Run preprocessing first.")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    return df_train, df_val, df_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R2 regression performance metrics.

    Args:
        y_true: Array of ground-truth traffic volume values.
        y_pred: Array of model predicted traffic volumes.

    Returns:
        Dictionary containing MAE, RMSE, and R2 scores.
    """
    # Enforce non-negative traffic volume predictions
    y_pred_clipped = np.clip(y_pred, 0.0, None)

    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(root_mean_squared_error(y_true, y_pred_clipped))
    r2 = float(r2_score(y_true, y_pred_clipped))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }


def train_linear_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
) -> Pipeline:
    """Train standardized Linear Regression with L2 regularization (Ridge)."""
    logger.info("Training Linear Regression (Ridge, alpha=%.2f)...", alpha)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 16,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Train Random Forest Regressor."""
    logger.info("Training Random Forest Regressor (n_estimators=%d, max_depth=%d)...", n_estimators, max_depth)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=n_jobs,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    n_jobs: int = -1,
) -> XGBRegressor:
    """Train Gradient Boosted Decision Trees (XGBoost Regressor)."""
    logger.info(
        "Training XGBoost Regressor (n_estimators=%d, lr=%.2f, max_depth=%d)...",
        n_estimators,
        learning_rate,
        max_depth,
    )
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=n_jobs,
    )
    xgb.fit(X_train, y_train)
    return xgb


def plot_actual_vs_predicted(
    timestamps: pd.Series,
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    best_model_name: str,
    output_path: str | Path = "docs/images/actual_vs_predicted_baseline.png",
    zoom_hours: int = 168,
) -> Path:
    """Generate dual-panel visualization: time-series zoom and prediction scatter plot.

    Args:
        timestamps: Series of test date_time timestamps.
        y_true: Ground truth test values.
        predictions: Dictionary mapping model names to predicted numpy arrays.
        best_model_name: Identifier for the top performing model.
        output_path: Output image destination path.
        zoom_hours: Number of consecutive hours to display in time series overlay (168h = 1 week).

    Returns:
        Path to the saved PNG image.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=300)

    # Subplot 1: 1-Week (168-hour) Time-Series Zoom Comparison
    zoom_slice = slice(0, zoom_hours)
    time_subset = timestamps.iloc[zoom_slice]
    true_subset = y_true.iloc[zoom_slice]

    axes[0].plot(time_subset, true_subset, label="Actual Ground Truth", color="black", linewidth=2.2, alpha=0.85)

    color_palette = {
        "Linear Regression": "#2ca02c",
        "Random Forest": "#ff7f0e",
        "XGBoost": "#1f77b4",
    }

    for name, pred in predictions.items():
        axes[0].plot(
            time_subset,
            pred[zoom_slice],
            label=f"{name} Forecast",
            color=color_palette.get(name, "#9467bd"),
            linewidth=1.8,
            linestyle="--" if name != best_model_name else "-",
        )

    axes[0].set_title(
        f"Hourly Traffic Volume Forecast vs. Actual (First {zoom_hours} Hours of Test Period: Jan 2018)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_ylabel("Traffic Volume (vehicles/hour)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # Subplot 2: Scatter Plot of Best Model (Actual vs. Predicted)
    best_pred = predictions[best_model_name]
    axes[1].scatter(y_true, best_pred, alpha=0.25, color="#1f77b4", s=12, edgecolors="none")

    # Reference ideal y = x line
    min_val = 0
    max_val = max(float(y_true.max()), float(best_pred.max())) + 200
    axes[1].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.8, label="Ideal Prediction (y = x)")

    axes[1].set_title(
        f"Test Set Prediction Accuracy: {best_model_name} (R² = {r2_score(y_true, best_pred):.4f})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xlabel("Actual Traffic Volume (veh/hr)", fontsize=10)
    axes[1].set_ylabel("Predicted Traffic Volume (veh/hr)", fontsize=10)
    axes[1].set_xlim(min_val, max_val)
    axes[1].set_ylim(min_val, max_val)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", frameon=True, facecolor="white")

    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    logger.info("Saved actual vs. predicted plot to %s", out)
    return out


def run_baseline_training(
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "models",
    docs_dir: str | Path = "docs",
) -> tuple[pd.DataFrame, str]:
    """Execute complete training, evaluation, logging, and artifact persistence.

    Args:
        processed_dir: Path to directory containing processed splits.
        models_dir: Directory where the best serialized model will be saved.
        docs_dir: Directory where comparison table and plots will be saved.

    Returns:
        Tuple of (results_dataframe, best_model_name).
    """
    logger.info("Loading processed train/val/test datasets...")
    df_train, df_val, df_test = load_datasets(processed_dir)

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN]

    X_val = df_val[FEATURE_COLUMNS]
    y_val = df_val[TARGET_COLUMN]

    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test[TARGET_COLUMN]

    logger.info(
        "Features: %d predictors. Training on %d samples, testing on %d samples.",
        len(FEATURE_COLUMNS),
        len(X_train),
        len(X_test),
    )

    # Model Dictionary
    models: dict[str, object] = {
        "Linear Regression": train_linear_model(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train, n_estimators=100, max_depth=16),
        "XGBoost": train_xgboost(X_train, y_train, n_estimators=250, learning_rate=0.05, max_depth=6),
    }

    results = []
    test_predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        logger.info("Evaluating %s on validation and test sets...", name)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = np.clip(model.predict(X_test), 0.0, None)
        test_predictions[name] = test_pred

        train_m = compute_metrics(y_train.values, train_pred)
        val_m = compute_metrics(y_val.values, val_pred)
        test_m = compute_metrics(y_test.values, test_pred)

        results.append(
            {
                "Model": name,
                "Test_MAE": test_m["MAE"],
                "Test_RMSE": test_m["RMSE"],
                "Test_R2": test_m["R2"],
                "Val_MAE": val_m["MAE"],
                "Val_RMSE": val_m["RMSE"],
                "Val_R2": val_m["R2"],
                "Train_MAE": train_m["MAE"],
                "Train_RMSE": train_m["RMSE"],
                "Train_R2": train_m["R2"],
            }
        )

    df_results = pd.DataFrame(results).sort_values(by="Test_RMSE").reset_index(drop=True)

    # Save comparison table
    docs_path = Path(docs_dir).resolve()
    docs_path.mkdir(parents=True, exist_ok=True)
    csv_results_path = docs_path / "baseline_results.csv"
    df_results.to_csv(csv_results_path, index=False)
    logger.info("Saved baseline comparison metrics to %s", csv_results_path)

    # Determine best model based on lowest Test RMSE
    best_model_name = df_results.iloc[0]["Model"]
    best_model = models[best_model_name]
    logger.info("Best model selected: %s (Test RMSE: %.2f, R²: %.4f)", best_model_name, df_results.iloc[0]["Test_RMSE"], df_results.iloc[0]["Test_R2"])

    # Generate and save actual vs. predicted plot
    img_path = docs_path / "images" / "actual_vs_predicted_baseline.png"
    plot_actual_vs_predicted(
        timestamps=df_test["date_time"],
        y_true=y_test,
        predictions=test_predictions,
        best_model_name=best_model_name,
        output_path=img_path,
    )

    # Save best model artifact to models/
    models_path = Path(models_dir).resolve()
    models_path.mkdir(parents=True, exist_ok=True)
    best_model_artifact_path = models_path / "best_baseline_model.joblib"

    artifact_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "test_metrics": df_results.iloc[0].to_dict(),
        "trained_at": datetime.now().isoformat(),
    }
    joblib.dump(artifact_bundle, best_model_artifact_path)
    logger.info("Saved best model artifact bundle to %s", best_model_artifact_path)

    # Display console summary
    print("\n" + "=" * 80)
    print(" FlowCast Phase 3: Baseline Models Performance Comparison (Test Set)")
    print("=" * 80)
    print(df_results[["Model", "Test_MAE", "Test_RMSE", "Test_R2", "Val_MAE", "Val_R2"]].to_string(index=False))
    print("=" * 80)
    print(f"Top Performer : {best_model_name}")
    print(f"Model Artifact: {best_model_artifact_path}")
    print(f"Results CSV   : {csv_results_path}")
    print(f"Accuracy Plot : {img_path}")
    print("=" * 80 + "\n")

    return df_results, best_model_name


def main() -> None:
    """CLI entry point for baseline model training."""
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models for FlowCast.")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed datasets.")
    parser.add_argument("--models-dir", default="models", help="Destination directory for model binaries.")
    parser.add_argument("--docs-dir", default="docs", help="Destination directory for metrics and plots.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_baseline_training(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        docs_dir=args.docs_dir,
    )


if __name__ == "__main__":
    main()
