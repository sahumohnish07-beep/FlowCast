"""Unit tests for Phase 3 baseline model training, evaluation, and serialization."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.baseline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    compute_metrics,
    load_datasets,
    train_linear_model,
)


def test_compute_metrics():
    """Verify accuracy metric calculations."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(6.67, rel=1e-2)
    assert metrics["RMSE"] == pytest.approx(8.16, rel=1e-2)
    assert metrics["R2"] > 0.95


def test_load_datasets():
    """Verify processed dataset splits are loaded cleanly."""
    train_df, val_df, test_df = load_datasets()
    assert len(train_df) > 30000
    assert len(val_df) > 8000
    assert len(test_df) > 6000

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in train_df.columns
        assert col in test_df.columns


def test_linear_model_training_and_prediction():
    """Verify linear baseline pipeline trains and makes valid predictions."""
    train_df, _, test_df = load_datasets()
    X_train = train_df[FEATURE_COLUMNS].head(200)
    y_train = train_df[TARGET_COLUMN].head(200)
    X_test = test_df[FEATURE_COLUMNS].head(50)

    model = train_linear_model(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == 50
    assert not np.isnan(preds).any()


def test_best_model_artifact_loading():
    """Verify that serialized model artifact loads and conducts valid inference."""
    artifact_path = Path("models/best_baseline_model.joblib")
    if not artifact_path.exists():
        pytest.skip("models/best_baseline_model.joblib not found. Run baseline.py first.")

    bundle = joblib.load(artifact_path)
    assert "model" in bundle
    assert "feature_names" in bundle
    assert "test_metrics" in bundle
    assert bundle["model_name"] in ["XGBoost", "Random Forest", "Linear Regression"]

    # Verify inference on sample
    _, _, test_df = load_datasets()
    X_sample = test_df[bundle["feature_names"]].head(10)
    preds = bundle["model"].predict(X_sample)
    assert len(preds) == 10
    assert (preds >= 0).all()


def test_baseline_model_prediction_range():
    """Verify that predictions on test set stay within plausible freeway capacities."""
    artifact_path = Path("models/best_baseline_model.joblib")
    if not artifact_path.exists():
        pytest.skip("models/best_baseline_model.joblib not found.")

    bundle = joblib.load(artifact_path)
    _, _, test_df = load_datasets()
    X_sample = test_df[bundle["feature_names"]].head(100)
    preds = bundle["model"].predict(X_sample)

    assert (preds >= 0.0).all()
    # Realistic highway maximum for 4 lanes is 8000-9000 veh/hr
    assert (preds <= 8500.0).all()


def test_baseline_evaluation_leaderboard_file():
    """Verify that docs/baseline_results.csv is formatted properly with all required metrics."""
    results_path = Path("docs/baseline_results.csv")
    assert results_path.exists(), "Leaderboard file docs/baseline_results.csv missing."

    df = pd.read_csv(results_path)
    expected_cols = ["Model", "Test_MAE", "Test_RMSE", "Test_R2"]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column {col} in docs/baseline_results.csv"

    assert len(df) >= 3, "Expected at least 3 models in docs/baseline_results.csv"


