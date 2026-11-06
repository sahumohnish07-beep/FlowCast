"""Unit tests for Phase 5 extra prediction modules (Congestion, Travel Time, Accident Risk)."""

from pathlib import Path
import joblib
import numpy as np
import pytest

from src.models.extra_modules import PREDICTOR_FEATURES


def test_congestion_classifier_artifact():
    """Verify serialized congestion classification model and inference."""
    path = Path("models/congestion_classifier.joblib")
    if not path.exists():
        pytest.skip("models/congestion_classifier.joblib not found.")

    bundle = joblib.load(path)
    assert "model" in bundle
    assert "classes" in bundle
    assert bundle["classes"] == ["Low", "Medium", "High"]
    assert bundle["metrics"]["F1_Macro"] > 0.85

    # Test dummy inference
    sample = np.zeros((3, len(PREDICTOR_FEATURES)))
    preds = bundle["model"].predict(sample)
    assert len(preds) == 3
    assert set(preds).issubset({0, 1, 2})


def test_travel_time_model_artifact():
    """Verify serialized travel time model and output plausibility."""
    path = Path("models/travel_time_model.joblib")
    if not path.exists():
        pytest.skip("models/travel_time_model.joblib not found.")

    bundle = joblib.load(path)
    assert "model" in bundle
    assert bundle["metrics"]["MAE_Minutes"] < 1.0  # High precision
    assert bundle["metrics"]["R2"] > 0.90

    sample = np.zeros((3, len(PREDICTOR_FEATURES)))
    preds = bundle["model"].predict(sample)
    assert len(preds) == 3
    assert (preds > 0).all()


def test_accident_risk_classifier_artifact():
    """Verify serialized accident risk model and probability calibration."""
    path = Path("models/accident_risk_classifier.joblib")
    if not path.exists():
        pytest.skip("models/accident_risk_classifier.joblib not found.")

    bundle = joblib.load(path)
    assert "model" in bundle
    assert bundle["imbalance_handling"] == "SMOTE"
    assert bundle["metrics"]["Recall"] > 0.85
    assert bundle["metrics"]["ROC_AUC"] > 0.90

    sample = np.zeros((3, len(PREDICTOR_FEATURES)))
    preds = bundle["model"].predict(sample)
    probs = bundle["model"].predict_proba(sample)[:, 1]

    assert len(preds) == 3
    assert set(preds).issubset({0, 1})
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

