"""Unit and integration tests for Phase 2 data cleaning and feature engineering."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.preprocess import (
    clean_traffic_data,
    engineer_features,
    merge_multimodal_data,
    split_and_save_data,
)


def test_clean_traffic_data_deduplication():
    """Verify that clean_traffic_data deduplicates timestamps and handles sensor anomalies."""
    raw_path = Path("data/raw/metro_traffic_raw.csv")
    if not raw_path.exists():
        pytest.skip("data/raw/metro_traffic_raw.csv missing.")

    df_cleaned = clean_traffic_data(raw_path)
    assert not df_cleaned.empty
    # No duplicate date_time
    assert df_cleaned["date_time"].duplicated().sum() == 0
    # No extreme rain glitch (> 100 mm)
    assert df_cleaned["rain_1h"].max() <= 100.0 or pd.isna(df_cleaned["rain_1h"].max())
    # No zero Kelvin temperature
    assert (df_cleaned["temp"] < 200.0).sum() == 0


def test_feature_engineering_outputs():
    """Verify engineered features, cyclical encodings, and targets."""
    full_path = Path("data/processed/full.parquet")
    if not full_path.exists():
        pytest.skip("data/processed/full.parquet missing.")

    df = pd.read_parquet(full_path)
    assert len(df) > 1000

    required_features = [
        # Temporal & cyclical
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        # Weather
        "temp_c",
        "weather_severity_index",
        # Lags
        "traffic_lag_1h",
        "traffic_lag_2h",
        "traffic_lag_24h",
        "traffic_lag_168h",
        # Rolling
        "traffic_rolling_mean_3h",
        "traffic_rolling_mean_24h",
        # Targets
        "traffic_volume",
        "congestion_level",
        "travel_time_min",
        "accident_risk_index",
    ]

    for feat in required_features:
        assert feat in df.columns, f"Missing expected feature: {feat}"

    # Verify no NaNs in primary features and target
    assert df["traffic_volume"].isna().sum() == 0
    assert df["traffic_lag_1h"].isna().sum() == 0
    assert df["congestion_level"].isin([0, 1, 2]).all()
    assert (df["accident_risk_index"] >= 0.0).all() and (df["accident_risk_index"] <= 1.0).all()


def test_time_based_split_chronology():
    """Verify strict chronological non-overlapping time splits."""
    proc_dir = Path("data/processed")
    train_path = proc_dir / "train.parquet"
    val_path = proc_dir / "val.parquet"
    test_path = proc_dir / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Processed splits missing.")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # Strictly chronological splits: train < val < test
    assert df_train["date_time"].max() < df_val["date_time"].min()
    assert df_val["date_time"].max() < df_test["date_time"].min()

    # Check non-empty splits
    assert len(df_train) > 30000
    assert len(df_val) > 8000
    assert len(df_test) > 6000


def test_cyclical_trigonometric_bounds():
    """Verify that sine/cosine temporal features lie on the unit circle."""
    full_path = Path("data/processed/full.parquet")
    if not full_path.exists():
        pytest.skip("data/processed/full.parquet missing.")

    df = pd.read_parquet(full_path)
    hour_norm = df["hour_sin"] ** 2 + df["hour_cos"] ** 2
    day_norm = df["day_sin"] ** 2 + df["day_cos"] ** 2
    month_norm = df["month_sin"] ** 2 + df["month_cos"] ** 2

    np.testing.assert_allclose(hour_norm, 1.0, atol=1e-5)
    np.testing.assert_allclose(day_norm, 1.0, atol=1e-5)
    np.testing.assert_allclose(month_norm, 1.0, atol=1e-5)


def test_bpr_travel_time_formula():
    """Verify BPR travel time function physics: T = T0 * (1 + alpha * (V/C)^beta)."""
    full_path = Path("data/processed/full.parquet")
    if not full_path.exists():
        pytest.skip("data/processed/full.parquet missing.")

    df = pd.read_parquet(full_path)
    # T0 is 15 miles / 65 mph * 60 = 13.846 minutes
    t0 = 15.0 / 65.0 * 60.0

    # Travel time must always be greater than or equal to free-flow time
    assert (df["travel_time_min"] >= t0 - 1e-4).all()

    # High volume should have strictly higher travel time than zero flow
    assert df.loc[df["traffic_volume"] > 5500, "travel_time_min"].mean() > df.loc[df["traffic_volume"] < 1000, "travel_time_min"].mean()


def test_weather_severity_index_bounds():
    """Verify weather severity index is strictly normalized in [0.0, 1.0]."""
    full_path = Path("data/processed/full.parquet")
    if not full_path.exists():
        pytest.skip("data/processed/full.parquet missing.")

    df = pd.read_parquet(full_path)
    severity = df["weather_severity_index"]
    assert severity.min() >= 0.0
    assert severity.max() <= 1.0


