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

