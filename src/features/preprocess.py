"""Feature engineering, data cleaning, and preprocessing pipeline for FlowCast.

Transforms raw traffic, weather, and calendar data into an ML-ready time-series
dataset with lag features, cyclical temporal encodings, weather severity metrics,
and multi-task prediction targets (volume, congestion level, travel time, and accident risk).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.calendar_generator import load_calendar_data
from src.data.traffic_loader import load_traffic_data
from src.data.weather_client import load_weather_data

logger = logging.getLogger(__name__)


def clean_traffic_data(raw_traffic_path: str | Path) -> pd.DataFrame:
    """Load, deduplicate, and clean sensor anomalies from raw traffic records.

    Deduplicates multiple hourly records caused by simultaneous weather observations,
    fixes kelvin temperature sensor zeroes, and clips extreme rainfall recording glitches.

    Args:
        raw_traffic_path: Path to `metro_traffic_raw.csv`.

    Returns:
        Cleaned, hourly aggregated pandas.DataFrame sorted by date_time.
    """
    df = load_traffic_data(raw_traffic_path)

    # 1. Handle physical temperature anomalies (Kelvin < 200 is impossible in MN)
    df.loc[df["temp"] < 200.0, "temp"] = np.nan

    # 2. Handle extreme rain glitch (9831.3 mm glitch in UCI dataset)
    df.loc[df["rain_1h"] > 100.0, "rain_1h"] = np.nan

    # 3. Deduplicate duplicate hourly timestamps efficiently using C-accelerated aggregations
    # Sort first so meaningful holiday indicators and higher precipitation values are prioritized
    df_sorted = df.sort_values(
        by=["date_time", "holiday", "rain_1h"],
        ascending=[True, False, False],
    )

    agg_rules = {
        "traffic_volume": "mean",
        "temp": "mean",
        "rain_1h": "max",
        "snow_1h": "max",
        "clouds_all": "mean",
        "weather_main": "first",
        "weather_description": "first",
        "holiday": "first",
    }

    df_cleaned = df_sorted.groupby("date_time", as_index=False).agg(agg_rules)
    df_cleaned["traffic_volume"] = df_cleaned["traffic_volume"].round().astype(int)
    df_cleaned = df_cleaned.sort_values(by="date_time").reset_index(drop=True)

    return df_cleaned


def merge_multimodal_data(
    df_traffic: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Merge traffic observations with Open-Meteo weather and calendar attributes.

    Constructs a regular continuous hourly time index, interpolates missing
    traffic records using diurnal seasonal medians, and enriches with high-precision
    meteorological variables.

    Args:
        df_traffic: Cleaned traffic DataFrame.
        df_weather: Raw Open-Meteo hourly weather DataFrame.
        df_calendar: Raw daily calendar DataFrame.

    Returns:
        Consolidated, continuous hourly DataFrame.
    """
    start_dt = df_traffic["date_time"].min().floor("h")
    end_dt = df_traffic["date_time"].max().ceil("h")
    full_index = pd.date_range(start=start_dt, end=end_dt, freq="h", name="date_time")

    # Reindex traffic to full continuous hourly grid
    df_merged = pd.DataFrame(index=full_index).reset_index()
    df_merged = pd.merge(df_merged, df_traffic, on="date_time", how="left")

    # Merge Open-Meteo weather
    weather_cols_to_use = [
        "date_time",
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
    ]
    df_weather_subset = df_weather[[c for c in weather_cols_to_use if c in df_weather.columns]]
    df_merged = pd.merge(df_merged, df_weather_subset, on="date_time", how="left")

    # Merge calendar attributes on daily date key
    df_merged["date_str"] = df_merged["date_time"].dt.strftime("%Y-%m-%d")
    df_calendar_work = df_calendar.copy()
    if pd.api.types.is_datetime64_any_dtype(df_calendar_work["date"]):
        df_calendar_work["date_str"] = df_calendar_work["date"].dt.strftime("%Y-%m-%d")
    else:
        df_calendar_work["date_str"] = df_calendar_work["date"].astype(str)

    cal_cols = ["date_str", "is_weekend", "is_holiday", "holiday_name", "is_holiday_eve", "quarter"]
    cal_subset = df_calendar_work[[c for c in cal_cols if c in df_calendar_work.columns]]
    df_merged = pd.merge(df_merged, cal_subset, on="date_str", how="left")
    df_merged = df_merged.drop(columns=["date_str"])

    # Harmonize temperature (convert Kelvin to Celsius if present, else use Open-Meteo)
    if "temp" in df_merged.columns:
        # Convert Kelvin to Celsius
        celsius_from_sensor = df_merged["temp"] - 273.15
        df_merged["temp_c"] = celsius_from_sensor.fillna(df_merged.get("temperature_2m", 10.0))
    else:
        df_merged["temp_c"] = df_merged.get("temperature_2m", 10.0)

    # Impute missing traffic volume with combination of short linear interpolation
    # and seasonal (day_of_week, hour) medians
    df_merged["hour"] = df_merged["date_time"].dt.hour
    df_merged["day_of_week"] = df_merged["date_time"].dt.day_of_week
    df_merged["traffic_imputed"] = df_merged["traffic_volume"].isna().astype(int)

    # Seasonal median lookup
    seasonal_medians = df_merged.groupby(["day_of_week", "hour"])["traffic_volume"].transform("median")
    df_merged["traffic_volume"] = (
        df_merged["traffic_volume"]
        .interpolate(method="linear", limit=3)
        .fillna(seasonal_medians)
        .round()
        .astype(int)
    )

    # Fill weather missing values (if any)
    for col in ["rain_1h", "snow_1h", "clouds_all", "precipitation", "rain", "snowfall"]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0.0)

    for col in ["relative_humidity_2m", "wind_speed_10m", "surface_pressure"]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].bfill().ffill()

    if "weather_main" in df_merged.columns:
        df_merged["weather_main"] = df_merged["weather_main"].fillna("Clear")
    if "holiday_name" in df_merged.columns:
        df_merged["holiday_name"] = df_merged["holiday_name"].fillna("None")
    if "is_holiday" in df_merged.columns:
        df_merged["is_holiday"] = df_merged["is_holiday"].fillna(0).astype(int)
    if "is_weekend" in df_merged.columns:
        df_merged["is_weekend"] = df_merged["is_weekend"].fillna(0).astype(int)

    return df_merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate lag features, rolling statistics, cyclical encodings, and multi-targets.

    Args:
        df: Merged continuous hourly DataFrame.

    Returns:
        DataFrame augmented with engineered feature vectors and prediction targets.
    """
    df = df.copy()
    dt = df["date_time"].dt

    # 1. Calendar & Temporal Features
    df["hour"] = dt.hour
    df["day_of_week"] = dt.day_of_week
    df["month"] = dt.month
    df["day_of_year"] = dt.day_of_year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Morning and evening commuter rush hours
    df["is_morning_rush"] = ((df["day_of_week"] < 5) & (df["hour"].between(7, 9))).astype(int)
    df["is_evening_rush"] = ((df["day_of_week"] < 5) & (df["hour"].between(16, 18))).astype(int)
    df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)

    # 2. Cyclical Encodings (sin / cos)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 3. Weather Features & Adverse Driving Indices
    rain_val = df.get("rain_1h", df.get("rain", 0.0))
    snow_val = df.get("snow_1h", df.get("snowfall", 0.0))
    wind_val = df.get("wind_speed_10m", 0.0)

    df["is_raining"] = (rain_val > 0.1).astype(int)
    df["is_snowing"] = (snow_val > 0.01).astype(int)
    df["is_freezing"] = (df["temp_c"] <= 0.0).astype(int)

    # Weather impact score: 0 (clear) to 1.0 (extreme adverse)
    adverse_score = (
        0.35 * df["is_raining"]
        + 0.50 * df["is_snowing"]
        + 0.25 * (wind_val > 30.0).astype(int)
        + 0.20 * df["is_freezing"]
    )
    df["weather_severity_index"] = adverse_score.clip(0.0, 1.0)

    # 4. Strictly Past Lag Features (no target leakage)
    vol = df["traffic_volume"]
    df["traffic_lag_1h"] = vol.shift(1)
    df["traffic_lag_2h"] = vol.shift(2)
    df["traffic_lag_3h"] = vol.shift(3)
    df["traffic_lag_24h"] = vol.shift(24)
    df["traffic_lag_168h"] = vol.shift(168)  # 1 week ago

    # 5. Rolling Window Features (closed='left' / shifted by 1 to prevent data leakage)
    vol_past = vol.shift(1)
    df["traffic_rolling_mean_3h"] = vol_past.rolling(window=3, min_periods=1).mean()
    df["traffic_rolling_mean_6h"] = vol_past.rolling(window=6, min_periods=1).mean()
    df["traffic_rolling_mean_24h"] = vol_past.rolling(window=24, min_periods=1).mean()
    df["traffic_rolling_std_24h"] = vol_past.rolling(window=24, min_periods=1).std().fillna(0.0)

    # 6. Targets & Derived Domain Metrics
    # Target 1: traffic_volume (continuous)
    # Target 2: Congestion Level (0: Normal, 1: Moderate, 2: Heavy)
    # Based on corridor capacity: Normal (<2500), Moderate (2500-4500), Heavy (>4500)
    df["congestion_level"] = pd.cut(
        df["traffic_volume"],
        bins=[-np.inf, 2500, 4500, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    # Target 3: Estimated Corridor Travel Time (minutes)
    # Bureau of Public Roads (BPR) function for 15-mile I-94 corridor
    # T = T0 * (1 + alpha * (V / C)^beta) with T0=15min, C=6000 veh/hr, alpha=0.15, beta=4
    c_capacity = 6000.0
    v_c_ratio = (df["traffic_volume"] / c_capacity).clip(0.0, 1.3)
    df["travel_time_min"] = (15.0 * (1.0 + 0.15 * np.power(v_c_ratio, 4))).round(2)

    # Target 4: Localized Accident Risk Probability (0.0 to 1.0)
    # Synthesized from volume ratio, adverse weather, night hours, and weekend surge
    is_night = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    risk_raw = (
        0.40 * v_c_ratio
        + 0.35 * df["weather_severity_index"]
        + 0.15 * is_night
        + 0.10 * df["is_holiday"]
    )
    df["accident_risk_index"] = (risk_raw / risk_raw.max()).clip(0.0, 1.0).round(4)

    # Drop leading rows with NaN from lag_168h
    df = df.dropna(subset=["traffic_lag_168h"]).reset_index(drop=True)

    return df


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Execute chronological time-based train/validation/test split and persist datasets.

    Train split: Start through 2016-12-31 (~70%)
    Val split:   2017-01-01 through 2017-12-31 (~15%)
    Test split:  2018-01-01 through End (~15%)

    Args:
        df: Fully processed, feature-engineered DataFrame.
        output_dir: Directory where processed parquet and csv files will be saved.

    Returns:
        Dictionary mapping split names to saved paths.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    train_mask = df["date_time"] <= "2016-12-31 23:00:00"
    val_mask = (df["date_time"] >= "2017-01-01 00:00:00") & (df["date_time"] <= "2017-12-31 23:00:00")
    test_mask = df["date_time"] >= "2018-01-01 00:00:00"

    df_train = df[train_mask].reset_index(drop=True)
    df_val = df[val_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    paths: dict[str, Path] = {}

    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "full": df,
    }

    for name, split_df in splits.items():
        parquet_file = out / f"{name}.parquet"
        csv_file = out / f"{name}.csv"

        split_df.to_parquet(parquet_file, index=False)
        split_df.to_csv(csv_file, index=False)
        paths[name] = parquet_file

        logger.info(
            "Saved %s split: %d rows from %s to %s -> %s",
            name.upper(),
            len(split_df),
            split_df["date_time"].min(),
            split_df["date_time"].max(),
            parquet_file,
        )

    print("\n" + "=" * 65)
    print(" FlowCast Phase 2 Processed Dataset Summary (Time-Based Split)")
    print("=" * 65)
    print(f"Train Set: {len(df_train):,} rows ({df_train['date_time'].min()} to {df_train['date_time'].max()})")
    print(f"Val Set  : {len(df_val):,} rows ({df_val['date_time'].min()} to {df_val['date_time'].max()})")
    print(f"Test Set : {len(df_test):,} rows ({df_test['date_time'].min()} to {df_test['date_time'].max()})")
    print(f"Total Features Engineered: {len(df.columns)} columns")
    print("=" * 65 + "\n")

    return paths


def run_preprocessing(
    raw_traffic_path: str | Path = "data/raw/metro_traffic_raw.csv",
    raw_weather_path: str | Path = "data/raw/weather_openmeteo_raw.csv",
    raw_calendar_path: str | Path = "data/raw/calendar_holidays_raw.csv",
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Execute end-to-end data cleaning, multimodal fusion, and feature engineering.

    Args:
        raw_traffic_path: Path to raw traffic CSV.
        raw_weather_path: Path to raw weather CSV.
        raw_calendar_path: Path to raw calendar CSV.
        output_dir: Destination folder for processed datasets.

    Returns:
        Dictionary of split file paths.
    """
    logger.info("Starting FlowCast preprocessing pipeline...")

    logger.info("Cleaning raw traffic sensor observations...")
    df_traffic_clean = clean_traffic_data(raw_traffic_path)

    logger.info("Loading raw weather and calendar data...")
    df_weather = load_weather_data(raw_weather_path)
    df_calendar = load_calendar_data(raw_calendar_path)

    logger.info("Merging multimodal data on regular hourly index...")
    df_merged = merge_multimodal_data(df_traffic_clean, df_weather, df_calendar)

    logger.info("Engineering temporal, lag, rolling, and target features...")
    df_featured = engineer_features(df_merged)

    logger.info("Performing chronological train/val/test split and saving...")
    saved_paths = split_and_save_data(df_featured, output_dir=output_dir)

    return saved_paths


def main() -> None:
    """CLI entrypoint for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess and engineer features for FlowCast.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_preprocessing(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
