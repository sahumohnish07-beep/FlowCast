"""Unit and integration tests for FlowCast Phase 1 data collection modules."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.calendar_generator import generate_calendar_data, load_calendar_data
from src.data.traffic_loader import load_traffic_data
from src.data.weather_client import load_weather_data


def test_calendar_generator(tmp_path: Path):
    """Test holiday and calendar generator logic for correctness."""
    csv_file = tmp_path / "test_calendar.csv"
    df = generate_calendar_data(
        start_date="2020-01-01",
        end_date="2020-01-07",
        country="US",
        subdiv="MN",
        output_path=csv_file,
    )

    assert csv_file.exists()
    assert len(df) == 7
    expected_cols = {
        "date",
        "year",
        "month",
        "day",
        "day_of_week",
        "day_name",
        "is_weekend",
        "is_holiday",
        "holiday_name",
        "is_holiday_eve",
    }
    assert expected_cols.issubset(df.columns)

    # 2020-01-01 is New Year's Day
    new_years_row = df[df["date"] == "2020-01-01"].iloc[0]
    assert new_years_row["is_holiday"] == 1
    assert "New Year" in new_years_row["holiday_name"]


def test_load_calendar_data(tmp_path: Path):
    """Verify loading and parsing of saved calendar CSV."""
    csv_file = tmp_path / "sample_cal.csv"
    generate_calendar_data("2021-07-01", "2021-07-07", output_path=csv_file)

    loaded = load_calendar_data(csv_file)
    assert isinstance(loaded, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
    assert len(loaded) == 7


def test_raw_traffic_file_if_present():
    """Verify raw traffic dataset schema if download has executed."""
    raw_path = Path("data/raw/metro_traffic_raw.csv")
    if not raw_path.exists():
        pytest.skip("data/raw/metro_traffic_raw.csv has not been downloaded yet.")

    df = load_traffic_data(raw_path)
    assert not df.empty
    assert "traffic_volume" in df.columns
    assert "date_time" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date_time"])
    assert df["traffic_volume"].min() >= 0


def test_raw_weather_file_if_present():
    """Verify raw weather dataset schema if fetch has executed."""
    raw_path = Path("data/raw/weather_openmeteo_raw.csv")
    if not raw_path.exists():
        pytest.skip("data/raw/weather_openmeteo_raw.csv has not been fetched yet.")

    df = load_weather_data(raw_path)
    assert not df.empty
    assert "date_time" in df.columns
    assert "temperature_2m" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date_time"])
