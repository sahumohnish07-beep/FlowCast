"""Open-Meteo historical and observational weather client.

Retrieves hourly meteorological measurements (temperature, humidity,
precipitation, rain, snowfall, wind speed, weather code) from the Open-Meteo
Historical Weather Archive API without requiring an API key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Default coordinates: Minneapolis-St. Paul, MN (matching I-94 traffic station)
DEFAULT_LATITUDE = 44.9778
DEFAULT_LONGITUDE = -93.2650


def fetch_openmeteo_weather(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    start_date: str = "2012-10-01",
    end_date: str = "2018-09-30",
    output_path: str | Path | None = "data/raw/weather_openmeteo_raw.csv",
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch hourly weather records from the Open-Meteo Archive API.

    Args:
        latitude: Geographical latitude (WGS84).
        longitude: Geographical longitude (WGS84).
        start_date: Query start date in ISO format (YYYY-MM-DD).
        end_date: Query end date in ISO format (YYYY-MM-DD).
        output_path: Optional destination path to persist as CSV.
        timeout: HTTP request timeout in seconds.

    Returns:
        pandas.DataFrame containing hourly weather measurements.

    Raises:
        requests.RequestException: If the API call fails or encounters HTTP errors.
        ValueError: If returned JSON schema does not match hourly format.
    """
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,"
            "precipitation,rain,snowfall,weather_code,surface_pressure,"
            "wind_speed_10m,wind_direction_10m"
        ),
        "timezone": "auto",
    }

    logger.info(
        "Requesting Open-Meteo weather for lat=%.4f, lon=%.4f from %s to %s",
        latitude,
        longitude,
        start_date,
        end_date,
    )

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError("Invalid Open-Meteo response structure: 'hourly.time' missing.")

    hourly_data = payload["hourly"]
    df = pd.DataFrame(hourly_data)
    df = df.rename(columns={"time": "date_time"})
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values(by="date_time").reset_index(drop=True)

    if output_path is not None:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved Open-Meteo weather dataset (%d rows) to %s", len(df), out)

    return df


def load_weather_data(
    file_path: str | Path = "data/raw/weather_openmeteo_raw.csv",
) -> pd.DataFrame:
    """Load and validate the raw Open-Meteo weather dataset.

    Args:
        file_path: Path to the weather CSV file.

    Returns:
        pandas.DataFrame with parsed timestamps and validated columns.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Weather data file not found at: {path}")

    df = pd.read_csv(path)
    if "date_time" not in df.columns:
        raise ValueError("Weather dataset missing 'date_time' column.")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time"])
    df = df.sort_values(by="date_time").reset_index(drop=True)

    return df

