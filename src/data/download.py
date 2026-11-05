"""Reproducible download and data generation script for FlowCast Phase 1.

Downloads the Metro Interstate Traffic Volume dataset, fetches historical
weather data from the Open-Meteo API, and synthesizes calendar & statutory
holiday indicators.

Usage:
    python src/data/download.py
    python -m src.data.download --output-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.calendar_generator import generate_calendar_data, load_calendar_data
from src.data.traffic_loader import download_metro_traffic_data, load_traffic_data
from src.data.weather_client import fetch_openmeteo_weather, load_weather_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("flowcast.data.download")


def run_pipeline(output_dir: Path | str = "data/raw") -> dict[str, Path]:
    """Execute the full data collection and generation pipeline.

    Args:
        output_dir: Directory where raw datasets will be written.

    Returns:
        Dictionary mapping dataset keys to their saved file paths.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    traffic_path = out_dir / "metro_traffic_raw.csv"
    weather_path = out_dir / "weather_openmeteo_raw.csv"
    calendar_path = out_dir / "calendar_holidays_raw.csv"

    # Step 1: Download Metro Interstate Traffic Volume dataset
    logger.info("=== [Step 1/3] Downloading Metro Interstate Traffic Dataset ===")
    download_metro_traffic_data(output_path=traffic_path)
    df_traffic = load_traffic_data(traffic_path)
    logger.info(
        "Traffic data verified: %d records from %s to %s (Memory: %.2f MB)",
        len(df_traffic),
        df_traffic["date_time"].min(),
        df_traffic["date_time"].max(),
        df_traffic.memory_usage(deep=True).sum() / (1024 * 1024),
    )

    # Step 2: Query Open-Meteo Historical Weather Archive
    logger.info("=== [Step 2/3] Fetching Open-Meteo Weather Dataset ===")
    fetch_openmeteo_weather(
        latitude=44.9778,
        longitude=-93.2650,
        start_date="2012-10-01",
        end_date="2018-09-30",
        output_path=weather_path,
    )
    df_weather = load_weather_data(weather_path)
    logger.info(
        "Weather data verified: %d hourly records from %s to %s (Memory: %.2f MB)",
        len(df_weather),
        df_weather["date_time"].min(),
        df_weather["date_time"].max(),
        df_weather.memory_usage(deep=True).sum() / (1024 * 1024),
    )

    # Step 3: Generate Calendar & Holiday Features
    logger.info("=== [Step 3/3] Generating Calendar & Holiday Features ===")
    generate_calendar_data(
        start_date="2012-10-01",
        end_date="2026-12-31",
        country="US",
        subdiv="MN",
        output_path=calendar_path,
    )
    df_calendar = load_calendar_data(calendar_path)
    logger.info(
        "Calendar data verified: %d daily records from %s to %s (Holidays: %d)",
        len(df_calendar),
        df_calendar["date"].min(),
        df_calendar["date"].max(),
        df_calendar["is_holiday"].sum(),
    )

    print("\n" + "=" * 60)
    print(" FlowCast Phase 1 Data Collection Summary")
    print("=" * 60)
    print(f"Traffic Volume : {traffic_path} ({len(df_traffic):,} rows, {traffic_path.stat().st_size / 1024:.1f} KB)")
    print(f"Weather Records: {weather_path} ({len(df_weather):,} rows, {weather_path.stat().st_size / 1024:.1f} KB)")
    print(f"Calendar Table : {calendar_path} ({len(df_calendar):,} rows, {calendar_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 60 + "\n")

    return {
        "traffic": traffic_path,
        "weather": weather_path,
        "calendar": calendar_path,
    }


def main() -> None:
    """CLI entrypoint for data download."""
    parser = argparse.ArgumentParser(
        description="Reproduce FlowCast Phase 1 Data Collection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Target directory for downloaded raw datasets (default: data/raw)",
    )
    args = parser.parse_args()

    try:
        run_pipeline(output_dir=args.output_dir)
    except Exception as exc:
        logger.error("Data pipeline execution failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
