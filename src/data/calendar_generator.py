"""Calendar and statutory holiday feature generator.

Synthesizes comprehensive calendar metadata and statutory holiday flags using
the `holidays` Python package for temporal traffic analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
import holidays
import pandas as pd

logger = logging.getLogger(__name__)


def generate_calendar_data(
    start_date: str = "2012-10-01",
    end_date: str = "2026-12-31",
    country: str = "US",
    subdiv: str | None = "MN",
    output_path: str | Path | None = "data/raw/calendar_holidays_raw.csv",
) -> pd.DataFrame:
    """Generate daily calendar records with statutory holiday indicators.

    Args:
        start_date: Range start date in ISO format (YYYY-MM-DD).
        end_date: Range end date in ISO format (YYYY-MM-DD).
        country: ISO country code for statutory holidays (default: 'US').
        subdiv: State / provincial subdivision (default: 'MN' for Minnesota).
        output_path: Optional file path to persist the generated CSV.

    Returns:
        pandas.DataFrame with calendar and holiday indicators.
    """
    logger.info("Generating calendar data from %s to %s (%s-%s)", start_date, end_date, country, subdiv)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    start_year = date_range[0].year
    end_year = date_range[-1].year

    # Retrieve official holiday lookup dictionary
    official_holidays = holidays.country_holidays(
        country=country,
        subdiv=subdiv,
        years=range(start_year, end_year + 1),
    )

    records: list[dict[str, object]] = []
    for dt in date_range:
        current_date = dt.date()
        holiday_name = official_holidays.get(current_date)
        is_holiday = int(holiday_name is not None)

        # Check if the next day is a holiday (holiday eve)
        next_day = (dt + pd.Timedelta(days=1)).date()
        is_holiday_eve = int(next_day in official_holidays)

        records.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "day_of_week": dt.day_of_week,  # 0=Monday, 6=Sunday
                "day_name": dt.day_name(),
                "quarter": dt.quarter,
                "day_of_year": dt.day_of_year,
                "week_of_year": int(dt.isocalendar().week),
                "is_weekend": int(dt.day_of_week >= 5),
                "is_holiday": is_holiday,
                "holiday_name": str(holiday_name) if holiday_name else "None",
                "is_holiday_eve": is_holiday_eve,
                "is_month_start": int(dt.is_month_start),
                "is_month_end": int(dt.is_month_end),
            }
        )

    df = pd.DataFrame(records)

    if output_path is not None:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved calendar & holiday dataset (%d days) to %s", len(df), out)

    return df


def load_calendar_data(
    file_path: str | Path = "data/raw/calendar_holidays_raw.csv",
) -> pd.DataFrame:
    """Load and validate the raw calendar and holiday dataset.

    Args:
        file_path: Path to the calendar CSV file.

    Returns:
        pandas.DataFrame with parsed date column.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Calendar data file not found at: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Calendar dataset missing 'date' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    return df
