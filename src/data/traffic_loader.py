"""Traffic dataset downloader and ingestion loader for FlowCast.

This module retrieves and loads the Metro Interstate Traffic Volume dataset
hosted by the UCI Machine Learning Repository (Station: I-94 westbound between
Minneapolis and St. Paul, MN).
"""

from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path
import zipfile
import requests
import pandas as pd

logger = logging.getLogger(__name__)

PRIMARY_UCI_URL = (
    "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
)
FALLBACK_RAW_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/traffic/traffic.csv"
)


def download_metro_traffic_data(
    output_path: str | Path = "data/raw/metro_traffic_raw.csv",
    timeout: int = 60,
) -> Path:
    """Download the Metro Interstate Traffic Volume dataset into data/raw.

    Attempts downloading the official UCI archive first. If unreachable, falls
    back to a verified raw mirror.

    Args:
        output_path: Destination path for the uncompressed CSV.
        timeout: HTTP request timeout in seconds.

    Returns:
        Path to the saved CSV file.

    Raises:
        RuntimeError: If all download sources fail.
    """
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Metro Interstate Traffic Volume dataset...")

    # Attempt primary UCI zip archive
    try:
        response = requests.get(PRIMARY_UCI_URL, timeout=timeout)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            gz_filename = "Metro_Interstate_Traffic_Volume.csv.gz"
            if gz_filename in zf.namelist():
                with zf.open(gz_filename) as gz_stream:
                    with gzip.GzipFile(fileobj=gz_stream) as gz_file:
                        decompressed_content = gz_file.read()
                        dest.write_bytes(decompressed_content)
                        logger.info("Successfully downloaded and extracted from UCI: %s", dest)
                        return dest
    except Exception as exc:
        logger.warning("Primary UCI download failed (%s). Attempting fallback mirror...", exc)

    # Fallback to direct raw CSV mirror
    try:
        fallback_resp = requests.get(FALLBACK_RAW_URL, timeout=timeout)
        fallback_resp.raise_for_status()
        dest.write_bytes(fallback_resp.content)
        logger.info("Successfully retrieved traffic data from fallback mirror: %s", dest)
        return dest
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Metro Interstate Traffic dataset from both primary and fallback URLs: {exc}"
        ) from exc


def load_traffic_data(file_path: str | Path = "data/raw/metro_traffic_raw.csv") -> pd.DataFrame:
    """Load and validate the raw Metro Interstate Traffic Volume dataset.

    Parses the `date_time` column into datetime timestamps, verifies expected
    columns, and sorts rows chronologically.

    Args:
        file_path: Path to the raw traffic CSV file.

    Returns:
        pandas.DataFrame with parsed timestamps and standard column names.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Traffic data file not found at: {path}")

    df = pd.read_csv(path)

    expected_cols = {"date_time", "traffic_volume"}
    if not expected_cols.issubset(df.columns):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Traffic dataset missing expected columns: {missing}")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time"])
    df = df.sort_values(by="date_time").reset_index(drop=True)

    return df
