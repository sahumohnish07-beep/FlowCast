# FlowCast Data Sources & Ingestion Specifications

This document outlines the source provenance, collection methodologies, schemas, and license terms for all data ingested in **FlowCast: AI-Powered Smart Traffic Flow Prediction Platform**.

---

## 1. Metro Interstate Traffic Volume Dataset

- **Origin**: Minnesota Department of Transportation (MnDOT) ATR Station 301, hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume).
- **Location**: Interstate 94 (I-94) westbound, approximately midway between Minneapolis and St. Paul, MN.
- **Temporal Span**: October 2, 2012 09:00:00 to September 30, 2018 23:00:00.
- **Temporal Resolution**: Hourly intervals (48,204 observations).
- **Target Variable**: `traffic_volume` (Hourly reported traffic volume).
- **Raw File Stored**: `data/raw/metro_traffic_raw.csv` (gitignored).

### Schema & Attributes

| Field | Type | Description | Unit / Range |
| :--- | :--- | :--- | :--- |
| `date_time` | Datetime (UTC/Local) | Hour of observation | `2012-10-02` to `2018-09-30` |
| `traffic_volume` | Integer | Hourly volume of traffic | 0 to 7,280 vehicles/hr |
| `holiday` | Categorical | Categorical US national holiday or `None` | Text |
| `temp` | Float | Average temperature in Kelvin | 0.0 to 310.07 K |
| `rain_1h` | Float | Amount in mm of rain that occurred in the hour | mm |
| `snow_1h` | Float | Amount in mm of snow that occurred in the hour | mm |
| `clouds_all` | Integer | Percentage of cloud cover | 0 to 100% |
| `weather_main` | Categorical | Short textual description of the current weather | Clear, Clouds, Rain, etc. |
| `weather_description` | Categorical | Granular meteorological description | Light rain, overcast clouds, etc. |

---

## 2. Open-Meteo Historical Weather Archive

- **Origin**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
- **Station Coordinates**: Latitude `44.9778° N`, Longitude `-93.2650° W` (Minneapolis-St. Paul metropolitan area).
- **Temporal Span**: October 1, 2012 to September 30, 2018 (Continuous: 52,584 hourly observations).
- **Relevance**: Provides gap-free, high-precision meteorological covariates (relative humidity, wind velocity, surface atmospheric pressure, exact precipitation) to enrich baseline sensor readings.
- **Raw File Stored**: `data/raw/weather_openmeteo_raw.csv` (gitignored).

### Schema & Attributes

| Field | Type | Description | Unit |
| :--- | :--- | :--- | :--- |
| `date_time` | Datetime (ISO 8601) | Timestamp of the hourly measurement | YYYY-MM-DD HH:MM:SS |
| `temperature_2m` | Float | Air temperature at 2 meters above ground | °C |
| `relative_humidity_2m` | Float | Relative humidity at 2 meters | % (0 to 100) |
| `dew_point_2m` | Float | Dew point temperature at 2 meters | °C |
| `precipitation` | Float | Total liquid precipitation (rain + showers + snow water) | mm |
| `rain` | Float | Liquid precipitation accumulated | mm |
| `snowfall` | Float | Snowfall depth accumulated | cm |
| `weather_code` | Integer | WMO standard meteorological code (0-99) | WMO Code |
| `surface_pressure` | Float | Atmospheric air pressure at ground elevation | hPa |
| `wind_speed_10m` | Float | Wind speed at 10 meters above ground | km/h |
| `wind_direction_10m` | Float | Wind direction at 10 meters | Degrees (0° to 360°) |

---

## 3. Calendar & Statutory Holiday Feature Generator

- **Origin**: Synthesized programmatically using the Python [`holidays`](https://pypi.org/project/holidays/) package.
- **Jurisdiction**: United States (`US`) with Minnesota state subdivision (`MN`).
- **Temporal Span**: October 1, 2012 to December 31, 2026 (5,205 calendar days).
- **Relevance**: Traffic patterns strongly correlate with calendar events, day-of-week commute habits, holiday travel, and extended holiday eve traffic surges.
- **Raw File Stored**: `data/raw/calendar_holidays_raw.csv` (gitignored).

### Schema & Attributes

| Field | Type | Description |
| :--- | :--- | :--- |
| `date` | Date (YYYY-MM-DD) | Calendar day |
| `year`, `month`, `day` | Integer | Standard date decomposition components |
| `day_of_week` | Integer | Day of week index (`0 = Monday`, `6 = Sunday`) |
| `day_name` | String | English name of the weekday |
| `quarter` | Integer | Calendar quarter (1 through 4) |
| `day_of_year` | Integer | Sequential day index within year (1 to 366) |
| `week_of_year` | Integer | ISO standard week number (1 to 53) |
| `is_weekend` | Integer (0/1) | Flag indicating Saturday or Sunday |
| `is_holiday` | Integer (0/1) | Flag indicating statutory federal or Minnesota state holiday |
| `holiday_name` | String | Canonical statutory name (e.g., `Thanksgiving`, `Christmas Day`) |
| `is_holiday_eve` | Integer (0/1) | Flag indicating the day directly preceding a statutory holiday |
| `is_month_start` / `is_month_end` | Integer (0/1) | Boundary flags for calendar months |

---

## 4. Derived Modeling Targets & Synthetic Ground Truth Documentation

### A. Congestion Level Thresholds
- **Low Flow (Class 0)**: Traffic volume $< 2,500\text{ veh/hr}$ (Free-flow speed maintained, uninhibited maneuverability).
- **Medium Flow (Class 1)**: Traffic volume between $2,500$ and $4,500\text{ veh/hr}$ (Stable traffic, moderate density).
- **High / Congested (Class 2)**: Traffic volume $> 4,500\text{ veh/hr}$ (Corridor bottleneck approaching or exceeding 6,000 veh/hr lane capacity).

### B. Travel Time Estimation (BPR Function)
Corridor travel time is modeled using the Highway Capacity Manual's Bureau of Public Roads (BPR) relationship calibrated for the 15-mile I-94 corridor:
$$T = T_0 \cdot \left[1 + \alpha \cdot \left(\frac{V}{C}\right)^\beta\right]$$
where $T_0 = 13.85\text{ minutes}$ (free-flow travel time at 65 mph), $C = 6,000\text{ veh/hr}$, $\alpha = 0.15$, $\beta = 4.0$.

### C. Accident Risk Labeling & Provenance
Because point-source accident geocaches for specific highway detector loops require proprietary police dispatch records, FlowCast formulates a continuous hazard probability index grounded in empirical transportation safety literature:
$$\text{Risk} = 0.40 \cdot \left(\frac{V}{C}\right) + 0.35 \cdot (\text{Severity}_{\text{weather}}) + 0.15 \cdot (\text{Night}) + 0.10 \cdot (\text{Holiday})$$
- **High-Risk Threshold**: An index score $\ge 0.60$ marks hazardous driving conditions (representing the top $\sim 19\%$ most dangerous hours).
- **Class Imbalance Resolution**: Addressed using Synthetic Minority Over-sampling Technique (SMOTE) on training partitions to equalize representation and boost incident recall.

---

## 5. Pipeline Reproducibility

To re-run and verify the download and synthesis of all raw datasets from scratch:

```bash
# Using the virtual environment:
python src/data/download.py

# Or specifying a custom destination:
python -m src.data.download --output-dir data/raw
```

