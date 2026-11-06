"""Pydantic request and response schemas for FlowCast FastAPI endpoints."""

from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field


class TrafficPredictionInput(BaseModel):
    """Input payload containing temporal, meteorological, and traffic lag features.

    Missing derived features (such as cyclical sine/cosine transforms or rush hour flags)
    are computed dynamically if omitted.
    """

    # Temporal & Calendar (Required)
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0 to 23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(default=10, ge=1, le=12, description="Month of the year (1 to 12)")
    is_holiday: int = Field(default=0, ge=0, le=1, description="Statutory holiday flag (0 or 1)")
    is_holiday_eve: int = Field(default=0, ge=0, le=1, description="Day before statutory holiday flag (0 or 1)")

    # Meteorological Covariates
    temp_c: float = Field(default=12.5, description="Ambient air temperature in Celsius")
    relative_humidity_2m: float = Field(default=60.0, ge=0.0, le=100.0, description="Relative humidity percentage")
    precipitation: float = Field(default=0.0, ge=0.0, description="Liquid precipitation in mm")
    wind_speed_10m: float = Field(default=12.0, ge=0.0, description="Wind speed at 10 meters in km/h")
    clouds_all: float = Field(default=40.0, ge=0.0, le=100.0, description="Cloud cover percentage")

    # Time-series Traffic Lags (Vehicles / hour)
    traffic_lag_1h: float = Field(default=3200.0, ge=0.0, description="Observed traffic volume 1 hour ago")
    traffic_lag_2h: float = Field(default=3150.0, ge=0.0, description="Observed traffic volume 2 hours ago")
    traffic_lag_3h: float = Field(default=3100.0, ge=0.0, description="Observed traffic volume 3 hours ago")
    traffic_lag_24h: float = Field(default=3300.0, ge=0.0, description="Observed traffic volume 24 hours ago")
    traffic_lag_168h: float = Field(default=3250.0, ge=0.0, description="Observed traffic volume 168 hours ago (last week)")

    # Rolling window summary statistics
    traffic_rolling_mean_3h: float = Field(default=3150.0, ge=0.0, description="Rolling 3-hour mean volume")
    traffic_rolling_mean_6h: float = Field(default=3100.0, ge=0.0, description="Rolling 6-hour mean volume")
    traffic_rolling_mean_24h: float = Field(default=3200.0, ge=0.0, description="Rolling 24-hour mean volume")
    traffic_rolling_std_24h: float = Field(default=850.0, ge=0.0, description="Rolling 24-hour standard deviation")

    def to_feature_dict(self, required_columns: list[str]) -> dict[str, float]:
        """Convert input model into a complete feature vector aligned with required columns."""
        is_weekend = int(self.day_of_week >= 5)
        is_morning_rush = int((self.day_of_week < 5) and (7 <= self.hour <= 9))
        is_evening_rush = int((self.day_of_week < 5) and (16 <= self.hour <= 18))
        is_rush_hour = int(is_morning_rush or is_evening_rush)

        hour_sin = math.sin(2 * math.pi * self.hour / 24.0)
        hour_cos = math.cos(2 * math.pi * self.hour / 24.0)
        day_sin = math.sin(2 * math.pi * self.day_of_week / 7.0)
        day_cos = math.cos(2 * math.pi * self.day_of_week / 7.0)
        month_sin = math.sin(2 * math.pi * self.month / 12.0)
        month_cos = math.cos(2 * math.pi * self.month / 12.0)

        is_raining = int(self.precipitation > 0.1)
        is_freezing = int(self.temp_c <= 0.0)
        is_snowing = int(self.precipitation > 0.05 and is_freezing)

        weather_severity_index = min(
            1.0,
            0.35 * is_raining + 0.50 * is_snowing + 0.25 * int(self.wind_speed_10m > 30.0) + 0.20 * is_freezing,
        )

        all_computed: dict[str, Any] = {
            "hour": float(self.hour),
            "day_of_week": float(self.day_of_week),
            "month": float(self.month),
            "is_weekend": float(is_weekend),
            "is_holiday": float(self.is_holiday),
            "is_holiday_eve": float(self.is_holiday_eve),
            "is_morning_rush": float(is_morning_rush),
            "is_evening_rush": float(is_evening_rush),
            "is_rush_hour": float(is_rush_hour),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "temp_c": float(self.temp_c),
            "clouds_all": float(self.clouds_all),
            "relative_humidity_2m": float(self.relative_humidity_2m),
            "precipitation": float(self.precipitation),
            "wind_speed_10m": float(self.wind_speed_10m),
            "is_raining": float(is_raining),
            "is_snowing": float(is_snowing),
            "is_freezing": float(is_freezing),
            "weather_severity_index": weather_severity_index,
            "traffic_lag_1h": float(self.traffic_lag_1h),
            "traffic_lag_2h": float(self.traffic_lag_2h),
            "traffic_lag_3h": float(self.traffic_lag_3h),
            "traffic_lag_24h": float(self.traffic_lag_24h),
            "traffic_lag_168h": float(self.traffic_lag_168h),
            "traffic_rolling_mean_3h": float(self.traffic_rolling_mean_3h),
            "traffic_rolling_mean_6h": float(self.traffic_rolling_mean_6h),
            "traffic_rolling_mean_24h": float(self.traffic_rolling_mean_24h),
            "traffic_rolling_std_24h": float(self.traffic_rolling_std_24h),
        }

        # Filter strictly to expected columns in the order expected by the model
        return {col: all_computed.get(col, 0.0) for col in required_columns}


class TrafficVolumeResponse(BaseModel):
    """Response schema for traffic volume forecasting."""

    predicted_traffic_volume: float = Field(..., description="Estimated vehicles per hour")
    unit: str = "vehicles/hour"
    confidence_interval: dict[str, float] = Field(..., description="Estimated 95% forecast interval")
    model_name: str
    timestamp: str


class CongestionResponse(BaseModel):
    """Response schema for road congestion classification."""

    predicted_congestion_level: str = Field(..., description="Low, Medium, or High")
    congestion_code: int = Field(..., description="0=Low, 1=Medium, 2=High")
    probabilities: dict[str, float] = Field(..., description="Softmax class probabilities")
    model_name: str
    timestamp: str


class TravelTimeResponse(BaseModel):
    """Response schema for corridor travel time estimation."""

    estimated_travel_time_min: float = Field(..., description="Estimated travel time in minutes")
    free_flow_travel_time_min: float = Field(..., description="Free flow travel time without congestion")
    estimated_delay_min: float = Field(..., description="Travel delay above free flow")
    corridor_length_miles: float = 15.0
    average_speed_mph: float = Field(..., description="Estimated corridor average speed")
    model_name: str
    timestamp: str


class AccidentRiskResponse(BaseModel):
    """Response schema for localized accident risk evaluation."""

    is_high_risk: bool = Field(..., description="True if conditions meet hazardous incident threshold")
    risk_probability: float = Field(..., description="Estimated accident likelihood (0.0 to 1.0)")
    risk_level: str = Field(..., description="Low, Medium, High, or Critical")
    key_contributing_factors: list[str] = Field(..., description="Explanatory risk factors")
    model_name: str
    timestamp: str


class OperationalRecommendation(BaseModel):
    """Individual rule-based operational recommendation for traffic management centers."""

    category: str = Field(..., description="Signal Timing, Highway Diversion, Speed Limit, or Safety Alert")
    action: str = Field(..., description="Recommended tactical action")
    priority: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")


class RecommendationResponse(BaseModel):
    """Comprehensive response schema for intelligent traffic authority recommendations."""

    corridor: str = "I-94 Westbound (Minneapolis - St. Paul)"
    congestion_level: str
    accident_risk_level: str
    recommendations: list[OperationalRecommendation]
    action_summary: str
    generated_at: str


class HealthResponse(BaseModel):
    """Response schema for API health monitoring."""

    status: str
    version: str
    environment: str
    models_loaded: dict[str, bool]
    timestamp: str
