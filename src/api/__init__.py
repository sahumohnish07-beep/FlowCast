"""FastAPI inference endpoints and schemas for FlowCast."""

from src.api.main import app
from src.api.schemas import (
    AccidentRiskResponse,
    CongestionResponse,
    HealthResponse,
    OperationalRecommendation,
    RecommendationResponse,
    TrafficPredictionInput,
    TrafficVolumeResponse,
    TravelTimeResponse,
)

__all__ = [
    "app",
    "TrafficPredictionInput",
    "TrafficVolumeResponse",
    "CongestionResponse",
    "TravelTimeResponse",
    "AccidentRiskResponse",
    "RecommendationResponse",
    "OperationalRecommendation",
    "HealthResponse",
]
