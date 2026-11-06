"""FastAPI application for FlowCast AI-Powered Traffic Prediction Platform.

Provides RESTful inference endpoints for traffic volume forecasting, road congestion
level classification, corridor travel time estimation, high accident risk evaluation,
and rule-based operational recommendations for traffic management centers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import logging
from pathlib import Path
import sys

# Ensure FlowCast root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd

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

logger = logging.getLogger("flowcast.api")

MODELS_DIR = Path("models").resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load serialized model artifacts at application startup."""
    logger.info("Initializing FlowCast prediction models from %s...", MODELS_DIR)

    app.state.traffic_bundle = None
    app.state.congestion_bundle = None
    app.state.travel_time_bundle = None
    app.state.accident_bundle = None

    # Load baseline traffic volume model
    traffic_path = MODELS_DIR / "best_baseline_model.joblib"
    if traffic_path.exists():
        app.state.traffic_bundle = joblib.load(traffic_path)
        logger.info("Loaded traffic volume model (%s)", app.state.traffic_bundle.get("model_name"))

    # Load congestion classifier
    congestion_path = MODELS_DIR / "congestion_classifier.joblib"
    if congestion_path.exists():
        app.state.congestion_bundle = joblib.load(congestion_path)
        logger.info("Loaded congestion level classifier.")

    # Load travel time estimator
    travel_path = MODELS_DIR / "travel_time_model.joblib"
    if travel_path.exists():
        app.state.travel_time_bundle = joblib.load(travel_path)
        logger.info("Loaded travel time estimator.")

    # Load accident risk classifier
    accident_path = MODELS_DIR / "accident_risk_classifier.joblib"
    if accident_path.exists():
        app.state.accident_bundle = joblib.load(accident_path)
        logger.info("Loaded accident risk classifier.")

    yield

    logger.info("Shutting down FlowCast API service.")


app = FastAPI(
    title="FlowCast: AI-Powered Smart Traffic Flow Prediction API",
    description=(
        "Production-grade RESTful API forecasting traffic volume, road congestion "
        "severity, corridor travel delays, and adverse accident risks along the "
        "Interstate 94 metropolitan corridor using sensor, weather, and calendar data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Return health status, service metadata, and model readiness indicators."""
    models_status = {
        "traffic_volume_model": app.state.traffic_bundle is not None,
        "congestion_classifier": app.state.congestion_bundle is not None,
        "travel_time_model": app.state.travel_time_bundle is not None,
        "accident_risk_classifier": app.state.accident_bundle is not None,
    }

    all_ready = all(models_status.values())
    status_str = "healthy" if all_ready else "degraded"

    return HealthResponse(
        status=status_str,
        version="1.0.0",
        environment="production",
        models_loaded=models_status,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/traffic", response_model=TrafficVolumeResponse, tags=["Inference"])
def predict_traffic_volume(payload: TrafficPredictionInput) -> TrafficVolumeResponse:
    """Predict expected hourly traffic volume (vehicles per hour)."""
    if app.state.traffic_bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Traffic volume model not loaded on server.",
        )

    bundle = app.state.traffic_bundle
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # Transform payload into aligned feature vector
    feat_dict = payload.to_feature_dict(feature_names)
    X = pd.DataFrame([feat_dict])[feature_names]

    try:
        raw_pred = float(model.predict(X)[0])
        pred_volume = max(0.0, round(raw_pred, 1))

        # 95% prediction interval using model test RMSE (~226 veh/hr)
        rmse = bundle.get("test_metrics", {}).get("Test_RMSE", 226.5)
        interval = {
            "lower_95": max(0.0, round(pred_volume - 1.96 * rmse, 1)),
            "upper_95": round(pred_volume + 1.96 * rmse, 1),
        }

        return TrafficVolumeResponse(
            predicted_traffic_volume=pred_volume,
            unit="vehicles/hour",
            confidence_interval=interval,
            model_name=bundle.get("model_name", "XGBoost Regressor"),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as exc:
        logger.error("Traffic prediction inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {exc}",
        ) from exc


@app.post("/predict/congestion", response_model=CongestionResponse, tags=["Inference"])
def predict_congestion_level(payload: TrafficPredictionInput) -> CongestionResponse:
    """Classify road congestion into discrete tiers: Low, Medium, or High."""
    if app.state.congestion_bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Congestion classification model not loaded on server.",
        )

    bundle = app.state.congestion_bundle
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    classes = bundle.get("classes", ["Low", "Medium", "High"])

    feat_dict = payload.to_feature_dict(feature_names)
    X = pd.DataFrame([feat_dict])[feature_names]

    try:
        pred_code = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]

        prob_dict = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
        pred_label = classes[pred_code] if pred_code < len(classes) else "Unknown"

        return CongestionResponse(
            predicted_congestion_level=pred_label,
            congestion_code=pred_code,
            probabilities=prob_dict,
            model_name="XGBoost Classifier",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as exc:
        logger.error("Congestion prediction inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {exc}",
        ) from exc


@app.post("/predict/travel-time", response_model=TravelTimeResponse, tags=["Inference"])
def predict_travel_time(payload: TrafficPredictionInput) -> TravelTimeResponse:
    """Estimate highway corridor travel time (minutes) and delay over free flow."""
    if app.state.travel_time_bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Travel time estimator model not loaded on server.",
        )

    bundle = app.state.travel_time_bundle
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    feat_dict = payload.to_feature_dict(feature_names)
    X = pd.DataFrame([feat_dict])[feature_names]

    try:
        raw_pred = float(model.predict(X)[0])
        free_flow_min = 13.85  # 15 miles at 65 mph free-flow speed
        pred_time = max(free_flow_min, round(raw_pred, 2))
        delay = max(0.0, round(pred_time - free_flow_min, 2))

        # Speed calculation (mph)
        avg_speed = round((15.0 / (pred_time / 60.0)), 1) if pred_time > 0 else 65.0

        return TravelTimeResponse(
            estimated_travel_time_min=pred_time,
            free_flow_travel_time_min=round(free_flow_min, 2),
            estimated_delay_min=delay,
            corridor_length_miles=15.0,
            average_speed_mph=avg_speed,
            model_name="XGBoost BPR Hybrid Estimator",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as exc:
        logger.error("Travel time prediction inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {exc}",
        ) from exc


@app.post("/predict/accident-risk", response_model=AccidentRiskResponse, tags=["Inference"])
def predict_accident_risk(payload: TrafficPredictionInput) -> AccidentRiskResponse:
    """Evaluate accident occurrence likelihood and identify primary risk drivers."""
    if app.state.accident_bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Accident risk classifier model not loaded on server.",
        )

    bundle = app.state.accident_bundle
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    feat_dict = payload.to_feature_dict(feature_names)
    X = pd.DataFrame([feat_dict])[feature_names]

    try:
        pred_binary = int(model.predict(X)[0])
        prob_high_risk = float(model.predict_proba(X)[0][1])

        # Categorize risk level
        if prob_high_risk >= 0.75:
            risk_level = "Critical"
        elif prob_high_risk >= 0.50:
            risk_level = "High"
        elif prob_high_risk >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Diagnose contributing risk factors
        factors: list[str] = []
        if payload.precipitation > 0.5:
            factors.append("Active heavy precipitation / slick road surfaces")
        if payload.temp_c <= 0.0:
            factors.append("Sub-zero freezing temperatures (black ice risk)")
        if payload.hour < 6 or payload.hour > 22:
            factors.append("Low ambient daylight / nighttime visibility impairment")
        if payload.traffic_lag_1h > 4500:
            factors.append("High vehicular volume density approaching corridor saturation")
        if payload.is_holiday or payload.is_holiday_eve:
            factors.append("Holiday travel surge / irregular commuting patterns")

        if not factors:
            factors.append("Normal environmental and traffic conditions")

        return AccidentRiskResponse(
            is_high_risk=bool(pred_binary == 1 or prob_high_risk >= 0.50),
            risk_probability=round(prob_high_risk, 4),
            risk_level=risk_level,
            key_contributing_factors=factors,
            model_name="XGBoost SMOTE Imbalance Classifier",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as exc:
        logger.error("Accident risk prediction inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {exc}",
        ) from exc


def generate_operational_recommendations(
    congestion_level: str,
    is_high_risk: bool,
    weather_severity: float = 0.0,
    hour: int = 12,
) -> list[OperationalRecommendation]:
    """Rule-based decision logic for traffic operations centers (TMC)."""
    recommendations: list[OperationalRecommendation] = []

    # Rule Set 1: Heavy Congestion Mitigations
    if congestion_level.lower() == "high":
        recommendations.append(
            OperationalRecommendation(
                category="Signal Timing",
                action="Deploy adaptive cycle extensions (+15s green phase) on arterial off-ramps (e.g., Hwy 280 & Snelling Ave) to clear corridor queues.",
                priority="HIGH",
            )
        )
        recommendations.append(
            OperationalRecommendation(
                category="Highway Diversion",
                action="Activate Dynamic Message Signs (DMS) recommending Alternate Bypass Route (via I-694/Hwy 36) for westbound transit.",
                priority="HIGH",
            )
        )
        recommendations.append(
            OperationalRecommendation(
                category="Ramp Metering",
                action="Trigger restrictive ramp meter rates at all upstream on-ramps to preserve mainline throughput velocity.",
                priority="MEDIUM",
            )
        )

    # Rule Set 2: Moderate Congestion Guidance
    elif congestion_level.lower() == "medium":
        recommendations.append(
            OperationalRecommendation(
                category="Signal Timing",
                action="Maintain standard dynamic corridor coordination; monitor on-ramp queue detection sensors.",
                priority="MEDIUM",
            )
        )
        recommendations.append(
            OperationalRecommendation(
                category="Ramp Metering",
                action="Enable standard peak-period ramp metering cadence.",
                priority="LOW",
            )
        )

    # Rule Set 3: High Accident Risk & Weather Adversity
    if is_high_risk or weather_severity >= 0.5:
        recommendations.append(
            OperationalRecommendation(
                category="Variable Speed Limit",
                action="Lower Variable Speed Limit (VSL) indicators from 65 mph to 45-50 mph to reduce rear-end crash probability.",
                priority="CRITICAL" if weather_severity >= 0.7 else "HIGH",
            )
        )
        recommendations.append(
            OperationalRecommendation(
                category="Safety Alert",
                action="Broadcast slippery road hazard / reduced visibility alerts via Minnesota 511 traveler feed and overhead overhead DMS.",
                priority="HIGH",
            )
        )
        recommendations.append(
            OperationalRecommendation(
                category="Incident Response",
                action="Pre-position Freeway Incident Response Safety Team (FIRST) patrol units near high-incident interchanges (Huron Blvd / Vandalia St).",
                priority="HIGH",
            )
        )

    # Rule Set 4: Nighttime Travel Precautions
    if hour < 6 or hour > 22:
        recommendations.append(
            OperationalRecommendation(
                category="Illumination & Safety",
                action="Ensure 100% operational highway illumination and verify wrong-way vehicle sensor triggers.",
                priority="LOW",
            )
        )

    # Default baseline if flow is optimal
    if not recommendations:
        recommendations.append(
            OperationalRecommendation(
                category="System Status",
                action="Corridor operating under optimal free-flow conditions. Maintain scheduled automated surveillance.",
                priority="LOW",
            )
        )

    return recommendations


@app.get("/recommendations", response_model=RecommendationResponse, tags=["Decision Support"])
def get_recommendations(
    congestion_level: str = Query("Medium", description="Estimated congestion tier: Low, Medium, or High"),
    accident_risk: bool = Query(False, description="Hazardous accident risk indicator"),
    weather_severity: float = Query(0.0, ge=0.0, le=1.0, description="Normalized weather severity (0.0 to 1.0)"),
    hour: int = Query(17, ge=0, le=23, description="Hour of the day"),
) -> RecommendationResponse:
    """Retrieve tactical operational guidance for traffic management authorities."""
    recs = generate_operational_recommendations(
        congestion_level=congestion_level,
        is_high_risk=accident_risk,
        weather_severity=weather_severity,
        hour=hour,
    )

    action_summary = (
        f"Corridor experiencing {congestion_level.upper()} congestion with "
        f"{'ELEVATED' if accident_risk else 'NORMAL'} incident risk. "
        f"{len(recs)} operational actions deployed."
    )

    return RecommendationResponse(
        corridor="I-94 Westbound (Minneapolis - St. Paul)",
        congestion_level=congestion_level,
        accident_risk_level="HIGH" if accident_risk else "LOW",
        recommendations=recs,
        action_summary=action_summary,
        generated_at=datetime.now().isoformat(),
    )


def main() -> None:
    """Run uvicorn server directly from CLI."""
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()

