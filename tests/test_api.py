"""Unit and integration tests for FlowCast FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with application lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify health endpoint returns healthy status and models are loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert all(data["models_loaded"].values())


def test_predict_traffic_volume(client: TestClient):
    """Verify traffic volume prediction returns realistic estimates."""
    payload = {
        "hour": 8,
        "day_of_week": 1,  # Tuesday morning rush
        "month": 10,
        "is_holiday": 0,
        "temp_c": 12.0,
        "precipitation": 0.0,
        "traffic_lag_1h": 4200.0,
        "traffic_lag_24h": 4500.0,
    }
    response = client.post("/predict/traffic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_traffic_volume" in data
    assert data["predicted_traffic_volume"] > 0
    assert "confidence_interval" in data
    assert data["confidence_interval"]["lower_95"] <= data["predicted_traffic_volume"]
    assert data["confidence_interval"]["upper_95"] >= data["predicted_traffic_volume"]


def test_predict_congestion_level(client: TestClient):
    """Verify congestion level classification produces expected tiers."""
    payload = {
        "hour": 17,
        "day_of_week": 2,  # Wednesday evening peak
        "month": 10,
        "is_holiday": 0,
        "traffic_lag_1h": 5200.0,
    }
    response = client.post("/predict/congestion", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_congestion_level"] in ["Low", "Medium", "High"]
    assert data["congestion_code"] in [0, 1, 2]
    assert len(data["probabilities"]) == 3
    prob_sum = sum(data["probabilities"].values())
    assert pytest.approx(prob_sum, rel=1e-2) == 1.0


def test_predict_travel_time(client: TestClient):
    """Verify corridor travel time estimation adheres to physics boundaries."""
    payload = {
        "hour": 17,
        "day_of_week": 3,
        "traffic_lag_1h": 4800.0,
        "precipitation": 1.2,
    }
    response = client.post("/predict/travel-time", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["estimated_travel_time_min"] >= data["free_flow_travel_time_min"]
    assert data["average_speed_mph"] > 0
    assert data["corridor_length_miles"] == 15.0


def test_predict_accident_risk(client: TestClient):
    """Verify accident risk evaluation detects adverse conditions."""
    payload = {
        "hour": 23,  # Night
        "day_of_week": 5,  # Weekend night
        "temp_c": -5.0,  # Freezing
        "precipitation": 3.0,  # Snow / freezing rain
        "traffic_lag_1h": 4000.0,
    }
    response = client.post("/predict/accident-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["is_high_risk"], bool)
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]
    assert len(data["key_contributing_factors"]) > 0


def test_recommendations_endpoint(client: TestClient):
    """Verify operational recommendations return actionable guidance."""
    # Test High Congestion & High Accident Risk
    response = client.get("/recommendations?congestion_level=High&accident_risk=true&weather_severity=0.8&hour=17")
    assert response.status_code == 200
    data = response.json()
    assert data["congestion_level"] == "High"
    assert data["accident_risk_level"] == "HIGH"
    assert len(data["recommendations"]) >= 3

    categories = [r["category"] for r in data["recommendations"]]
    assert "Signal Timing" in categories or "Highway Diversion" in categories
    assert "Variable Speed Limit" in categories or "Safety Alert" in categories


def test_validation_error_on_invalid_input(client: TestClient):
    """Verify FastAPI returns 422 Unprocessable Entity when hour is out of bounds."""
    invalid_payload = {
        "hour": 35,  # Invalid hour
        "day_of_week": 2,
    }
    response = client.post("/predict/traffic", json=invalid_payload)
    assert response.status_code == 422
