"""Script to verify deployment start command and API endpoints."""

import os
import subprocess
import sys
import time
import requests

def test_endpoints():
    test_port = 8008
    env = os.environ.copy()
    env["PORT"] = str(test_port)
    env["ENVIRONMENT"] = "production"

    print(f"Launching uvicorn on port {test_port} with start command...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(test_port),
    ]

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        base_url = f"http://127.0.0.1:{test_port}"
        print("Waiting for server to become healthy...")
        healthy = False
        for _ in range(30):
            try:
                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                time.sleep(0.5)

        if not healthy:
            print("Server failed to respond to /health!")
            return False

        # 1. Test /health
        r_health = requests.get(f"{base_url}/health")
        print("GET /health ->", r_health.status_code, r_health.json())
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "healthy"

        payload = {
            "hour": 17,
            "day_of_week": 0,
            "month": 10,
            "is_holiday": 0,
            "is_holiday_eve": 0,
            "temp_c": 12.0,
            "precipitation": 0.0,
            "wind_speed_10m": 15.0,
            "relative_humidity_2m": 50.0,
            "traffic_lag_1h": 4800.0,
            "traffic_lag_24h": 4950.0,
        }

        # 2. Test /predict/traffic
        r_traffic = requests.post(f"{base_url}/predict/traffic", json=payload)
        print("POST /predict/traffic ->", r_traffic.status_code, r_traffic.json()["predicted_traffic_volume"])
        assert r_traffic.status_code == 200

        # 3. Test /predict/congestion
        r_cong = requests.post(f"{base_url}/predict/congestion", json=payload)
        print("POST /predict/congestion ->", r_cong.status_code, r_cong.json()["predicted_congestion_level"])
        assert r_cong.status_code == 200

        # 4. Test /predict/travel-time
        r_time = requests.post(f"{base_url}/predict/travel-time", json=payload)
        print("POST /predict/travel-time ->", r_time.status_code, r_time.json()["estimated_travel_time_min"])
        assert r_time.status_code == 200

        # 5. Test /predict/accident-risk
        r_risk = requests.post(f"{base_url}/predict/accident-risk", json=payload)
        print("POST /predict/accident-risk ->", r_risk.status_code, r_risk.json()["risk_probability"])
        assert r_risk.status_code == 200

        # 6. Test /recommendations
        r_rec = requests.get(f"{base_url}/recommendations?congestion_level=High&accident_risk=true&hour=17")
        print("GET /recommendations ->", r_rec.status_code, len(r_rec.json()["recommendations"]), "advisories")
        assert r_rec.status_code == 200

        print("All deployment endpoints successfully verified!")
        return True

    finally:
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == "__main__":
    success = test_endpoints()
    if not success:
        sys.exit(1)
