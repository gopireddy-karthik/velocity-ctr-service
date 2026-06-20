from fastapi.testclient import TestClient

from app import app


def test_predict_ctr_known_user():
    with TestClient(app) as client:
        response = client.post(
            "/predict_ctr",
            json={"user_id": "user_101", "ad_id": "ad_123", "ad_position": "banner"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ad_id"] == "ad_123"
    assert 0.0 <= payload["click_probability"] <= 1.0
    assert payload["server_latency_ms"] >= 0.0


def test_predict_ctr_unknown_user_uses_fallback():
    with TestClient(app) as client:
        response = client.post(
            "/predict_ctr",
            json={"user_id": "user_999", "ad_id": "ad_456", "ad_position": "native"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ad_id"] == "ad_456"
    assert 0.0 <= payload["click_probability"] <= 1.0


def test_predict_ctr_invalid_payload():
    with TestClient(app) as client:
        response = client.post(
            "/predict_ctr",
            json={"user_id": "user_101", "ad_position": "banner"},
        )
    assert response.status_code == 422
