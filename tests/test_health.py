"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_service_status() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "regimelab-api"
    assert body["version"] == "0.1.0"
    assert "model_loaded" in body
    assert "latest_experiment_id" in body
