"""Tests for the ``GET /health`` endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_uses_get_only(client: TestClient) -> None:
    assert client.post("/health").status_code == 405
