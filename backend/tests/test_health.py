"""Tests for the system endpoints.

These run against the real FastAPI application via the test client, so a
failure here means the app itself does not start or does not route correctly.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "Adaptive University RAG Assistant"
    assert body["environment"]


def test_health_content_type_is_json() -> None:
    response = client.get("/health")

    assert response.headers["content-type"].startswith("application/json")


def test_unknown_route_returns_404() -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
