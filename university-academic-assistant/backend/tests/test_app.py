"""Tests for basic application startup and routing."""

from fastapi.testclient import TestClient


def test_app_starts_and_lists_routes(client: TestClient) -> None:
    """The app boots and exposes the core route groups."""
    paths = {route.path for route in client.app.routes}
    for expected in ("/health", "/chat", "/documents", "/history", "/openapi.json"):
        assert expected in paths


def test_openapi_schema_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "University Academic Assistant" in response.json()["info"]["title"]


def test_chat_returns_grounded_response(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert isinstance(body["sources"], list)
