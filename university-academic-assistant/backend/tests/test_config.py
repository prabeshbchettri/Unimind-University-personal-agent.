"""Tests for centralized configuration loading."""

from app.config import Settings


def test_settings_defaults_are_sane() -> None:
    s = Settings()
    assert s.app_name == "University Academic Assistant"
    assert s.api_port == 8000
    assert s.ollama_model == "llama3.1:8b"


def test_settings_accept_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("APP_NAME", "Overridden")
    monkeypatch.setenv("API_PORT", "9090")
    s = Settings()
    assert s.app_name == "Overridden"
    assert s.api_port == 9090


def test_settings_configuration_fields_exist() -> None:
    """Future-component config fields are present even if unused in Phase 1."""
    s = Settings()
    for field in (
        "database_url",
        "qdrant_url",
        "qdrant_api_key",
        "neo4j_uri",
        "neo4j_username",
        "neo4j_password",
        "ollama_base_url",
        "ollama_model",
        "embedding_model",
    ):
        assert hasattr(s, field)
