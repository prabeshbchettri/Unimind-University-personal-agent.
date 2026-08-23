"""Tests for the LLM layer (Ollama client, stub and factory)."""

import asyncio

import httpx
import pytest

from app.config import Settings
from app.llm import LLMUnavailableError, OllamaLLMClient, StubLLMClient, build_llm


def _run(coro):
    return asyncio.run(coro)


def test_stub_returns_default_answer() -> None:
    client = StubLLMClient(default_answer="Grounded reply.")
    assert client.name == "stub"
    assert client.complete("system", "prompt") == "Grounded reply."
    assert _run(client.close()) is None


def test_stub_uses_responder() -> None:
    client = StubLLMClient(responder=lambda prompt: prompt.upper())
    assert client.complete("system", "hi") == "HI"


def test_factory_builds_stub_when_configured() -> None:
    settings = Settings(llm_backend="stub")
    client = build_llm(settings)
    assert isinstance(client, StubLLMClient)
    assert _run(client.close()) is None


def test_factory_builds_ollama_when_configured() -> None:
    settings = Settings(llm_backend="ollama", ollama_base_url="http://ollama:11434", ollama_model="llama3.1:8b")
    client = build_llm(settings)
    assert isinstance(client, OllamaLLMClient)
    assert client.model == "llama3.1:8b"
    assert _run(client.close()) is None


def test_ollama_returns_generation_text(monkeypatch) -> None:
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"response": " Normalization removes redundancy. \n"})

    monkeypatch.setattr("app.llm.ollama.httpx.post", fake_post)
    client = OllamaLLMClient(base_url="http://ollama:11434", model="llama3.1:8b", timeout=10)

    result = client.complete("system prompt", "Explain normalization.")

    assert result == "Normalization removes redundancy."
    assert captured["url"] == "http://ollama:11434/api/generate"
    assert captured["payload"]["model"] == "llama3.1:8b"
    assert captured["payload"]["stream"] is False


def test_ollama_unreachable_raises(monkeypatch) -> None:
    def fake_post(url, **kwargs):
        raise httpx.ConnectError("connection refused", request=None)

    monkeypatch.setattr("app.llm.ollama.httpx.post", fake_post)
    client = OllamaLLMClient(base_url="http://ollama:11434", model="llama3.1:8b", timeout=5)

    with pytest.raises(LLMUnavailableError):
        client.complete("system", "prompt")


def test_ollama_http_error_raises(monkeypatch) -> None:
    def fake_post(url, **kwargs):
        raise httpx.HTTPStatusError("500 Server Error", request=httpx.Request("POST", url), response=httpx.Response(500))

    monkeypatch.setattr("app.llm.ollama.httpx.post", fake_post)
    client = OllamaLLMClient(base_url="http://ollama:11434", model="llama3.1:8b", timeout=5)

    with pytest.raises(LLMUnavailableError):
        client.complete("system", "prompt")