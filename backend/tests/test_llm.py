"""Tests for the LLM interface and the Ollama provider (via HTTP stub)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.config import Settings
from app.llm import (
    FallbackLLMClient,
    GroqLLMClient,
    LLMConfigurationError,
    LLMError,
    OllamaLLMClient,
    build_llm_client,
)
from tests.conftest import FakeLLM


def test_llm_interface_records_prompts() -> None:
    llm = FakeLLM(text="answer [1]")
    response = llm.generate(system_prompt="SYS", user_query="Q", context="EVIDENCE")

    assert response.text == "answer [1]"
    assert response.provider == "fake"
    assert llm.calls == [
        {"system_prompt": "SYS", "user_query": "Q", "context": "EVIDENCE"}
    ]


def test_build_llm_client_rejects_unknown_provider() -> None:
    settings = Settings(llm_provider="openai", _env_file=None)
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        build_llm_client(settings)


class _OllamaChatStub(BaseHTTPRequestHandler):
    behavior: dict = {}

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _OllamaChatStub.behavior["last_request"] = body

        if self.behavior.get("status") != 200:
            self.send_response(self.behavior.get("status", 500))
            self.end_headers()
            return
        if self.behavior.get("malformed"):
            payload = {"unexpected": True}
        elif self.behavior.get("empty"):
            payload = {"message": {"role": "assistant", "content": "  "}}
        else:
            payload = {"message": {"role": "assistant", "content": "Grounded answer [1]."}}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def chat_stub_url():
    _OllamaChatStub.behavior = {"status": 200}
    server = HTTPServer(("127.0.0.1", 0), _OllamaChatStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_ollama_client_sends_system_query_and_context(chat_stub_url) -> None:
    client = OllamaLLMClient(base_url=chat_stub_url, model="test-model", temperature=0.1, timeout=10)
    response = client.generate(system_prompt="SYS", user_query="What is the rule?", context="[1] evidence")

    assert response.text == "Grounded answer [1]."
    assert response.provider == "ollama"
    assert response.model == "test-model"

    sent = _OllamaChatStub.behavior["last_request"]
    assert sent["model"] == "test-model"
    roles = [message["role"] for message in sent["messages"]]
    assert roles == ["system", "user"]  # instructions, then one evidence+question turn
    assert sent["messages"][0]["content"] == "SYS"
    user_content = sent["messages"][1]["content"]
    assert "[1] evidence" in user_content
    assert user_content.index("Evidence:") < user_content.index("Question:")
    assert "What is the rule?" in user_content


def test_ollama_client_wraps_http_errors(chat_stub_url) -> None:
    _OllamaChatStub.behavior = {"status": 500}
    client = OllamaLLMClient(base_url=chat_stub_url, model="m", temperature=0.1, timeout=10)
    with pytest.raises(LLMError, match="HTTP 500"):
        client.generate(system_prompt="s", user_query="q", context="c")


def test_ollama_client_rejects_malformed_and_empty_completions(chat_stub_url) -> None:
    client = OllamaLLMClient(base_url=chat_stub_url, model="m", temperature=0.1, timeout=10)

    _OllamaChatStub.behavior = {"status": 200, "malformed": True}
    with pytest.raises(LLMError, match="malformed"):
        client.generate(system_prompt="s", user_query="q", context="c")

    _OllamaChatStub.behavior = {"status": 200, "empty": True}
    with pytest.raises(LLMError, match="empty completion"):
        client.generate(system_prompt="s", user_query="q", context="c")


def test_ollama_client_fails_cleanly_when_unreachable() -> None:
    client = OllamaLLMClient(base_url="http://127.0.0.1:1", model="m", temperature=0.1, timeout=2)
    with pytest.raises(LLMError, match="Ollama request failed"):
        client.generate(system_prompt="s", user_query="q", context="c")


def test_ollama_client_rejects_negative_temperature() -> None:
    with pytest.raises(LLMError, match="temperature"):
        OllamaLLMClient(base_url="http://x", model="m", temperature=-1, timeout=1)


# ---------------------------------------------------------------------------
# Phase 5: provider selection (Groq primary, Ollama fallback)
# ---------------------------------------------------------------------------


def settings_with(**overrides) -> Settings:
    base = dict(
        llm_provider="auto",
        groq_api_key="",
        ollama_base_url="http://localhost:11434",
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


def test_build_auto_without_groq_key_uses_ollama_directly() -> None:
    client = build_llm_client(settings_with())
    assert isinstance(client, OllamaLLMClient)
    assert client.name == "ollama"


def test_build_auto_with_groq_key_builds_fallback_chain() -> None:
    client = build_llm_client(settings_with(groq_api_key="gsk-test-123"))
    assert isinstance(client, FallbackLLMClient)


def test_build_groq_without_api_key_raises_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        build_llm_client(settings_with(llm_provider="groq"))


def test_build_groq_with_key_returns_groq_client() -> None:
    client = build_llm_client(settings_with(llm_provider="groq", groq_api_key="gsk-test-123"))
    assert isinstance(client, GroqLLMClient)
    assert client.name == "groq"


def test_build_ollama_returns_ollama_client() -> None:
    client = build_llm_client(settings_with(llm_provider="ollama"))
    assert isinstance(client, OllamaLLMClient)


def test_build_unknown_provider_raises_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="Unknown LLM_PROVIDER"):
        build_llm_client(settings_with(llm_provider="openai"))
