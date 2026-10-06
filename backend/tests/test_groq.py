"""Tests for the Groq LLM client (HTTP calls served by a local stub).

No real network and no real API key are used: the stub speaks the Groq
chat-completions contract. Error mapping (auth, timeout, unavailable,
generation) and the request payload (Bearer token, messages) are all asserted.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.llm import (
    GroqLLMClient,
    LLMConfigurationError,
    LLMGenerationError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm import build_user_content


class _GroqStub(BaseHTTPRequestHandler):
    """Local stub for ``POST /chat/completions``."""

    behavior: dict = {}
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _GroqStub.requests.append({"headers": dict(self.headers), "body": body})

        if self.behavior.get("slow"):
            time.sleep(0.6)
        if self.behavior.get("status", 200) != 200:
            self.send_response(self.behavior["status"])
            self.end_headers()
            return
        if self.behavior.get("malformed"):
            payload = {"unexpected": True}
        elif self.behavior.get("empty"):
            payload = {"choices": [{"message": {"role": "assistant", "content": "  "}}]}
        elif self.behavior.get("no_choices"):
            payload = {"choices": []}
        else:
            payload = {"choices": [{"message": {"role": "assistant", "content": "Grounded answer [1]."}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args) -> None:  # keep test output clean
        pass


@pytest.fixture(scope="module")
def groq_stub_url() -> str:
    """One stub server shared by every test in this module (started once)."""
    server = HTTPServer(("127.0.0.1", 0), _GroqStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _reset_groq_stub():
    """Reset per-test behavior and recorded requests."""
    _GroqStub.behavior = {"status": 200}
    _GroqStub.requests = []
    yield


API_KEY = "test-key-not-real"


def make_client(base_url: str, timeout: float = 10.0) -> GroqLLMClient:
    return GroqLLMClient(
        api_key=API_KEY,
        model="openai/gpt-oss-20b",
        base_url=base_url,
        temperature=0.1,
        timeout=timeout,
    )


def test_groq_client_generates_grounded_answer(groq_stub_url) -> None:
    client = make_client(groq_stub_url)

    response = client.generate(
        system_prompt="SYSTEM", user_query="What is the rule?", context="[1] evidence"
    )

    assert response.text == "Grounded answer [1]."
    assert response.provider == "groq"
    assert response.model == "openai/gpt-oss-20b"


def test_groq_client_sends_expected_payload(groq_stub_url) -> None:
    make_client(groq_stub_url).generate(
        system_prompt="SYSTEM", user_query="What is the rule?", context="[1] evidence"
    )

    sent = _GroqStub.requests[0]
    assert sent["headers"]["Authorization"] == f"Bearer {API_KEY}"
    # Groq's Cloudflare edge 403s the default Python-urllib UA (error 1010).
    assert sent["headers"]["User-Agent"].startswith("unimind-rag/")
    body = sent["body"]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["stream"] is False
    roles = [message["role"] for message in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][0]["content"] == "SYSTEM"
    assert body["messages"][1]["content"] == build_user_content("[1] evidence", "What is the rule?")


def test_groq_wraps_authentication_failure_as_configuration_error(groq_stub_url) -> None:
    _GroqStub.behavior = {"status": 401}
    with pytest.raises(LLMConfigurationError, match="authentication"):
        make_client(groq_stub_url).generate(system_prompt="s", user_query="q", context="c")
    _GroqStub.behavior = {"status": 403}
    with pytest.raises(LLMConfigurationError, match="authentication"):
        make_client(groq_stub_url).generate(system_prompt="s", user_query="q", context="c")


def test_groq_wraps_overload_and_server_errors_as_unavailable(groq_stub_url) -> None:
    for status in (429, 500, 503):
        _GroqStub.behavior = {"status": status}
        with pytest.raises(LLMUnavailableError, match="service unavailable"):
            make_client(groq_stub_url).generate(system_prompt="s", user_query="q", context="c")


def test_groq_maps_timeout_status_to_timeout_error(groq_stub_url) -> None:
    _GroqStub.behavior = {"status": 408}
    with pytest.raises(LLMTimeoutError, match="timed out"):
        make_client(groq_stub_url).generate(system_prompt="s", user_query="q", context="c")


def test_groq_times_out_when_server_is_slow(groq_stub_url) -> None:
    _GroqStub.behavior = {"slow": True}
    with pytest.raises(LLMTimeoutError, match="timed out"):
        make_client(groq_stub_url, timeout=0.3).generate(
            system_prompt="s", user_query="q", context="c"
        )


def test_groq_fails_cleanly_when_unreachable() -> None:
    client = make_client(base_url="http://127.0.0.1:1", timeout=2)
    with pytest.raises(LLMUnavailableError, match="Groq request failed"):
        client.generate(system_prompt="s", user_query="q", context="c")


def test_groq_rejects_malformed_and_empty_completions(groq_stub_url) -> None:
    client = make_client(groq_stub_url)

    _GroqStub.behavior = {"malformed": True}
    with pytest.raises(LLMGenerationError, match="malformed"):
        client.generate(system_prompt="s", user_query="q", context="c")

    _GroqStub.behavior = {"empty": True}
    with pytest.raises(LLMGenerationError, match="empty completion"):
        client.generate(system_prompt="s", user_query="q", context="c")

    _GroqStub.behavior = {"no_choices": True}
    with pytest.raises(LLMGenerationError, match="malformed"):
        client.generate(system_prompt="s", user_query="q", context="c")


def test_groq_requires_api_key_and_model() -> None:
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        GroqLLMClient(api_key="", model="m", base_url="http://x")
    with pytest.raises(LLMConfigurationError, match="GROQ_MODEL"):
        GroqLLMClient(api_key="k", model="", base_url="http://x")
    with pytest.raises(LLMConfigurationError, match="GROQ_BASE_URL"):
        GroqLLMClient(api_key="k", model="m", base_url="")


def test_groq_rejects_negative_temperature() -> None:
    with pytest.raises(LLMConfigurationError, match="temperature"):
        GroqLLMClient(api_key="k", model="m", base_url="http://x", temperature=-1)


def test_groq_errors_do_not_echo_the_api_key(groq_stub_url) -> None:
    _GroqStub.behavior = {"status": 500}
    try:
        make_client(groq_stub_url).generate(system_prompt="s", user_query="q", context="c")
    except LLMUnavailableError as exc:
        assert API_KEY not in str(exc)
    else:
        raise AssertionError("expected an LLMUnavailableError")