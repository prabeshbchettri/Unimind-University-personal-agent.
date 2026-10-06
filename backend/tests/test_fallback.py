"""Tests for Groq -> Ollama provider fallback.

Real Groq/Ollama clients hit two local HTTP stubs, so the entire failover
logic runs without a network. Coverage: success on the primary, failover on
timeout / connection failure / service unavailability, clean failure when
every provider is down, no fallback on configuration errors, and observable
provider relabeling (``ollama_fallback``).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.llm import (
    FallbackLLMClient,
    GroqLLMClient,
    LLMConfigurationError,
    LLMUnavailableError,
    OllamaLLMClient,
)


def make_stub(mode: str) -> type[BaseHTTPRequestHandler]:
    """Create a stub handler class for the Groq or the Ollama response shape."""

    class Stub(BaseHTTPRequestHandler):
        behavior: dict = {"status": 200}
        requests: list[dict] = []

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            Stub.requests.append(body)

            if self.behavior.get("slow"):
                time.sleep(0.6)
            if self.behavior.get("status", 200) != 200:
                self.send_response(self.behavior["status"])
                self.end_headers()
                return
            if mode == "groq":
                payload = {"choices": [{"message": {"role": "assistant", "content": "Groq answer [1]."}}]}
            else:
                payload = {"message": {"role": "assistant", "content": "Ollama answer [1]."}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args) -> None:  # keep test output clean
            pass

    return Stub


@pytest.fixture(scope="module")
def providers():
    """One Groq-shaped and one Ollama-shaped stub, started once per module.

    Returns (classes, urls); per-test state is reset by the autouse fixture.
    """
    GroqStub = make_stub("groq")
    OllamaStub = make_stub("ollama")
    groq_server = HTTPServer(("127.0.0.1", 0), GroqStub)
    ollama_server = HTTPServer(("127.0.0.1", 0), OllamaStub)
    threads = [
        threading.Thread(target=groq_server.serve_forever, daemon=True),
        threading.Thread(target=ollama_server.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    yield (
        GroqStub,
        OllamaStub,
        f"http://127.0.0.1:{groq_server.server_port}",
        f"http://127.0.0.1:{ollama_server.server_port}",
    )
    groq_server.shutdown()
    ollama_server.shutdown()
    groq_server.server_close()
    ollama_server.server_close()


@pytest.fixture(autouse=True)
def _reset_providers(providers):
    """Reset per-test stub behavior and recorded requests."""
    groq_stub, ollama_stub, *_ = providers
    groq_stub.behavior = {"status": 200}
    groq_stub.requests = []
    ollama_stub.behavior = {"status": 200}
    ollama_stub.requests = []
    yield


def build_chain(groq_url: str, ollama_url: str, groq_timeout: float = 10.0) -> FallbackLLMClient:
    groq = GroqLLMClient(
        api_key="test-key", model="groq-model", base_url=groq_url, timeout=groq_timeout
    )
    ollama = OllamaLLMClient(
        base_url=ollama_url, model="ollama-model", temperature=0.1, timeout=10.0
    )
    return FallbackLLMClient([groq, ollama])


def generate(client, query: str = "What is the rule?"):
    return client.generate(system_prompt="SYS", user_query=query, context="[1] evidence")


def test_fallback_uses_groq_when_it_succeeds(providers) -> None:
    groq_stub, ollama_stub, groq_url, ollama_url = providers
    groq_stub.behavior = {"status": 200}

    response = generate(build_chain(groq_url, ollama_url))

    assert response.provider == "groq"
    assert response.model == "groq-model"
    assert response.text == "Groq answer [1]."
    assert len(ollama_stub.requests) == 0  # no failover happened


def test_fallback_groq_service_unavailable_uses_ollama(providers) -> None:
    groq_stub, ollama_stub, groq_url, ollama_url = providers
    groq_stub.behavior = {"status": 503}

    response = generate(build_chain(groq_url, ollama_url))

    assert response.provider == "ollama_fallback"
    assert response.model == "ollama-model"
    assert "Ollama answer" in response.text
    assert len(ollama_stub.requests) == 1


def test_fallback_groq_timeout_uses_ollama(providers) -> None:
    groq_stub, _, groq_url, ollama_url = providers
    groq_stub.behavior = {"slow": True}

    response = generate(build_chain(groq_url, ollama_url, groq_timeout=0.3))

    assert response.provider == "ollama_fallback"
    assert "Ollama answer" in response.text


def test_fallback_groq_connection_failure_uses_ollama(providers) -> None:
    _, ollama_stub, _, ollama_url = providers
    dead_groq_url = "http://127.0.0.1:1"

    response = generate(build_chain(dead_groq_url, ollama_url))

    assert response.provider == "ollama_fallback"
    assert len(ollama_stub.requests) == 1


def test_fallback_all_providers_fail_raises_clean_error(providers) -> None:
    groq_stub, ollama_stub, groq_url, ollama_url = providers
    groq_stub.behavior = {"status": 503}
    ollama_stub.behavior = {"status": 500}

    with pytest.raises(LLMUnavailableError) as exc_info:
        generate(build_chain(groq_url, ollama_url))
    assert "All LLM providers failed" in str(exc_info.value)
    message = str(exc_info.value)
    assert "groq" in message and "ollama" in message


def test_fallback_configuration_error_does_not_fallback(providers) -> None:
    groq_stub, ollama_stub, groq_url, ollama_url = providers
    groq_stub.behavior = {"status": 401}  # rejected API key -> configuration error

    with pytest.raises(LLMConfigurationError, match="authentication"):
        generate(build_chain(groq_url, ollama_url))
    assert len(ollama_stub.requests) == 0  # the fallback must not mask bad config


def test_fallback_single_provider_is_not_relabeled(providers) -> None:
    _, ollama_stub, _, ollama_url = providers
    ollama = OllamaLLMClient(
        base_url=ollama_url, model="m", temperature=0.1, timeout=10.0
    )

    response = generate(FallbackLLMClient([ollama]))

    assert response.provider == "ollama"  # no relabel when there is no failover


def test_fallback_requires_at_least_one_provider() -> None:
    with pytest.raises(LLMConfigurationError, match="at least one provider"):
        FallbackLLMClient([])