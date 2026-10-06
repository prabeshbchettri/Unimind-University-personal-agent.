"""Tests for the embedding interface and provider selection.

No test calls a real embedding model; the Ollama HTTP provider is tested
against a local HTTP server stub so request/response handling is exercised
without a network dependency.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.config import Settings
from app.embeddings import (
    EmbeddingError,
    OllamaEmbeddingClient,
    build_embedding_client,
)
from tests.conftest import FakeEmbedding


def test_fake_embedding_is_deterministic() -> None:
    client = FakeEmbedding(dim=32)
    first = client.embed_query("minimum attendance requirement")
    second = client.embed_query("minimum attendance requirement")
    assert first == second
    assert len(first) == 32
    assert all(-1.0 <= value <= 1.0 for value in first)


def test_fake_embedding_empty_batch() -> None:
    assert FakeEmbedding().embed_texts([]) == []


def test_build_embedding_client_selects_local() -> None:
    settings = Settings(embedding_provider="local", _env_file=None)
    client = build_embedding_client(settings)
    assert client.name == "local"


def test_build_embedding_client_rejects_unknown_provider() -> None:
    settings = Settings(embedding_provider="openai", _env_file=None)
    with pytest.raises(EmbeddingError, match="Unknown EMBEDDING_PROVIDER"):
        build_embedding_client(settings)


class _OllamaStub(BaseHTTPRequestHandler):
    """HTTP stub speaking the Ollama /api/embed contract."""

    behavior: dict = {}

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        if self.behavior.get("fail"):
            self.send_response(500)
            self.end_headers()
            return
        vectors = [[0.1, 0.2, 0.3] for _ in body["input"]]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"embeddings": vectors}).encode())

    def log_message(self, *args) -> None:  # keep test output clean
        pass


@pytest.fixture
def ollama_stub_url() -> str:
    _OllamaStub.behavior = {"fail": False}
    server = HTTPServer(("127.0.0.1", 0), _OllamaStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_ollama_client_embeds_batch(ollama_stub_url) -> None:
    client = OllamaEmbeddingClient(base_url=ollama_stub_url, model="nomic-embed-text")
    vectors = client.embed_texts(["attendance", "exams"])
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert client.dim == 3


def test_ollama_client_wraps_http_errors(ollama_stub_url) -> None:
    _OllamaStub.behavior = {"fail": True}
    client = OllamaEmbeddingClient(base_url=ollama_stub_url, model="nomic-embed-text")
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        client.embed_texts(["attendance"])


def test_ollama_client_fails_cleanly_when_unreachable() -> None:
    client = OllamaEmbeddingClient(base_url="http://127.0.0.1:1", model="m", timeout=2)
    with pytest.raises(EmbeddingError, match="Ollama embedding request failed"):
        client.embed_texts(["attendance"])
