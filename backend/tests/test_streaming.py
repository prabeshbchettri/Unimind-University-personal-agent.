"""Tests for streamed generation: LLM providers, RAG event flow, SSE endpoint."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from app.llm import LLMStreamHandle, LLMUnavailableError
from app.rag import GREETING_ANSWER, INSUFFICIENT_EVIDENCE_ANSWER, SYSTEM_PROMPT, stream_answer
from tests.conftest import FakeRetriever, make_retrieved
from tests.test_rag import build_rag


class StreamingFakeLLM:
    """FakeLLM equivalent that streams canned deltas. Records prompts."""

    name = "fake"
    model = "fake-stream-model"

    def __init__(self, deltas: list[str] | None = None) -> None:
        self.deltas = deltas or ["Answer ", "with ", "citations [1]."]
        self.calls: list[dict] = []
        self.fail_stream = False

    def generate(self, *, system_prompt: str, user_query: str, context: str, history=None):
        raise AssertionError("stream path must not call generate()")

    def stream_generate(
        self, *, system_prompt: str, user_query: str, context: str, history=None
    ):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_query": user_query,
                "context": context,
                "history": list(history or []),
            }
        )
        if self.fail_stream:
            raise LLMUnavailableError("fake stream down")

        def tokens():
            yield from self.deltas

        return LLMStreamHandle(tokens(), provider=self.name, model=self.model)


@pytest.fixture
def rag_settings() -> Settings:
    return Settings(
        qdrant_url="http://fake",
        qdrant_collection="test_docs",
        embedding_dim=64,
        retrieval_top_k=4,
        min_relevance_score=0.0,
        max_context_chars=4000,
        _env_file=None,
    )


# ---------------------------------------------------------------------------
# RAG-level event flow
# ---------------------------------------------------------------------------


def test_stream_answer_grounded_flow_events_and_final_shape(rag_settings) -> None:
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": "75 percent rule."})
    llm = StreamingFakeLLM()

    events = list(
        stream_answer(
            "What is the attendance rule?",
            settings=rag_settings,
            retriever=retriever,
            llm=llm,
        )
    )

    assert [event["type"] for event in events] == ["meta", "delta", "delta", "delta", "done"]
    meta = events[0]
    assert meta["provider"] == "fake" and meta["model"] == "fake-stream-model"

    done = events[-1]
    assert done["answer"] == "Answer with citations [1]."
    assert done["enough_evidence"] is True
    assert done["provider"] == "fake"
    assert done["sources"] == [{"document": "attendance_policy.pdf", "page": 1}]
    assert set(done) >= {
        "answer", "sources", "enough_evidence", "retrieval", "provider", "model",
        "strategy", "strategy_reason",
    }

    # The model received the same shared grounding prompt as non-streamed answers.
    assert llm.calls[0]["system_prompt"] == SYSTEM_PROMPT


def test_stream_answer_greeting_is_one_done_event(rag_settings) -> None:
    retriever = FakeRetriever([make_retrieved("evidence", score=0.9)])
    llm = StreamingFakeLLM()

    events = list(
        stream_answer("Hello", settings=rag_settings, retriever=retriever, llm=llm)
    )

    assert len(events) == 1
    assert events[0]["type"] == "done"
    assert events[0]["answer"] == GREETING_ANSWER
    assert events[0]["strategy"] == "greeting"
    assert llm.calls == [] and retriever.queries == []


def test_stream_answer_without_evidence_declines_without_llm(rag_settings) -> None:
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": "75 percent rule."})
    llm = StreamingFakeLLM()
    settings = rag_settings.model_copy(update={"min_relevance_score": 0.99})

    events = list(
        stream_answer(
            "Who won the football World Cup?",
            settings=settings,
            retriever=retriever,
            llm=llm,
        )
    )

    assert len(events) == 1
    assert events[0]["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert events[0]["enough_evidence"] is False
    assert events[0]["sources"] == []
    assert llm.calls == []


def test_stream_answer_deltas_are_citation_validated(rag_settings) -> None:
    """Deltas containing invented citation markers are cleaned before display."""
    retriever, _ = build_rag(rag_settings, {"attendance_policy.pdf": "75 percent rule."})
    llm = StreamingFakeLLM(deltas=["Real claim [1]", " invented [9]", " end."])

    events = list(
        stream_answer("attendance?", settings=rag_settings, retriever=retriever, llm=llm)
    )
    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "[9]" not in deltas
    assert events[-1]["answer"] == "Real claim [1] invented  end."


# ---------------------------------------------------------------------------
# API endpoint (SSE)
# ---------------------------------------------------------------------------


def test_api_stream_returns_sse_event_sequence(monkeypatch) -> None:
    retriever = FakeRetriever([make_retrieved("evidence", score=0.8)])
    llm = StreamingFakeLLM()
    monkeypatch.setattr(
        main, "get_pipeline", lambda: {"retriever": retriever, "llm": llm}
    )
    client = TestClient(main.app)

    with client.stream(
        "POST", "/api/chat/stream", json={"message": "What is the rule?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk.decode("utf-8") for chunk in response.iter_raw())

    events = [
        json.loads(line[len("data: "):])
        for line in body.strip().split("\n\n")
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["meta", "delta", "delta", "delta", "done"]
    assert events[-1]["answer"] == "Answer with citations [1]."


def test_api_stream_maps_llm_failure_to_error_event(monkeypatch) -> None:
    retriever = FakeRetriever([make_retrieved("evidence", score=0.8)])
    llm = StreamingFakeLLM()
    llm.fail_stream = True
    monkeypatch.setattr(
        main, "get_pipeline", lambda: {"retriever": retriever, "llm": llm}
    )
    client = TestClient(main.app)

    with client.stream("POST", "/api/chat/stream", json={"message": "anything"}) as response:
        assert response.status_code == 200  # errors travel inside the SSE stream
        body = "".join(chunk.decode("utf-8") for chunk in response.iter_raw())
    events = [
        json.loads(line[len("data: "):])
        for line in body.strip().split("\n\n")
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["meta", "error"]
    assert events[-1]["message"]


def test_api_stream_rejects_empty_message(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_pipeline", lambda: {"retriever": FakeRetriever([]), "llm": StreamingFakeLLM()})
    client = TestClient(main.app)
    response = client.post("/api/chat/stream", json={"message": "   "})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Provider streaming (HTTP stubs)
# ---------------------------------------------------------------------------


class _OllamaStreamStub(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for piece in ["Hel", "lo [1", "]."]:
            self.wfile.write(
                (json.dumps({"message": {"role": "assistant", "content": piece}, "done": False}) + "\n").encode()
            )
        self.wfile.write((json.dumps({"done": True}) + "\n").encode())

    def log_message(self, *args) -> None:
        pass


class _GroqStreamStub(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ["Ground", "ed [1", "]."]:
            chunk = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args) -> None:
        pass


def _serve(stub_class) -> str:
    server = HTTPServer(("127.0.0.1", 0), stub_class)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}"


def test_ollama_stream_yields_deltas():
    from app.llm import OllamaLLMClient

    client = OllamaLLMClient(base_url=_serve(_OllamaStreamStub), model="m", temperature=0.1, timeout=10)
    handle = client.stream_generate(system_prompt="s", user_query="q", context="c")
    tokens = list(handle)
    assert "".join(tokens) == "Hello [1]."
    assert handle.provider == "ollama" and handle.model == "m"
    assert handle.text == "Hello [1]."


def test_groq_stream_yields_sse_deltas():
    from app.llm import GroqLLMClient

    client = GroqLLMClient(api_key="test-key", model="m", base_url=_serve(_GroqStreamStub))
    handle = client.stream_generate(system_prompt="s", user_query="q", context="c")
    assert "".join(handle) == "Grounded [1]."
    assert handle.provider == "groq"


def test_fallback_stream_falls_over_to_next_provider():
    from app.llm import FallbackLLMClient

    class DownProvider:
        name = "down"
        model = "x"

        def generate(self, **kwargs):
            raise LLMUnavailableError("down")

        def stream_generate(self, **kwargs):
            raise LLMUnavailableError("down")

    up = StreamingFakeLLM()
    chain = FallbackLLMClient([DownProvider(), up])
    handle = chain.stream_generate(system_prompt="s", user_query="q", context="c")
    tokens = list(handle)
    assert "".join(tokens) == "Answer with citations [1]."
    assert handle.provider == "fake_fallback"
