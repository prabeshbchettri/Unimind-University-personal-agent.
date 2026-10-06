"""API tests for POST /api/chat (pipeline fakes injected via dependency override)."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.llm import LLMError
from app.retriever import RetrievalError
from app.vectorstore import VectorStoreError
from tests.conftest import FakeLLM, FakeRetriever, make_retrieved


@pytest.fixture
def client():
    return TestClient(main.app)


def override_pipeline(monkeypatch, retriever, llm) -> None:
    monkeypatch.setattr(main, "get_pipeline", lambda: {"retriever": retriever, "llm": llm})


def test_health_still_works(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_verifies_vector_store(monkeypatch) -> None:
    class PingableStore:
        def ping(self) -> None:
            return None

    monkeypatch.setattr(main, "get_pipeline", lambda: {"store": PingableStore()})
    client = TestClient(main.app)
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["vector_store"] == "ok"


def test_ready_maps_store_failure_to_503(monkeypatch) -> None:
    class BrokenStore:
        def ping(self) -> None:
            raise VectorStoreError("embedded engine unavailable")

    monkeypatch.setattr(main, "get_pipeline", lambda: {"store": BrokenStore()})
    client = TestClient(main.app)
    response = client.get("/ready")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "unready"
    assert "vector_store" in detail["checks"]


def test_chat_request_log_has_metadata_but_no_question_text(monkeypatch, client, caplog) -> None:
    override_pipeline(
        monkeypatch,
        FakeRetriever([make_retrieved("evidence", score=0.8)]),
        FakeLLM(text="answer [1]"),
    )
    with caplog.at_level(logging.INFO, logger="app.main"):
        response = client.post("/api/chat", json={"message": "XYZ-PRIVATE-QUESTION attendance?"})
    assert response.status_code == 200
    logs = [record.getMessage() for record in caplog.records]
    assert any(
        "strategy=normal" in line and "provider=fake" in line and "latency_ms=" in line
        for line in logs
    )
    assert all("XYZ-PRIVATE-QUESTION" not in line for line in logs)


def test_chat_request_log_reports_provider_dash_on_failure(monkeypatch, client, caplog) -> None:
    class BrokenRetriever:
        def retrieve(self, query: str):
            raise RetrievalError("store down")

    override_pipeline(monkeypatch, BrokenRetriever(), FakeLLM())
    with caplog.at_level(logging.INFO, logger="app.main"):
        client.post("/api/chat", json={"message": "anything"})
    assert any("provider=-" in record.getMessage() for record in caplog.records)


def test_chat_returns_grounded_answer_with_sources(monkeypatch, client) -> None:
    retriever = FakeRetriever(
        [make_retrieved("Attendance requires 75 percent.", document="attendance_policy.pdf", page=1, score=0.8)]
    )
    llm = FakeLLM(text="The minimum is 75 percent [1].")
    override_pipeline(monkeypatch, retriever, llm)

    response = client.post("/api/chat", json={"message": "What is the attendance rule?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The minimum is 75 percent [1]."
    assert body["sources"] == [{"document": "attendance_policy.pdf", "page": 1}]
    assert body["enough_evidence"] is True
    assert body["provider"] == "fake"
    assert body["strategy"] == "normal"  # no router injected -> Phase 3 path
    assert body["strategy_reason"]
    assert body["retrieval"]["chunks_retrieved"] == 1
    assert retriever.queries == ["What is the attendance rule?"]


def test_chat_insufficient_evidence(monkeypatch, client) -> None:
    from app.rag import INSUFFICIENT_EVIDENCE_ANSWER

    retriever = FakeRetriever([make_retrieved("weak match", score=0.2)])
    llm = FakeLLM()
    override_pipeline(monkeypatch, retriever, llm)

    response = client.post("/api/chat", json={"message": "Who won the world cup?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert body["sources"] == []
    assert body["enough_evidence"] is False
    assert body["provider"] == "none"
    assert body["strategy"] == "normal"
    assert llm.calls == []


def test_chat_rejects_empty_message(monkeypatch, client) -> None:
    override_pipeline(monkeypatch, FakeRetriever([]), FakeLLM())
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_rejects_whitespace_only_message(monkeypatch, client) -> None:
    override_pipeline(monkeypatch, FakeRetriever([]), FakeLLM())
    response = client.post("/api/chat", json={"message": "   \n\t  "})
    assert response.status_code == 422


def test_chat_rejects_overlong_message(monkeypatch, client) -> None:
    override_pipeline(monkeypatch, FakeRetriever([]), FakeLLM())
    response = client.post("/api/chat", json={"message": "x" * 3000})
    assert response.status_code == 422


def test_chat_maps_retrieval_failure_to_503(monkeypatch, client) -> None:
    class BrokenRetriever:
        def retrieve(self, query: str):
            raise RetrievalError("vector store down")

    override_pipeline(monkeypatch, BrokenRetriever(), FakeLLM())
    response = client.post("/api/chat", json={"message": "anything"})
    assert response.status_code == 503
    assert response.json()["detail"] == "retrieval pipeline unavailable"


def test_chat_maps_llm_failure_to_503(monkeypatch, client) -> None:
    class BrokenLLM:
        name = "broken"

        def generate(self, **kwargs):
            raise LLMError("provider down")

    retriever = FakeRetriever([make_retrieved("evidence", score=0.9)])
    override_pipeline(monkeypatch, retriever, BrokenLLM())
    response = client.post("/api/chat", json={"message": "attendance?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "generation pipeline unavailable"


def test_chat_validates_request_body(client) -> None:
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_chat_reports_hybrid_strategy_when_router_configured(monkeypatch, client) -> None:
    from app.routing import QueryRouter

    retriever = FakeRetriever([])
    hybrid = FakeRetriever(
        [make_retrieved("CS201 is a four credit core course", document="cs201_syllabus.pdf", score=0.7)]
    )
    llm = FakeLLM(text="CS201 is four credits [1].")
    monkeypatch.setattr(
        main,
        "get_pipeline",
        lambda: {
            "retriever": retriever,
            "hybrid_retriever": hybrid,
            "llm": llm,
            "router": QueryRouter(),
        },
    )

    response = client.post("/api/chat", json={"message": "What is CS201?"})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "hybrid"
    assert "course code" in body["strategy_reason"]
    assert body["enough_evidence"] is True
    assert body["sources"] == [{"document": "cs201_syllabus.pdf", "page": 1}]
    assert hybrid.queries == ["What is CS201?"]
    assert retriever.queries == []
