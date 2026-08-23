"""API tests for ``POST /chat`` (Phase 5)."""

from pathlib import Path

from fastapi.testclient import TestClient

NO_EVIDENCE_ANSWER = "I could not find sufficient information in the available sources."


def _index_syllabus(client: TestClient, make_text_pdf, tmp_path: Path) -> None:
    path = make_text_pdf(
        tmp_path / "chat_syllabus.pdf",
        [
            "Database Management System Syllabus\n"
            "DBMS is a database management system.\n"
            "Unit 1: Introduction\nDatabases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.",
        ],
    )
    with open(path, "rb") as handle:
        response = client.post(
            "/documents/index",
            files={"file": ("chat_syllabus.pdf", handle, "application/pdf")},
        )
    assert response.status_code == 200


def test_chat_returns_answer_and_sources(client: TestClient, make_text_pdf, tmp_path: Path) -> None:
    _index_syllabus(client, make_text_pdf, tmp_path)

    response = client.post("/chat", json={"message": "Explain normalization.", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str)
    assert body["answer"]
    assert isinstance(body["sources"], list)
    assert body["sources"]
    top = body["sources"][0]
    assert {"text", "score", "metadata"} <= top.keys()
    assert top["retrieval_method"] in {"dense", "sparse", "hybrid"}
    assert top["metadata"]["document_type"] == "syllabus"


def test_chat_question_not_in_database_returns_no_sources(
    client: TestClient, make_text_pdf, tmp_path: Path
) -> None:
    _index_syllabus(client, make_text_pdf, tmp_path)

    response = client.post("/chat", json={"message": "What is the capital of France?"})

    assert response.status_code == 200
    body = response.json()
    # No supporting evidence -> no sources; the stub LLM returns the explicit
    # insufficiency answer instead of a confident fabricated reply.
    assert body["sources"] == []
    assert body["answer"] == NO_EVIDENCE_ANSWER


def test_chat_empty_message_rejected(client: TestClient) -> None:
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_top_k_out_of_range_rejected(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello", "top_k": 0})
    assert response.status_code == 422


def test_chat_unknown_session_returns_404(client: TestClient) -> None:
    # Phase 10: a session_id must reference a persisted session.
    response = client.post("/chat", json={"message": "hello", "session_id": "sess-123"})
    assert response.status_code == 404
    assert "sess-123" in response.json()["detail"]


def test_chat_creates_session_and_returns_session_id(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    sessions = client.get("/chat/sessions").json()
    assert sessions["total"] == 1
    assert sessions["items"][0]["session_id"] == body["session_id"]


def test_chat_embedding_unavailable_returns_503(client: TestClient, monkeypatch) -> None:
    from app.embedding.errors import EmbeddingUnavailableError

    def _boom(*args, **kwargs):
        raise EmbeddingUnavailableError("embedder unavailable")

    monkeypatch.setattr(client.app.state.embedder, "embed_one", _boom)
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 503


def test_chat_llm_unavailable_returns_503(client: TestClient, monkeypatch) -> None:
    from app.llm.errors import LLMUnavailableError

    def _boom(*args, **kwargs):
        raise LLMUnavailableError("llm unavailable")

    monkeypatch.setattr(client.app.state.chat_service, "answer", _boom)
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 503