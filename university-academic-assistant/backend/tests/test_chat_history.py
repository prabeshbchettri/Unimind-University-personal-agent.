"""Tests for Phase 10: PostgreSQL chat history (sessions + messages).

Covers the acceptance flow — create session, send message, store assistant
response, retrieve session, continue conversation, delete session — at the
repository, service and API levels, plus bounded history injection into the
generation prompt. All tests run on the in-memory backend; the PostgreSQL
path shares the same repository code.
"""

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from tests import rag_factory as factory

from app.rag.context import ContextBuilder
from app.repositories.chat_history import ChatHistoryRepository
from app.services.chat import ChatService, SessionNotFoundError, derive_title

NO_EVIDENCE_ANSWER = "I could not find sufficient information in the available sources."


def _run(coro):
    return asyncio.run(coro)


class _CapturingLLM:
    """LLM stand-in that records every prompt it receives."""

    name = "capture"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        if "CONTEXT:" in prompt and "END OF CONTEXT" in prompt and "Normalization" in prompt:
            return "Normalization removes redundancy in tables."
        return NO_EVIDENCE_ANSWER


def _indexed_service(history=None, **kwargs) -> ChatService:
    service = factory.make_indexing_service()
    factory.index(
        service,
        factory.sample_structured(
            "Unit 1: Introduction\nDBMS is a database management system. Databases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.\n"
            "Unit 3: Transactions\nACID properties guarantee reliable processing."
        ),
    )
    retriever = factory.make_retriever(service.embedder, service.repository, top_k=3)
    llm = kwargs.pop("llm", _CapturingLLM())
    return ChatService(
        retriever=retriever,
        context_builder=factory.make_context_builder(),
        llm=llm,
        top_k=3,
        history=history,
        history_max_chars=kwargs.pop("history_max_chars", 2000),
        history_max_messages=kwargs.pop("history_max_messages", 12),
    )


# --- repository ------------------------------------------------------------


def test_repository_create_and_get_session() -> None:
    repo = factory.make_chat_history()
    session = _run(repo.create_session("My first conversation"))

    assert session.session_id
    assert session.title == "My first conversation"
    assert session.created_at is not None
    assert session.updated_at is not None

    fetched = _run(repo.get_session(session.session_id))
    assert fetched is not None
    assert fetched.session_id == session.session_id
    assert _run(repo.get_session("does-not-exist")) is None


def test_repository_add_and_list_messages() -> None:
    repo = factory.make_chat_history()
    session = _run(repo.create_session("chat"))

    user = _run(repo.add_message(session.session_id, "user", "What is normalization?"))
    assistant = _run(repo.add_message(session.session_id, "assistant", "Normalization removes redundancy."))

    assert user.role == "user"
    assert assistant.role == "assistant"
    assert user.content == "What is normalization?"

    messages = _run(repo.get_messages(session.session_id))
    assert [m.role for m in messages] == ["user", "assistant"]
    assert _run(repo.message_count(session.session_id)) == 2


def test_repository_rejects_unknown_role_and_session() -> None:
    repo = factory.make_chat_history()
    session = _run(repo.create_session("chat"))

    try:
        _run(repo.add_message(session.session_id, "admin", "hello"))
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        _run(repo.add_message("nope", "user", "hello"))
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_repository_list_sessions_with_counts_most_recent_first() -> None:
    repo = factory.make_chat_history()
    first = _run(repo.create_session("first"))
    second = _run(repo.create_session("second"))
    _run(repo.add_message(first.session_id, "user", "hi"))
    _run(repo.add_message(first.session_id, "assistant", "hello"))

    sessions, total = _run(repo.list_sessions())
    assert total == 2
    # first got a message after second was created -> first is most recent;
    # second has 0 messages.
    assert sessions[0][0].session_id == first.session_id
    assert sessions[0][1] == 2
    assert sessions[1][0].session_id == second.session_id
    assert sessions[1][1] == 0


def test_repository_delete_session_cascades_messages() -> None:
    repo = factory.make_chat_history()
    session = _run(repo.create_session("doomed"))
    _run(repo.add_message(session.session_id, "user", "hi"))
    _run(repo.add_message(session.session_id, "assistant", "hello"))

    assert _run(repo.delete_session(session.session_id)) is True
    assert _run(repo.get_session(session.session_id)) is None
    assert _run(repo.get_messages(session.session_id)) == []
    assert _run(repo.delete_session(session.session_id)) is False


def test_repository_ping_and_backend_resolution() -> None:
    repo = factory.make_chat_history()
    assert _run(repo.ping()) is True

    auto_memory = ChatHistoryRepository(backend="auto", database_url="sqlite:///:memory:", create_tables=False)
    assert auto_memory.backend == "auto"
    auto_postgres = ChatHistoryRepository(
        backend="auto",
        database_url="postgresql+psycopg://u:p@localhost:5432/db",
        create_tables=False,
    )
    assert auto_postgres.engine.url.get_backend_name() == "postgresql"

    postgres = ChatHistoryRepository(
        backend="postgres",
        database_url="postgresql+psycopg://u:p@localhost:5432/db",
        create_tables=False,
    )
    assert postgres.engine.url.get_backend_name() == "postgresql"

    try:
        ChatHistoryRepository(backend="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- service ---------------------------------------------------------------


def test_service_creates_session_and_persists_messages() -> None:
    history = factory.make_chat_history()
    llm = _CapturingLLM()
    service = _indexed_service(history=history, llm=llm)

    result = _run(service.answer("Explain normalization."))

    assert result.session_id
    assert result.answer == "Normalization removes redundancy in tables."
    assert result.sources

    session = _run(history.get_session(result.session_id))
    assert session is not None
    assert session.title == "Explain normalization"
    messages = _run(history.get_messages(result.session_id))
    assert [(m.role, m.content) for m in messages] == [
        ("user", "Explain normalization."),
        ("assistant", "Normalization removes redundancy in tables."),
    ]


def test_service_unknown_session_raises() -> None:
    history = factory.make_chat_history()
    service = _indexed_service(history=history)

    try:
        _run(service.answer("hello", session_id="missing"))
        assert False, "expected SessionNotFoundError"
    except SessionNotFoundError:
        pass


def test_service_continues_session_with_bounded_history_in_prompt() -> None:
    history = factory.make_chat_history()
    llm = _CapturingLLM()
    service = _indexed_service(history=history, llm=llm, history_max_messages=4)

    first = _run(service.answer("Explain normalization."))
    assert "CONVERSATION:" not in llm.prompts[0]  # first turn has no history

    second = _run(service.answer("And transactions?", session_id=first.session_id))
    prompt = llm.prompts[1]
    assert "CONVERSATION:" in prompt
    assert "user: Explain normalization." in prompt
    assert "assistant: Normalization removes redundancy in tables." in prompt
    assert second.session_id == first.session_id


def test_service_history_is_bounded_by_message_count() -> None:
    history = factory.make_chat_history()
    llm = _CapturingLLM()
    service = _indexed_service(history=history, llm=llm, history_max_messages=2)

    session_id = _run(service.answer("Explain normalization.")).session_id
    _run(service.answer("Explain transactions.", session_id=session_id))
    _run(service.answer("Explain indexing.", session_id=session_id))
    _run(service.answer("Explain DBMS.", session_id=session_id))

    prompt = llm.prompts[3]
    # 6 prior messages, window 2: only the two most recent turns are shown.
    assert "user: Explain indexing." in prompt
    assert "assistant: Normalization removes redundancy in tables." in prompt
    assert "user: Explain transactions." not in prompt
    assert "user: Explain normalization." not in prompt


def test_service_history_is_bounded_by_characters() -> None:
    history = factory.make_chat_history()
    llm = _CapturingLLM()
    service = _indexed_service(history=history, llm=llm, history_max_chars=50)

    session_id = _run(service.answer("Explain normalization.")).session_id
    _run(service.answer("Explain transactions.", session_id=session_id))
    _run(service.answer("Explain indexing.", session_id=session_id))

    prompt = llm.prompts[2]
    # The 50-char budget keeps only the most recent assistant turn (which is
    # the "indexing" answer); older turns are dropped entirely.
    assert "assistant: Normalization removes redundancy in tables." in prompt
    assert "user: Explain indexing." not in prompt
    assert "user: Explain transactions." not in prompt
    assert "user: Explain normalization." not in prompt


def test_derive_title_cleans_and_truncates() -> None:
    assert derive_title("  What is   normalization?  ") == "What is normalization?"
    assert derive_title("   ") == "New conversation"
    assert derive_title("abcde", max_chars=3) == "..."
    long = " ".join(["word"] * 30)
    assert len(derive_title(long, max_chars=20)) <= 20


# --- context builder -------------------------------------------------------


def test_context_builder_includes_history_section() -> None:
    builder = ContextBuilder()
    system, prompt = builder.build("next question", [], history="user: prior\nassistant: answer")

    assert "CONVERSATION:" in prompt
    assert "user: prior" in prompt
    assert "assistant: answer" in prompt
    assert "User Query: next question" in prompt
    assert system


def test_context_builder_without_history_has_no_conversation_section() -> None:
    builder = ContextBuilder()
    _, prompt = builder.build("question", [])
    assert "CONVERSATION:" not in prompt


# --- API -------------------------------------------------------------------


def _index_syllabus(client: TestClient, make_text_pdf, tmp_path: Path) -> None:
    path = make_text_pdf(
        tmp_path / "history_syllabus.pdf",
        [
            "Database Management System Syllabus\n"
            "Unit 1: Introduction\nDatabases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.",
        ],
    )
    with open(path, "rb") as handle:
        response = client.post(
            "/documents/index",
            files={"file": ("history_syllabus.pdf", handle, "application/pdf")},
        )
    assert response.status_code == 200


def test_api_session_lifecycle(client: TestClient, make_text_pdf, tmp_path: Path) -> None:
    _index_syllabus(client, make_text_pdf, tmp_path)

    # Fresh app -> no sessions yet.
    empty = client.get("/chat/sessions")
    assert empty.status_code == 200
    assert empty.json()["total"] == 0

    # 1. Create session + 2. send message + 3. store assistant response.
    first = client.post("/chat", json={"message": "Explain normalization."})
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    assert session_id

    # 4. Continue the conversation on the same session.
    second = client.post("/chat", json={"message": "And transactions?", "session_id": session_id})
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id

    # 5. Retrieve the session with its full message history.
    detail = client.get(f"/chat/sessions/{session_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["session_id"] == session_id
    assert body["title"] == "Explain normalization"
    assert body["message_count"] == 4
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user", "assistant"]
    assert body["messages"][0]["content"] == "Explain normalization."
    assert body["messages"][1]["content"] == first.json()["answer"]

    # Session list reflects the persisted session.
    listing = client.get("/chat/sessions")
    assert listing.json()["total"] == 1
    item = listing.json()["items"][0]
    assert item["session_id"] == session_id
    assert item["message_count"] == 4

    # 6. Delete the session.
    deleted = client.delete(f"/chat/sessions/{session_id}")
    assert deleted.status_code == 204
    assert client.get(f"/chat/sessions/{session_id}").status_code == 404
    assert client.get("/chat/sessions").json()["total"] == 0


def test_api_session_errors(client: TestClient) -> None:
    assert client.get("/chat/sessions/missing").status_code == 404
    assert client.delete("/chat/sessions/missing").status_code == 404
    response = client.post("/chat", json={"message": "hello", "session_id": "missing"})
    assert response.status_code == 404


def test_api_history_endpoint_lists_sessions(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello there"})
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    history = client.get("/history")
    assert history.status_code == 200
    body = history.json()
    assert body["total"] == 1
    assert body["items"][0]["session_id"] == session_id
    assert body["items"][0]["title"] == "hello there"