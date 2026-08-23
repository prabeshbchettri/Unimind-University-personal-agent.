"""Chat service (Phase 5 + Phase 10 + Phase 11).

Coordinates the complete vertical slice:

    user query + conversation history -> retriever -> Qdrant
        -> context builder -> Llama -> answer

Phase 10 adds conversation persistence: ``answer`` optionally runs inside a
chat session (create when ``session_id`` is None, continue otherwise). The
user message and the assistant answer are stored, and the recent bounded
history is injected into the prompt so the LLM can follow up on the ongoing
conversation. History is conversation persistence only — no long-term
personal memory, personality or preferences.

Phase 11 adds web search: the adaptive router may return web results
(``collection == "web"``) alongside or instead of university documents. The
service separates them so the context builder renders web sources in their
own ``WEB RESULTS`` block with URL attribution, keeping them clearly
distinguished from university sources.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.llm import LLMClient
from app.rag.context import ContextBuilder
from app.retrieval import SearchResult
from app.retrieval.cached import CachedRetriever
from app.retrieval.normal_retriever import NormalRetriever
from app.services.base import Service
from app.web import WEB_COLLECTION

if TYPE_CHECKING:
    from app.repositories.chat_history import ChatHistoryRepository

logger = get_logger("app.services.chat")


class SessionNotFoundError(Exception):
    """Raised when a chat message references an unknown session."""


@dataclass
class ChatResult:
    """Answer plus the sources it was grounded on."""

    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    session_id: str | None = None


def derive_title(message: str, max_chars: int = 60) -> str:
    """A session title from the first user message, cleaned and truncated."""
    title = re.sub(r"\s+", " ", message).strip(" .:-–—")
    if not title:
        return "New conversation"
    if len(title) <= max_chars:
        return title
    return title[: max_chars - 3].rsplit(" ", 1)[0] + "..."


class ChatService(Service):
    """End-to-end grounded question answering with optional session history."""

    def __init__(
        self,
        retriever: NormalRetriever,
        context_builder: ContextBuilder,
        llm: LLMClient,
        top_k: int = 5,
        history: ChatHistoryRepository | None = None,
        history_max_chars: int = 2000,
        history_max_messages: int = 12,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm = llm
        self.top_k = top_k
        self.history = history
        self.history_max_chars = history_max_chars
        self.history_max_messages = history_max_messages

    async def health(self) -> dict:
        return {
            "name": self.name,
            "status": "ok",
            "retriever": type(self.retriever).__name__,
            "context_builder": type(self.context_builder).__name__,
            "llm": self.llm.name,
            "history": type(self.history).__name__ if self.history else None,
        }

    async def answer(
        self,
        message: str,
        top_k: int | None = None,
        session_id: str | None = None,
    ) -> ChatResult:
        """Answer ``message`` grounded on retrieved university context.

        When ``history`` is configured: a None ``session_id`` creates a new
        session; a known ``session_id`` continues it (unknown ids raise
        :class:`SessionNotFoundError`). Both the user message and the answer
        are persisted, and the bounded recent history is added to the prompt.
        """
        limit = top_k or self.top_k
        history_text = ""
        session_state = "none"

        if self.history is None:
            target_session_id = session_id
        else:
            target_session_id, history_text = await self._prepare_session(message, session_id)
            session_state = "new" if session_id is None else "existing"
            await self.history.add_message(target_session_id, "user", message)

        retrieval_started = time.perf_counter()
        chunks = await self.retriever.retrieve(message, top_k=limit)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        internal = [chunk for chunk in chunks if chunk.collection != WEB_COLLECTION]
        web_results = [chunk for chunk in chunks if chunk.collection == WEB_COLLECTION]
        system, prompt = self.context_builder.build(
            message,
            internal,
            history=history_text,
            web_results=web_results,
        )
        llm_started = time.perf_counter()
        answer = await asyncio.to_thread(self.llm.complete, system, prompt)
        llm_ms = (time.perf_counter() - llm_started) * 1000

        if self.history is not None:
            await self.history.add_message(target_session_id, "assistant", answer)

        # Metrics only: strategy, counts and latencies. User messages, chat
        # history, document contents and answer text are never logged.
        logger.info(
            "Chat answer generated",
            extra={
                "extra_fields": {
                    "strategy": type(getattr(self.retriever, "retriever", self.retriever)).__name__,
                    "cache_enabled": int(isinstance(self.retriever, CachedRetriever)),
                    "chunks": len(chunks),
                    "web_chunks": len(web_results),
                    "session": session_state,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "llm_ms": round(llm_ms, 2),
                    "total_ms": round(retrieval_ms + llm_ms, 2),
                }
            },
        )

        return ChatResult(answer=answer, sources=chunks[:limit], session_id=target_session_id)

    async def _prepare_session(self, message: str, session_id: str | None) -> tuple[str, str]:
        """Resolve the target session and render its bounded history."""
        if session_id is None:
            session = await self.history.create_session(title=derive_title(message))
            return session.session_id, ""
        session = await self.history.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return session.session_id, await self._render_history(session_id)

    async def _render_history(self, session_id: str) -> str:
        """Render the most recent messages that fit the character budget."""
        messages = await self.history.get_messages(session_id, limit=self.history_max_messages)
        selected: list[str] = []
        used = 0
        for msg in reversed(messages):
            line = f"{msg.role}: {msg.content}"
            if selected and used + len(line) > self.history_max_chars:
                break
            selected.append(line)
            used += len(line) + 1
        return "\n".join(reversed(selected))