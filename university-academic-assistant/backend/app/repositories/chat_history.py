"""Chat history repository (Phase 10).

SQLAlchemy-backed persistence for chat sessions and messages. Two backends
follow the codebase convention of working offline in tests:

- ``memory``  — in-process SQLite (no server, per-engine database)
- ``postgres`` — PostgreSQL via ``DATABASE_URL`` (psycopg 3)

The engine is synchronous; repository methods run their work in a worker
thread (``asyncio.to_thread``) so the event loop is never blocked. All
operations are conversation persistence — nothing here models user
personality, preferences or long-term memory.
"""

from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.chat import ChatMessage, ChatSession, MESSAGE_ROLES, _utcnow
from app.repositories.base import Repository

_MEMORY_SENTINELS = ("sqlite://", "sqlite:///:memory:")


class ChatHistoryRepository(Repository):
    """Persists chat sessions and their messages."""

    def __init__(
        self,
        database_url: str = "sqlite://",
        backend: str = "memory",
        create_tables: bool = True,
    ) -> None:
        super().__init__()
        self.backend = backend
        resolved_backend = self._resolve_backend(backend, database_url)
        if resolved_backend == "postgres":
            self.engine = create_engine(database_url)
        else:
            # One shared connection keeps the in-memory database visible to
            # every worker thread.
            self.engine = create_engine(
                "sqlite://",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        if create_tables:
            Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def _resolve_backend(backend: str, database_url: str) -> str:
        choice = backend.lower()
        if choice == "auto":
            return "memory" if any(database_url.startswith(s) for s in _MEMORY_SENTINELS) else "postgres"
        if choice in {"memory", "postgres"}:
            return choice
        raise ValueError(f"Unknown database backend: {backend!r}")

    def _run(self, operation) -> object:
        with self._session_factory() as session:
            return operation(session)

    # --- sessions -----------------------------------------------------------

    async def create_session(self, title: str) -> ChatSession:
        """Create and persist a new session, returning it."""
        session_id = uuid.uuid4().hex

        def _create(s: Session) -> ChatSession:
            session = ChatSession(session_id=session_id, title=title[:120])
            s.add(session)
            s.commit()
            return session

        return await asyncio.to_thread(self._run, _create)

    async def get_session(self, session_id: str) -> ChatSession | None:
        """Return the session with ``session_id`` or None."""

        def _get(s: Session) -> ChatSession | None:
            return s.get(ChatSession, session_id)

        return await asyncio.to_thread(self._run, _get)

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[tuple[ChatSession, int]], int]:
        """Return ``((session, message_count), total)`` most recently updated first."""

        def _list(s: Session) -> tuple[list[tuple[ChatSession, int]], int]:
            total = s.scalar(select(func.count()).select_from(ChatSession)) or 0
            rows = s.execute(
                select(ChatSession, func.count(ChatMessage.message_id))
                .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.session_id)
                .group_by(ChatSession.session_id)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            return [(session, int(count)) for session, count in rows], int(total)

        return await asyncio.to_thread(self._run, _list)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages; return whether it existed."""

        def _delete(s: Session) -> bool:
            session = s.get(ChatSession, session_id)
            if session is None:
                return False
            s.delete(session)  # messages cascade
            s.commit()
            return True

        return await asyncio.to_thread(self._run, _delete)

    # --- messages -----------------------------------------------------------

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """Return the session's messages, oldest first.

        ``limit`` keeps the most recent ``limit`` messages (the window used
        for conversation context).
        """
        query = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.message_id.desc())
        )
        if limit is not None:
            query = query.limit(limit)

        def _get(s: Session) -> list[ChatMessage]:
            return list(reversed(list(s.scalars(query).all())))

        return await asyncio.to_thread(self._run, _get)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> ChatMessage:
        """Append a message to a session (raising KeyError when unknown)."""
        if role not in MESSAGE_ROLES:
            raise ValueError(f"Unknown message role: {role!r}")
        message_id = uuid.uuid4().hex

        def _add(s: Session) -> ChatMessage:
            session = s.get(ChatSession, session_id)
            if session is None:
                raise KeyError(f"Unknown session: {session_id}")
            message = ChatMessage(
                message_id=message_id,
                session_id=session_id,
                role=role,
                content=content,
            )
            session.updated_at = _utcnow()
            s.add(message)
            s.commit()
            return message

        return await asyncio.to_thread(self._run, _add)

    async def message_count(self, session_id: str) -> int:
        """Return how many messages a session contains."""

        def _count(s: Session) -> int:
            return int(
                s.scalar(
                    select(func.count())
                    .select_from(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                )
                or 0
            )

        return await asyncio.to_thread(self._run, _count)

    # --- lifecycle ----------------------------------------------------------

    async def ping(self) -> bool:
        """Return whether the backing database is reachable."""

        def _ping() -> bool:
            try:
                with self.engine.connect() as connection:
                    connection.execute(select(1))
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_ping)

    async def close(self) -> None:
        """Dispose of the engine and its pooled connections."""
        self.engine.dispose()


__all__ = ["ChatHistoryRepository"]