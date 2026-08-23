"""Chat history endpoint schemas (Phase 10).

The placeholder summaries are replaced by the Phase 10 session/message
schemas; the response envelope keeps the original shape.
"""

from pydantic import BaseModel

from app.schemas.chat_history import ChatSessionSummary


class HistoryResponse(ChatSessionSummary):
    """A single chat session summary (Phase 10)."""


class HistoryListResponse(BaseModel):
    """List of chat sessions (Phase 10)."""

    items: list[ChatSessionSummary]
    total: int
