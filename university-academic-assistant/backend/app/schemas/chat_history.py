"""Chat history schemas (Phase 10)."""

from datetime import datetime

from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    """A stored chat message."""

    message_id: str = Field(..., description="Unique message id.")
    session_id: str = Field(..., description="Session the message belongs to.")
    role: str = Field(..., description="user | assistant | system")
    content: str = Field(..., description="Message text.")
    created_at: datetime = Field(..., description="When the message was stored.")


class ChatSessionSummary(BaseModel):
    """A chat session without its messages."""

    session_id: str = Field(..., description="Unique session id.")
    title: str = Field(..., description="Derived from the first user message.")
    created_at: datetime = Field(..., description="When the session was created.")
    updated_at: datetime = Field(..., description="When the session last changed.")
    message_count: int = Field(0, description="Number of stored messages.")


class ChatSessionDetail(ChatSessionSummary):
    """A chat session including its messages (oldest first)."""

    messages: list[ChatMessageSchema] = Field(default_factory=list, description="Stored messages.")


class SessionListResponse(BaseModel):
    """List of chat sessions, most recently updated first."""

    items: list[ChatSessionSummary]
    total: int


__all__ = [
    "ChatMessageSchema",
    "ChatSessionDetail",
    "ChatSessionSummary",
    "SessionListResponse",
]