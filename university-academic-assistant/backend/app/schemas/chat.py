"""Chat endpoint schemas (Phase 5).

``POST /chat`` accepts a user message and returns a grounded answer plus the
retrieved sources that were used.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""

    message: str = Field(..., min_length=1, description="User message text.")
    session_id: str | None = Field(None, description="Optional existing session id.")
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Number of retrieved context chunks to use (default: server setting).",
    )


class SourceSchema(BaseModel):
    """A retrieved source chunk that grounded the answer."""

    text: str = Field(..., description="Chunk text (or web snippet).")
    score: float = Field(..., description="Retrieval score against the query.")
    metadata: dict = Field(default_factory=dict, description="Source metadata (title, page, subject, topic, ...).")
    collection: str = Field("", description="Collection the chunk came from (university_docs | past_questions | library_books | knowledge_graph | web).")
    retrieval_method: str = Field("", description="How the chunk was retrieved: dense | sparse | hybrid | graph | web.")
    kind: str = Field("university", description="Source kind: 'university' | 'web'.")
    url: str = Field("", description="Source URL (web results only).")
    retrieved_at: str = Field("", description="ISO-8601 retrieval time (web results only).")


class ChatResponse(BaseModel):
    """Response body for ``POST /chat``."""

    answer: str = Field(..., description="Grounded assistant answer.")
    sources: list[SourceSchema] = Field(default_factory=list, description="Retrieved sources used.")
    session_id: str | None = Field(None, description="Chat session id; None when history persistence is disabled.")
    strategy: str = Field("NORMAL", description="Retrieval strategy the adaptive router selected.")