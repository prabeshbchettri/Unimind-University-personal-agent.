"""Web search endpoint schemas (Phase 11)."""

from pydantic import BaseModel, Field


class WebResultSchema(BaseModel):
    """A single web search result with full source attribution."""

    title: str = Field(..., description="Result title.")
    url: str = Field(..., description="Result URL.")
    snippet: str = Field(..., description="Result snippet/content excerpt.")
    retrieved_at: str = Field(..., description="ISO-8601 UTC time the result was retrieved.")
    score: float = Field(1.0, description="Position-based relevance score.")


class WebSearchRequest(BaseModel):
    """Request body for ``POST /web/search``."""

    query: str = Field(..., min_length=1, description="Search query.")
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Number of results to return (default: server setting).",
    )


class WebSearchResponse(BaseModel):
    """Response body for ``POST /web/search``."""

    query: str = Field(..., description="The original query.")
    backend: str = Field(..., description="The provider that served the results (stub | duckduckgo).")
    results: list[WebResultSchema] = Field(default_factory=list, description="Web search results.")