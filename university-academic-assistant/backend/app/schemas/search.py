"""Search and indexing endpoint schemas (Phase 4)."""

from pydantic import BaseModel


class IndexResultSchema(BaseModel):
    """Summary of a document indexing operation."""

    document_id: str
    document_type: str
    chunk_count: int
    collections: dict[str, int]
    graph: dict | None = None


class SearchResultSchema(BaseModel):
    """A single vector-search result."""

    text: str
    score: float
    metadata: dict


class SearchResponseSchema(BaseModel):
    """Vector search response."""

    query: str
    collection: str | None
    top_k: int
    results: list[SearchResultSchema]