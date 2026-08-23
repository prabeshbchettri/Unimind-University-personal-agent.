"""Retrieval result models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchHit:
    """A raw vector-store hit (payload + score)."""

    chunk_id: str
    score: float
    payload: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A user-facing search result."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    collection: str | None = None
    # Retrieval method that produced this result: "vector" (legacy dense),
    # "dense", "sparse", or "hybrid" (fused + reranked).
    method: str = "vector"
