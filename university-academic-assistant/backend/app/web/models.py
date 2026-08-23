"""Web search result model (Phase 11)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WebResult:
    """A single web search result with full source attribution.

    ``retrieved_at`` is an ISO-8601 UTC timestamp recording when the result
    was retrieved, so answers carry the recency of the source.
    """

    title: str
    url: str
    snippet: str
    retrieved_at: str
    score: float = 1.0
