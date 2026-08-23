"""Retrieval service interface (placeholder).

Defines the boundary for all retrieval strategies (normal, hybrid, graph).
Strategy-specific implementations arrive in Phases 5-7.
"""

from __future__ import annotations

from app.services.base import Service


class RetrievalService(Service):
    """Retrieves relevant context for a query."""

    async def health(self) -> dict:
        return {"name": self.name, "status": "ok"}

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return ranked context chunks for ``query``.

        Not implemented in Phase 1.
        """
        raise NotImplementedError("Retrieval is implemented in a later phase.")
