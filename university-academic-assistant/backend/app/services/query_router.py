"""Query router service interface (placeholder).

In later phases this decides between normal RAG, hybrid RAG, graph RAG and
web search based on the analysis of an incoming query.
"""

from __future__ import annotations

from app.services.base import Service


class QueryRouter(Service):
    """Analyses a query and selects a retrieval strategy."""

    async def health(self) -> dict:
        return {"name": self.name, "status": "ok"}

    async def route(self, query: str) -> str:
        """Return the name of the chosen retrieval strategy.

        Not implemented in Phase 1.
        """
        raise NotImplementedError("Query routing is implemented in a later phase.")
