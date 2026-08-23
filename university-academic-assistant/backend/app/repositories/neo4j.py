"""Neo4j repository (placeholder).

Boundary for the academic knowledge graph. Neo4j driver wiring arrives in
later phases.
"""

from __future__ import annotations

from app.repositories.base import Repository


class Neo4jRepository(Repository):
    """Placeholder for graph queries against Neo4j."""

    async def ping(self) -> bool:
        return False
