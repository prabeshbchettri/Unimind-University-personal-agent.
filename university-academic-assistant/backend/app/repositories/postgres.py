"""PostgreSQL repository (placeholder).

Boundary for relational persistence (chat sessions, messages, and any
relational document metadata). SQLAlchemy wiring arrives in later phases.
"""

from __future__ import annotations

from app.repositories.base import Repository


class PostgresRepository(Repository):
    """Placeholder for relational persistence against PostgreSQL."""

    async def ping(self) -> bool:
        return False
