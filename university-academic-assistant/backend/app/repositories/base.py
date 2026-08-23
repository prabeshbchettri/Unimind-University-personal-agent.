"""Base repository interface.

Repositories abstract access to backing stores (PostgreSQL, Qdrant, Neo4j).
Phase 1 defines the boundary only; concrete store-specific repositories are
added in later phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.logging import get_logger


class Repository(ABC):
    """Base class for data-access repositories."""

    def __init__(self) -> None:
        self.logger = get_logger(f"app.repositories.{self.name}")

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def ping(self) -> bool:
        """Return whether the backing store is reachable."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release any resources held by the repository."""
        return None
