"""Base service interface.

All domain services inherit from :class:`Service` to establish a stable
boundary. Later phases can extend or replace implementations without changing
the rest of the application, as long as the public interface is preserved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.logging import get_logger


class Service(ABC):
    """Base class for application services.

    Provides a shared logger and a ``name`` used for structured logging.
    """

    def __init__(self) -> None:
        self.logger = get_logger(f"app.services.{self.name}")

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def health(self) -> dict:
        """Report whether this service is healthy.

        Phase 1 returns a minimal status. Concrete services override this in
        later phases to reflect their backing-store connectivity.
        """
        raise NotImplementedError
