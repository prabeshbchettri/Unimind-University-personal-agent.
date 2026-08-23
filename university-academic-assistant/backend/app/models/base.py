"""SQLAlchemy ORM models.

Phase 1 defines only the shared declarative base. Concrete models
(e.g. ``chat_sessions`` and ``messages``) are added in later phases.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


__all__ = ["Base"]