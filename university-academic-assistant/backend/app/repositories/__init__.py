"""Data-access repositories.

Concrete backing stores (Qdrant, Neo4j, PostgreSQL) are imported from their
modules directly; this package init stays import-free so importing any one
repository cannot create circular imports with the retrieval/graph layers.
"""

from .base import Repository

__all__ = ["Repository"]
