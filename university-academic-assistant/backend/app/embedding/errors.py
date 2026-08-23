"""Embedding errors.

Raised by embedder implementations so callers can distinguish "the model
could not be reached" from "the model produced unusable output".
"""

from __future__ import annotations


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingUnavailableError(EmbeddingError):
    """The embedding backend/model could not be reached or loaded."""
