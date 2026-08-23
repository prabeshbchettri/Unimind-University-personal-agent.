"""LLM errors.

Raised by LLM client implementations so callers can distinguish "the model
could not be reached" from other failures.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMUnavailableError(LLMError):
    """The LLM backend/model could not be reached."""
