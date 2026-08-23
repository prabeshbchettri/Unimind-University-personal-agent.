"""Web search errors (Phase 11).

Web content is external and unreliable: providers fail, time out or return
nothing. These errors let the router degrade gracefully — a failed web search
falls back to internal retrieval instead of failing the whole answer.
"""

from __future__ import annotations


class WebSearchError(Exception):
    """Base class for web search failures."""


class WebSearchTimeoutError(WebSearchError):
    """A web search provider exceeded its allowed time budget."""