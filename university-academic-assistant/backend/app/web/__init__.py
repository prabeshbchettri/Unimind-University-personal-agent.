"""Web search package (Phase 11).

Web search is a controlled retrieval backend for questions that require
current or external information. Providers implement the
:class:`~app.web.providers.WebSearchProvider` interface; the router selects
the ``WEB`` strategy and the retriever wraps provider results into the same
``SearchResult`` shape the rest of the pipeline uses.

The stub provider is the default (hermetic, offline); ``duckduckgo`` performs
live searches over the public DuckDuckGo HTML endpoint with a hard timeout.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.web.errors import WebSearchError, WebSearchTimeoutError
from app.web.models import WebResult
from app.web.providers import (
    DuckDuckGoWebSearchProvider,
    StubWebSearchProvider,
    WebSearchProvider,
)

WEB_COLLECTION = "web"


def build_web_search_provider(settings: Settings) -> WebSearchProvider:
    """Build the configured web search provider."""
    backend = settings.web_search_backend.lower()
    if backend == "duckduckgo":
        return DuckDuckGoWebSearchProvider(
            timeout_seconds=settings.web_search_timeout,
            max_results=settings.web_max_sources,
        )
    if backend == "stub":
        return StubWebSearchProvider()
    raise ValueError(
        f"Unknown web search backend {settings.web_search_backend!r}; "
        "expected 'stub' or 'duckduckgo'."
    )


__all__ = [
    "WEB_COLLECTION",
    "WebResult",
    "WebSearchError",
    "WebSearchTimeoutError",
    "WebSearchProvider",
    "StubWebSearchProvider",
    "DuckDuckGoWebSearchProvider",
    "build_web_search_provider",
]
