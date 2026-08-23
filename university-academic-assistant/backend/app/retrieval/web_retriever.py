"""Web retrieval (Phase 11).

Wraps a :class:`~app.web.providers.WebSearchProvider` behind the same
``retrieve(query, top_k)`` interface as the other retrievers, producing
``SearchResult`` objects that the context builder and the API can consume.

Every provider call runs under ``asyncio.wait_for``: a provider that exceeds
the budget raises :class:`WebSearchTimeoutError`, which the adaptive router
treats as a graceful failure (fallback to internal retrieval).
"""

from __future__ import annotations

import asyncio
import logging

from app.retrieval.models import SearchResult
from app.web import WEB_COLLECTION
from app.web.errors import WebSearchError, WebSearchTimeoutError
from app.web.providers import WebSearchProvider

logger = logging.getLogger("app.retrieval.web_retriever")


class WebRetrievalError(Exception):
    """Raised when web retrieval fails or times out."""


class WebRetriever:
    """Retrieves external/current information from a web search provider."""

    name = "WebRetriever"

    def __init__(
        self,
        provider: WebSearchProvider,
        top_k: int = 4,
        timeout: float = 8.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.provider = provider
        self.top_k = top_k
        self.timeout = timeout

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search the web for ``query``, returning SearchResult-compatible rows.

        Raises :class:`WebRetrievalError` when the provider fails or times
        out; an empty result set is a legitimate outcome (no useful matches).
        """
        limit = top_k or self.top_k
        try:
            results = await asyncio.wait_for(
                self.provider.search(query, top_k=limit),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as exc:
            raise WebRetrievalError(f"Web search timed out after {self.timeout}s.") from exc
        except WebSearchTimeoutError as exc:
            raise WebRetrievalError(str(exc)) from exc
        except WebSearchError as exc:
            raise WebRetrievalError(str(exc)) from exc

        return [self._to_result(result) for result in results[:limit]]

    @staticmethod
    def _to_result(result) -> SearchResult:
        """Map a provider result onto the shared SearchResult shape.

        ``text`` carries the snippet, and title/url/retrieved_at live in
        metadata so the context builder can format citations and the API can
        expose them directly.
        """
        return SearchResult(
            text=result.snippet,
            score=result.score,
            metadata={
                "title": result.title,
                "url": result.url,
                "retrieved_at": result.retrieved_at,
            },
            collection=WEB_COLLECTION,
            method="web",
        )
