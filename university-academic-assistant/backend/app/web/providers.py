"""Web search providers (Phase 11).

Two backends behind one async interface:

- :class:`StubWebSearchProvider` — deterministic, offline canned results
  (tests / development without network). Returns nothing for queries it does
  not recognize, modelling an empty result set rather than an error.
- :class:`DuckDuckGoWebSearchProvider` — live search over the public
  DuckDuckGo HTML endpoint using only the standard library (no API key).
  Guarded by a hard timeout; failures raise :class:`WebSearchError`.

Safety: providers only return title/url/snippet/retrieved_at. Web content is
never trusted as authoritative — the router keeps university documents
preferred and the grounding prompt requires URL attribution.
"""

from __future__ import annotations

import asyncio
import html as html_module
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Protocol

from app.web.errors import WebSearchError, WebSearchTimeoutError
from app.web.models import WebResult

_USER_AGENT = "Mozilla/5.0 (academic-assistant/1.0; offline-friendly)"

# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class WebSearchProvider(Protocol):
    """Async web search interface implemented by every backend."""

    name: str

    async def search(self, query: str, top_k: int = 4) -> list[WebResult]:
        """Return up to ``top_k`` results for ``query`` (may be empty)."""
        ...


# ---------------------------------------------------------------------------
# Stub backend (offline, deterministic)
# ---------------------------------------------------------------------------

# (keywords, title, url, snippet) — the first entry whose keywords all appear
# in the query wins. Keywords are matched as whole words, case-insensitively.
_CANNED_RESULTS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (
        ("python", "version"),
        "Python 3.13 Released",
        "https://www.python.org/downloads/",
        "Python 3.13 is the latest stable release, with improved error "
        "messages, a new interactive interpreter and a JIT compiler.",
    ),
    (
        ("python", "pricing"),
        "Python 3.13 Released",
        "https://www.python.org/downloads/",
        "Python 3.13 is the latest stable release, available free of charge.",
    ),
    (
        ("openai", "pricing"),
        "OpenAI API Pricing",
        "https://openai.com/api/pricing",
        "OpenAI API pricing varies by model; newer models are billed per "
        "input and output token.",
    ),
    (
        ("ai", "news"),
        "Today's AI News Digest",
        "https://example.com/ai-news",
        "Major AI providers announced new model releases and pricing updates "
        "today.",
    ),
    (
        ("attendance",),
        "Attendance Policy Update 2026",
        "https://example.com/attendance-policy",
        "Universities updated attendance policies for the 2026 academic year; "
        "verify with your institution.",
    ),
    (
        ("weather",),
        "Weather Forecast",
        "https://weather.example.com",
        "Weather forecasts are available for the requested location.",
    ),
)


class StubWebSearchProvider:
    """Deterministic offline provider with canned results.

    The retrieval timestamp is fixed at construction so tests are stable.
    """

    name = "stub"

    def __init__(self) -> None:
        self._retrieved_at = datetime.now(timezone.utc).isoformat()

    async def search(self, query: str, top_k: int = 4) -> list[WebResult]:
        tokens = re.findall(r"[a-z0-9']+", query.lower())
        for keywords, title, url, snippet in _CANNED_RESULTS:
            if all(keyword in tokens for keyword in keywords):
                return [
                    WebResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        retrieved_at=self._retrieved_at,
                        score=round(max(1.0 - index * 0.1, 0.5), 2),
                    )
                    for index in range(min(top_k, 1))
                ]
        return []


# ---------------------------------------------------------------------------
# Live backend (DuckDuckGo HTML, no API key)
# ---------------------------------------------------------------------------


class DuckDuckGoWebSearchProvider:
    """Live web search over DuckDuckGo's HTML endpoint.

    Runs the network request in a worker thread with ``asyncio.wait_for``.
    Timeouts raise :class:`WebSearchTimeoutError`; transport/parse problems
    raise :class:`WebSearchError`. Only the result title, URL, snippet and
    retrieval time are returned.
    """

    name = "duckduckgo"

    _URL_TEMPLATE = "https://html.duckduckgo.com/html/?q={query}"
    _RESULT_LINK = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    _RESULT_SNIPPET = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)

    def __init__(self, timeout_seconds: float = 8.0, max_results: int = 4) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    async def search(self, query: str, top_k: int = 4) -> list[WebResult]:
        if not query or not query.strip():
            return []
        try:
            page = await self._fetch(query)
        except WebSearchError:
            raise
        except asyncio.TimeoutError as exc:
            raise WebSearchTimeoutError(f"Web search timed out after {self.timeout_seconds}s.") from exc
        except Exception as exc:  # network/DNS/HTTP errors
            raise WebSearchError(f"Web search failed: {exc}") from exc
        return self._parse_html(page, top_k or self.max_results)

    async def _fetch(self, query: str) -> str:
        """Fetch the search results HTML under the configured timeout."""
        url = self._URL_TEMPLATE.format(query=urllib.parse.quote_plus(query))
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        return await asyncio.wait_for(
            asyncio.to_thread(self._open, request),
            timeout=self.timeout_seconds,
        )

    def _open(self, request: urllib.request.Request) -> str:
        """Blocking HTML fetch (runs in a worker thread)."""
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")

    def _parse_html(self, page: str, top_k: int) -> list[WebResult]:
        """Extract result links and snippets from the DuckDuckGo HTML page."""
        links = self._RESULT_LINK.findall(page)
        snippets = [
            html_module.unescape(re.sub(r"<[^>]+>", "", part)).strip()
            for part in self._RESULT_SNIPPET.findall(page)
        ]
        retrieved_at = datetime.now(timezone.utc).isoformat()
        results: list[WebResult] = []
        for index, (href, title) in enumerate(links[:top_k]):
            url = self._extract_url(href)
            snippet = snippets[index] if index < len(snippets) else ""
            results.append(
                WebResult(
                    title=html_module.unescape(re.sub(r"<[^>]+>", "", title)).strip(),
                    url=url,
                    snippet=snippet,
                    retrieved_at=retrieved_at,
                    score=round(max(1.0 - index * 0.1, 0.5), 2),
                )
            )
        return results

    @staticmethod
    def _extract_url(href: str) -> str:
        """Resolve DuckDuckGo redirect links to the real target URL."""
        parsed = urllib.parse.urlparse(html_module.unescape(href))
        if parsed.netloc and "duckduckgo.com" not in parsed.netloc:
            return href
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if target and target[0]:
            return target[0]
        return href