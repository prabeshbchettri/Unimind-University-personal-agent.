"""Process-local caches (Phase 13, Part 3).

Two bounded caches, each with an explicit invalidation story so cached data
can never go stale:

- :class:`EmbeddingCache` — memoizes embeddings keyed by ``(model, text)``.
  Embeddings are deterministic for a fixed model, so entries stay valid for
  the process lifetime; the cache is bounded (LRU eviction) and never grows
  without limit. No TTL is needed: a model is fixed at startup.
- :class:`RetrievalCache` — memoizes retrieval results keyed by
  ``(retriever, query, top_k)``. Every entry stores the *index generation*
  counter it was computed under; :meth:`RetrievalCache.invalidate` (called by
  the indexing service after any document mutation) bumps the generation and
  drops all entries, so a cached result is only served while the index is
  exactly the same as when it was computed. A TTL additionally bounds the
  lifetime of generation-stable entries (e.g. after a document is edited in
  place without a re-index).

Both caches are intentionally process-local: the system is a single server
instance, no shared state is required, and wholesale invalidation on
mutation keeps the invalidation strategy simple and correct.
"""

from __future__ import annotations

import threading
import time

# LRU-style eviction: dicts preserve insertion order in CPython; re-inserting
# a key on set() moves it to the end, and eviction pops the oldest entry.


class EmbeddingCache:
    """Memoizes ``(model, text) -> vector`` with LRU eviction."""

    def __init__(self, max_entries: int = 5000) -> None:
        self.max_entries = max(1, max_entries)
        self._store: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()

    def get(self, model: str, text: str) -> list[float] | None:
        """Return the cached vector for ``(model, text)`` or None."""
        return self._store.get((model, text))

    def set(self, model: str, text: str, vector: list[float]) -> None:
        """Store ``vector`` for ``(model, text)``, evicting oldest if full."""
        with self._lock:
            key = (model, text)
            if key in self._store:
                self._store.pop(key)
            elif len(self._store) >= self.max_entries:
                self._store.pop(next(iter(self._store)))
            self._store[key] = vector

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class RetrievalCache:
    """Memoizes retrieval results with index-generation + TTL invalidation."""

    def __init__(self, max_entries: int = 256, ttl_seconds: int = 300) -> None:
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = ttl_seconds
        self._generation = 0
        self._store: dict[tuple[str, str, int], tuple[int, float, list[dict]]] = {}
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        """Drop every entry and advance the index generation counter."""
        with self._lock:
            self._generation += 1
            self._store.clear()

    @property
    def generation(self) -> int:
        return self._generation

    def get(
        self,
        name: str,
        query: str,
        top_k: int,
        now: float | None = None,
    ) -> list[dict] | None:
        """Return the serialized results when current and fresh, else None."""
        key = (name, query, top_k)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            generation, stored_at, results = entry
            current = now if now is not None else time.monotonic()
            if generation != self._generation or current - stored_at > self.ttl_seconds:
                return None
            return results

    def set(
        self,
        name: str,
        query: str,
        top_k: int,
        results: list[dict],
        now: float | None = None,
    ) -> None:
        """Store serialized results under the current index generation."""
        key = (name, query, top_k)
        with self._lock:
            if key in self._store:
                self._store.pop(key)
            elif len(self._store) >= self.max_entries:
                self._store.pop(next(iter(self._store)))
            self._store[key] = (
                self._generation,
                now if now is not None else time.monotonic(),
                results,
            )

    def __len__(self) -> int:
        return len(self._store)
