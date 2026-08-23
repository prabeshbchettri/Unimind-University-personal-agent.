"""Sparse (keyword) retrieval index — Okapi BM25 (Phase 6).

A lightweight in-memory BM25 index over indexed chunks. It complements the
dense vector index for queries where exact terms matter (subject codes,
regulation numbers, dates, academic years, question numbers, names, numbers
and specific phrases).

Implemented in-house so Phase 6 needs no extra dependencies and runs offline
(in-memory, like the embedded Qdrant client). ``tokenize`` keeps alphanumeric
tokens (codes like ``CSIT 325``, ``regulation 12``, years like ``2080``) and
drops a small stopword set.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.chunking.models import DocumentChunk

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "its", "of", "on", "or", "the", "that", "this", "to", "was",
    "what", "with",
}


def tokenize(text: str) -> list[str]:
    """Return lower-cased word tokens for ``text`` (stopwords removed)."""
    words = _TOKEN_RE.findall(text.lower())
    return [word for word in words if word not in _STOPWORDS and len(word) >= 2]


@dataclass
class SparseHit:
    """A BM25 match with the source payload for reconstructing results."""

    chunk_id: str
    score: float
    payload: dict = field(default_factory=dict)
    collection: str | None = None


class BM25Index:
    """Okapi BM25 over the indexed chunks (in-memory)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b in [0, 1]")
        self._k1 = k1
        self._b = b
        # chunk_id -> (term_frequencies, document_length, payload, collection)
        self._docs: dict[str, tuple[Counter, int, dict, str | None]] = {}
        # term -> number of documents containing it
        self._document_frequency: Counter = Counter()
        self._avgdl = 0.0

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._docs)

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    def add(self, chunk: DocumentChunk, collection: str | None = None) -> None:
        """Index a single chunk under ``collection``."""
        terms = tokenize(chunk.text)
        term_frequencies = Counter(terms)
        self._docs[chunk.chunk_id] = (term_frequencies, len(terms), chunk.payload(), collection)
        for term in term_frequencies:
            self._document_frequency[term] += 1
        self._avgdl = (
            sum(doc_length for _, doc_length, _, _ in self._docs.values()) / len(self._docs)
        )

    def add_chunks(self, chunks: Sequence[DocumentChunk], collection: str | None = None) -> None:
        """Index a sequence of chunks, all under the same ``collection``."""
        for chunk in chunks:
            self.add(chunk, collection)

    def clear(self) -> None:
        """Drop all indexed chunks (used when re-indexing the corpus)."""
        self._docs.clear()
        self._document_frequency.clear()
        self._avgdl = 0.0

    def search(self, query: str, top_k: int = 10) -> list[SparseHit]:
        """Return the best-matching chunks for ``query`` (BM25 score).

        Chunks with no term overlap score zero and are omitted.
        """
        query_terms = set(tokenize(query))
        if not query_terms or not self._docs:
            return []

        total_docs = len(self._docs)
        hits: list[SparseHit] = []
        for chunk_id, (term_frequencies, doc_length, payload, collection) in self._docs.items():
            score = 0.0
            for term in query_terms:
                term_frequency = term_frequencies.get(term)
                if not term_frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = term_frequency + self._k1 * (
                    1 - self._b + self._b * (doc_length / self._avgdl)
                )
                score += inverse_document_frequency * (
                    term_frequency * (self._k1 + 1)
                ) / denominator
            if score > 0:
                hits.append(
                    SparseHit(chunk_id=chunk_id, score=score, payload=payload, collection=collection)
                )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]