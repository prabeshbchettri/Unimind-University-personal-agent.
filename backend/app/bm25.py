"""Lexical retrieval with Okapi BM25.

The lexical half of the HYBRID strategy. Semantic embeddings blur exact tokens
(course codes, precise policy wording); BM25 scores them directly, so a query
that names an identifier retrieves the exact document regardless of paraphrase.

Pipeline position: built once per process from the vector store payloads
(:func:`build_bm25_index`), then searched per request. Pure Python, no external
services.

The class is intentionally small so the scoring formula is readable and easy to
explain: for every unique query term, idf weights how rare the term is across
the corpus and the frequency term behaves like a soft saturating ``tf``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.ingestion import Chunk
from app.vectorstore import QdrantVectorStore

#: Tokens are lowercase alphanumeric runs, so "CS201", "cs201" and "cs 201"
#: all normalize to the same search token.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class LexicalHit:
    """One BM25 hit with its source metadata and score."""

    chunk_id: str
    score: float
    text: str
    document: str
    page: int
    chunk_index: int
    #: Full source path, carried through so document-type intent ranking can
    #: classify lexical-only hits by their folder too.
    source_path: str = ""


class BM25Index:
    """Deterministic Okapi BM25 index over a static set of chunks.

    A chunk is added once at build time (from the ingested corpus). ``search``
    returns the best ``top_k`` chunks for a query ranked by BM25 score, each
    with its source metadata so hybrid fusion can report citations.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunk_ids: list[str] = []
        self._chunk_tokens: list[dict[str, int]] = []
        self._doc_lens: list[int] = []
        self._df: dict[str, int] = {}
        self._meta: dict[str, Chunk] = {}
        self._avgdl = 0.0

    def add(self, chunk: Chunk) -> None:
        """Index one chunk; re-adding a chunk id replaces its previous entry."""
        if chunk.id in self._meta:
            self._remove(chunk.id)
        tokens = tokenize(chunk.text)
        if not tokens:
            return
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        self._chunk_ids.append(chunk.id)
        self._chunk_tokens.append(frequencies)
        self._doc_lens.append(len(tokens))
        for term in frequencies:
            self._df[term] = self._df.get(term, 0) + 1
        self._meta[chunk.id] = chunk
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens)

    def _remove(self, chunk_id: str) -> None:
        """Drop a chunk's term statistics and metadata (used by ``add``)."""
        index = self._chunk_ids.index(chunk_id)
        frequencies = self._chunk_tokens[index]
        for term in frequencies:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        del self._chunk_ids[index]
        del self._chunk_tokens[index]
        del self._doc_lens[index]
        del self._meta[chunk_id]
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunk_ids)

    def search(self, query: str, top_k: int = 5) -> list[LexicalHit]:
        """Return the ``top_k`` chunks matching ``query``, best first."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if self.size == 0:
            return []
        # Unique query terms; repeating a word in the query adds no score.
        terms = {token for token in tokenize(query)}
        if not terms:
            return []

        doc_count = self.size
        scored: list[LexicalHit] = []
        for index, frequencies in enumerate(self._chunk_tokens):
            doc_len = self._doc_lens[index]
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._df.get(term, 0)
                idf = math.log(1 + (doc_count - document_frequency + 0.5) / (document_frequency + 0.5))
                normalized_len = doc_len / self._avgdl if self._avgdl else 1.0
                denominator = frequency + self._k1 * (1 - self._b + self._b * normalized_len)
                score += idf * (frequency * (self._k1 + 1)) / denominator
            if score <= 0:
                continue
            chunk = self._meta[self._chunk_ids[index]]
            scored.append(
                LexicalHit(
                    chunk_id=chunk.id,
                    score=score,
                    text=chunk.text,
                    document=chunk.document_name,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    source_path=chunk.source_path,
                )
            )

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]


def build_bm25_index(store: QdrantVectorStore) -> BM25Index:
    """Build a BM25 index over every chunk currently stored in ``store``.

    Qdrant payloads are the single source of truth for the corpus (text plus
    source metadata), so the lexical index always matches the vector store.
    """
    index = BM25Index()
    for chunk in store.scroll_chunks():
        index.add(chunk)
    return index