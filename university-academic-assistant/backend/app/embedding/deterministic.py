"""Deterministic feature-hash embedder.

A lightweight, dependency-free embedding backend used for tests and offline
development. It maps each token to a fixed, hashed position in a dense vector
so that texts sharing tokens produce similar (cosine) vectors. It is NOT a
semantic model; production deployments should use the Ollama/BGE backend.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from app.embedding.base import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicEmbedder(Embedder):
    """Token-hashing embedder producing normalized dense vectors."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % self._dimensions
            sign = 1.0 if (int(digest, 16) >> 8) % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
