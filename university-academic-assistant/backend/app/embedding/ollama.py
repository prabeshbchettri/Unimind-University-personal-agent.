"""Ollama embedding backend.

Calls Ollama's ``/api/embed`` endpoint with a BGE (or any compatible)
embedding model. The embedding model is configured via ``EMBEDDING_MODEL``
and is deliberately separate from the chat model (``OLLAMA_MODEL``) used for
answer generation.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.embedding.base import Embedder
from app.embedding.errors import EmbeddingError, EmbeddingUnavailableError


class OllamaEmbedder(Embedder):
    """Embed text by calling an Ollama embedding model over HTTP."""

    def __init__(
        self,
        base_url: str,
        model: str,
        default_dimensions: int = 384,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._default_dimensions = default_dimensions
        self._dimensions: int | None = None
        self._timeout = timeout

    @property
    def dimensions(self) -> int:
        """Dimensionality, inferred from the first embedding response."""
        return self._dimensions or self._default_dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": list(texts)}
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailableError(
                f"Ollama embedding endpoint unreachable ({self.base_url}): {exc}"
            ) from exc

        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise EmbeddingError(
                f"Ollama returned no embeddings for model '{self.model}'."
            )
        if self._dimensions is None:
            self._dimensions = len(embeddings[0])
        return [list(map(float, embedding)) for embedding in embeddings]
