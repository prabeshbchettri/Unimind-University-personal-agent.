"""Embedding clients.

The rest of the system depends on the :class:`EmbeddingClient` protocol, not on
a concrete provider, so the embedding implementation can be swapped by
configuration only.

``local`` (default) runs `fastembed <https://github.com/qdrant/fastembed>`_
in-process: no external server, ONNX CPU inference, the model file is
downloaded on first use and cached. This is the practical choice for a
portfolio project that must run on a laptop without a GPU.

``ollama`` calls a local Ollama server (``/api/embed``) and exists because the
same server is already part of the stack as the LLM fallback provider.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from app.config import Settings

#: Stable on-disk cache for downloaded embedding models, so the model is
#: downloaded once and survives cache cleanups of the OS temp directory.
MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".models"


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails or returns unusable output."""


class EmbeddingClient(Protocol):
    """Interface every embedding provider must satisfy."""

    #: Short provider label stored alongside vectors and shown in metadata.
    name: str

    #: Vector dimension; the Qdrant collection is created with this size.
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...


def _validate_vectors(vectors: list[list[float]], expected: int) -> list[list[float]]:
    """Validate embedding output shape before it reaches the vector store."""
    if len(vectors) != expected:
        raise EmbeddingError(
            f"Embedding provider returned {len(vectors)} vectors for {expected} texts"
        )
    for vector in vectors:
        if not vector or not all(isinstance(value, float) and math.isfinite(value) for value in vector):
            raise EmbeddingError("Embedding provider returned a malformed vector")
    return vectors


class LocalEmbeddingClient:
    """In-process embeddings via fastembed (ONNX runtime, CPU)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.name = "local"
        self._model_name = model_name
        self._model: object | None = None
        self._dim: int | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(
                    model_name=self._model_name,
                    cache_dir=str(MODEL_CACHE_DIR),
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load local embedding model '{self._model_name}': {exc}"
                ) from exc

    @property
    def dim(self) -> int:
        """Vector dimension, discovered from the model itself on first use."""
        self._ensure_model()
        if self._dim is None:
            assert self._model is not None
            probe = next(iter(self._model.embed(["dimension probe"])))  # type: ignore[attr-defined]
            self._dim = int(len(probe))
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_model()
        assert self._model is not None

        vectors: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                batch_vectors = [
                    [float(value) for value in embedding]
                    for embedding in self._model.embed(batch)  # type: ignore[attr-defined]
                ]
            except Exception as exc:
                raise EmbeddingError(f"Local embedding failed: {exc}") from exc
            _validate_vectors(batch_vectors, len(batch))
            vectors.extend(batch_vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OllamaEmbeddingClient:
    """Embeddings served by a local Ollama server (``POST /api/embed``)."""

    def __init__(self, base_url: str, model: str, timeout: float = 30.0) -> None:
        self.name = f"ollama:{model}"
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        request = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=json.dumps({"model": self._model, "input": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed with HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed: {exc.reason if isinstance(exc, urllib.error.URLError) else exc}"
            ) from exc

        vectors = body.get("embeddings")
        if not isinstance(vectors, list):
            raise EmbeddingError("Ollama returned a malformed embedding response")
        return _validate_vectors(vectors, len(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """Create the embedding client selected by ``EMBEDDING_PROVIDER``."""
    provider = settings.embedding_provider.strip().lower()
    if provider == "local":
        return LocalEmbeddingClient(model_name=settings.embedding_model)
    if provider == "ollama":
        return OllamaEmbeddingClient(
            base_url=settings.ollama_base_url, model=settings.ollama_embedding_model
        )
    raise EmbeddingError(
        f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}' (expected 'local' or 'ollama')"
    )
