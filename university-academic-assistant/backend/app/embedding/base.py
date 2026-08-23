"""Embedder interface.

Embeddings are deliberately kept separate from the LLM: an embedding model
produces the vector representation used for retrieval, while the chat LLM
(Llama 3.1) performs answer generation. This separation is preserved across
every backend so backends can be swapped without touching the rest of the app.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Embedder(ABC):
    """Converts text into dense vector representations."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the produced vectors."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` into a list of dense vectors (one per input)."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]

    async def close(self) -> None:
        """Release any resources held by the embedder."""
        return None
