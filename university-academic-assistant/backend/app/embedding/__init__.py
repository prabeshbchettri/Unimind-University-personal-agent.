"""Embedding backends.

Provides dense text embeddings, kept separate from the chat LLM. The default
``build_embedder`` returns a deterministic fallback unless a real backend
(Ollama/BGE) is configured.
"""

from .base import Embedder
from .deterministic import DeterministicEmbedder
from .errors import EmbeddingError, EmbeddingUnavailableError
from .factory import build_embedder
from .ollama import OllamaEmbedder

__all__ = [
    "Embedder",
    "DeterministicEmbedder",
    "OllamaEmbedder",
    "EmbeddingError",
    "EmbeddingUnavailableError",
    "build_embedder",
]
