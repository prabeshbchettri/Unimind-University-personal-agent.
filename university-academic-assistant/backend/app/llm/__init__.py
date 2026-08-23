"""LLM generation.

Provides an abstraction over the answer-generation model (Llama 3.1 8B via
Ollama by default). The model used here is deliberately separate from the
embedding model used for retrieval.
"""

from .base import LLMClient
from .errors import LLMError, LLMUnavailableError
from .factory import build_llm
from .ollama import OllamaLLMClient
from .stub import StubLLMClient

__all__ = [
    "LLMClient",
    "OllamaLLMClient",
    "StubLLMClient",
    "LLMError",
    "LLMUnavailableError",
    "build_llm",
]