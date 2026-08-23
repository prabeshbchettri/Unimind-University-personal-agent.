"""LLM client factory.

Builds an :class:`LLMClient` from settings:

- ``ollama`` — Llama 3.1 8B (or any Ollama model) via ``/api/generate``
- ``stub`` — deterministic fallback (tests / offline dev)
- ``auto`` — Ollama when a base URL is configured, otherwise the stub
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMClient
from app.llm.ollama import OllamaLLMClient
from app.llm.stub import StubLLMClient

# Default stub reply follows the grounding rules: without a backend the
# "answer" never asserts university facts.
STUB_DEFAULT_ANSWER = "I could not find sufficient information in the available sources."


def build_llm(settings: Settings) -> LLMClient:
    """Return the configured LLM backend."""
    backend = (settings.llm_backend or "auto").strip().lower()

    if backend == "ollama":
        return OllamaLLMClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

    if backend == "stub":
        return StubLLMClient(default_answer=STUB_DEFAULT_ANSWER)

    if settings.ollama_base_url:
        return OllamaLLMClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

    return StubLLMClient(default_answer=STUB_DEFAULT_ANSWER)
