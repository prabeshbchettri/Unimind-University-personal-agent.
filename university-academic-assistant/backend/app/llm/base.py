"""LLM client interface.

Encapsulates all interaction with the answer-generation model. Phase 5 ships
the Ollama (Llama 3.1 8B) client plus a deterministic stub for offline/tests.
The interface keeps the model swappable without touching the rest of the app.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Generates completions given a system prompt and a user prompt."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str:
        """Return a completion for ``prompt`` guided by ``system``.

        This is a synchronous call; callers offload it to a worker thread.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Release any resources held by the client."""
        return None
