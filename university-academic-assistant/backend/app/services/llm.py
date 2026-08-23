"""LLM service interface (placeholder).

Encapsulates all LLM interaction (Ollama + Llama 3.1 8B). Later phases add
prompting, generation, and structured-output helpers here.
"""

from __future__ import annotations

from app.services.base import Service


class LLMService(Service):
    """Generates answers from retrieved context."""

    async def health(self) -> dict:
        return {"name": self.name, "status": "ok"}

    async def generate(self, prompt: str) -> str:
        """Generate a completion for ``prompt``.

        Not implemented in Phase 1.
        """
        raise NotImplementedError("LLM generation is implemented in a later phase.")
