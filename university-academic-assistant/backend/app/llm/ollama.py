"""Ollama LLM client.

Calls Ollama's ``/api/generate`` endpoint with the configured chat model
(Llama 3.1 8B by default). The embedding model used for retrieval is a
different model (see ``app/embedding``).
"""

from __future__ import annotations

import httpx

from app.llm.base import LLMClient
from app.llm.errors import LLMUnavailableError


class OllamaLLMClient(LLMClient):
    """Completes prompts by calling an Ollama model over HTTP."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama LLM endpoint unreachable ({self.base_url}, model '{self.model}'): {exc}"
            ) from exc

        data = response.json()
        return (data.get("response") or "").strip()
