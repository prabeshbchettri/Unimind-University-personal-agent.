"""Stub LLM client.

A deterministic, dependency-free LLM backend for tests and offline development.
By default it returns a fixed answer; a ``responder`` callable may be injected
to make behaviour prompt-aware (e.g. detecting that no context was retrieved).
"""

from __future__ import annotations

from collections.abc import Callable

from app.llm.base import LLMClient

Responder = Callable[[str], str]


class StubLLMClient(LLMClient):
    """Returns a canned or callable-driven completion."""

    def __init__(
        self,
        default_answer: str = "Stub answer: no LLM backend configured.",
        responder: Responder | None = None,
    ) -> None:
        self._default_answer = default_answer
        self._responder = responder

    @property
    def name(self) -> str:
        return "stub"

    def complete(self, system: str, prompt: str) -> str:
        if self._responder is not None:
            return self._responder(prompt)
        return self._default_answer
