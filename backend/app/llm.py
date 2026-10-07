"""LLM clients: provider-independent interface with Groq primary + Ollama fallback.

The RAG layer depends only on the :class:`LLMClient` protocol, never on a
concrete provider, so generation can be swapped or chained entirely by
configuration and the retrieval pipeline is untouched.

Providers:
- :class:`GroqLLMClient`     -- hosted Groq API (OpenAI-compatible endpoint)
- :class:`OllamaLLMClient`   -- local Ollama server (``POST /api/chat``)
- :class:`FallbackLLMClient` -- tries providers in order; :func:`build_llm_client`
  builds the default chain Groq -> Ollama (``LLM_PROVIDER=auto``).

Failures are classified so the caller can tell a configuration problem from a
provider availability problem without inspecting raw HTTP errors, and so the
fallback chain knows what is worth retrying with the next provider.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)

#: OpenAI-compatible chat-completions base URL used by Groq.
GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

#: Groq's edge (Cloudflare) rejects the default ``Python-urllib`` User-Agent
#: with 403 "error code: 1010", so every hosted request identifies itself.
USER_AGENT = "unimind-rag/0.1"


class LLMError(RuntimeError):
    """Base class for every LLM-layer failure."""


class LLMConfigurationError(LLMError):
    """Invalid or missing configuration (bad key, unknown provider, ...).

    Never a fallback trigger: the deployment itself must be fixed.
    """


class LLMUnavailableError(LLMError):
    """The provider could not be reached or is refusing service."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMGenerationError(LLMError):
    """The provider answered, but the completion is unusable."""


@dataclass(frozen=True)
class LLMResponse:
    """A generated completion plus observable provider metadata."""

    text: str
    provider: str
    model: str


class LLMStreamHandle:
    """Incrementally generated completion (streaming) with final metadata.

    ``tokens`` yields text deltas as they arrive from the provider. After the
    stream ends, ``provider``/``model`` describe who generated the answer and
    ``text`` holds the full completion (safety net for chunk loss; normally
    equal to the concatenation of the deltas).
    """

    def __init__(self, tokens: Iterator[str], provider: str, model: str) -> None:
        self._tokens = tokens
        self.provider = provider
        self.model = model
        self._text_parts: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self._text_parts)

    def __iter__(self) -> Generator[str, None, None]:
        for token in self._tokens:
            self._text_parts.append(token)
            yield token


def _sse_data_payload(raw_line: str) -> str:
    """Extract the payload of one SSE ``data:`` line (empty when absent)."""
    return raw_line[5:].strip() if raw_line.startswith("data:") else ""


def _history_messages(history: list[dict] | None) -> list[dict]:
    """Convert sanitized history into provider chat messages."""
    messages: list[dict] = []
    for turn in history or []:
        role = turn.get("role")
        content = str(turn.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content[:2000]})
    return messages


def _limited_history(history: list[dict] | None, *, limit: int = 10) -> list[dict]:
    return _history_messages(history)[-limit:] if history else []


class LLMClient(Protocol):
    """Interface every generation provider satisfies (all the RAG layer needs)."""

    #: Short provider label, surfaced in API responses for observability.
    name: str

    def generate(
        self, *, system_prompt: str, user_query: str, context: str, history: list[dict] | None = None
    ) -> LLMResponse:
        """Generate an answer for ``user_query`` grounded in ``context``."""
        ...


def build_user_content(context: str, user_query: str) -> str:
    """Assemble one user turn: evidence first, question and task last.

    Small local models handle this structure far better than several
    consecutive user messages, and it keeps the prompt identical across
    providers: the same system prompt + this turn drive Groq and Ollama
    alike, so both providers behave the same way.
    """
    return (
        "Evidence: numbered university sources, cited as [1], [2], ...\n"
        f"{context}\n\n"
        f"Question: {user_query}\n\n"
        "Using only the evidence above, answer the question the user asked: "
        "match the requested style, length and structure, write naturally in "
        "your own words, and cite the sources you use. If any source above "
        "contains the requested information -- even briefly, indirectly, or in "
        "different wording -- answer from it rather than declining. Only use "
        "facts the evidence states; if the evidence truly does not contain what "
        "is needed, say so instead of guessing."
    )


def _error_body_snippet(exc: urllib.error.HTTPError, limit: int = 200) -> str:
    """Read the first bytes of an error response body for diagnostics.

    Providers do not echo the API key in error bodies, and the snippet only
    ever reaches the server log -- the API maps errors to generic messages.
    """
    try:
        return exc.read().decode("utf-8", errors="replace").strip()[:limit]
    except Exception:  # noqa: BLE001 -- diagnostics must never raise
        return ""


def _raise_http_error(exc: urllib.error.HTTPError, provider: str) -> None:
    """Map an HTTP status code to the matching LLM error class.

    - 401/403: rejected credentials or blocked client -> configuration error
      (no fallback; the response body, e.g. Cloudflare's "error code: 1010",
      distinguishes a WAF block from a real key problem)
    - 408: request timeout
    - 429 or 5xx: provider overloaded/errored -> unavailable (fallback-worthy)
    - other 4xx: the request itself was rejected -> generation error
    """
    code = exc.code
    body = _error_body_snippet(exc)
    detail = f" (response: {body})" if body else ""
    if code in (401, 403):
        raise LLMConfigurationError(
            f"{provider} authentication failed (HTTP {code}); check the API key"
            f"{detail}"
        ) from exc
    if code == 408:
        raise LLMTimeoutError(f"{provider} request timed out (HTTP {code}){detail}") from exc
    if code == 429 or code >= 500:
        raise LLMUnavailableError(
            f"{provider} service unavailable (HTTP {code}){detail}"
        ) from exc
    raise LLMGenerationError(
        f"{provider} rejected the request (HTTP {code}){detail}"
    ) from exc


def _extract_chat_text(body: dict, provider: str) -> str:
    """Extract assistant text from an Ollama-style ``/api/chat`` response."""
    message = body.get("message")
    if not isinstance(message, dict):
        raise LLMGenerationError(f"{provider} returned a malformed response: missing message")
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise LLMGenerationError(f"{provider} returned an empty completion")
    return text


def _extract_openai_text(body: dict, provider: str) -> str:
    """Extract assistant text from an OpenAI-compatible completions response."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMGenerationError(f"{provider} returned a malformed response: missing choices")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise LLMGenerationError(f"{provider} returned an empty completion")
    return text


class OllamaLLMClient:
    """Generation via a local Ollama server (``POST /api/chat``)."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, temperature: float, timeout: float) -> None:
        if not base_url:
            raise LLMConfigurationError("OLLAMA_BASE_URL is not set")
        if not model:
            raise LLMConfigurationError("OLLAMA_MODEL is not set")
        if temperature < 0:
            raise LLMConfigurationError(f"temperature must be non-negative, got {temperature}")
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout

    def _payload(
        self, *, system_prompt: str, user_query: str, context: str, stream: bool, history: list[dict] | None = None
    ) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *_limited_history(history),
                {"role": "user", "content": build_user_content(context, user_query)},
            ],
            "stream": stream,
            "options": {
                "temperature": self._temperature,
                # Stop the model from drifting past the supplied evidence.
                "num_ctx": 8192,
            },
        }

    def _request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    @staticmethod
    def _wrap_transport_error(exc: Exception) -> LLMError:
        """Convert a transport-level exception into the matching LLM error."""
        if isinstance(exc, TimeoutError):
            return LLMTimeoutError("Ollama request timed out")
        if isinstance(exc, urllib.error.HTTPError):
            _raise_http_error(exc, "Ollama")
            return LLMUnavailableError("Ollama request failed")  # pragma: no cover
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        return LLMUnavailableError(f"Ollama request failed: {reason}")

    def generate(
        self, *, system_prompt: str, user_query: str, context: str, history: list[dict] | None = None
    ) -> LLMResponse:
        payload = self._payload(
            system_prompt=system_prompt, user_query=user_query, context=context, stream=False,
            history=history,
        )
        request = self._request(payload)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, TimeoutError, urllib.error.URLError, OSError) as exc:
            raise self._wrap_transport_error(exc) from exc
        except ValueError as exc:
            raise LLMGenerationError("Ollama returned a non-JSON response") from exc

        return LLMResponse(
            text=_extract_chat_text(body, "Ollama"), provider=self.name, model=self.model
        )

    def stream_generate(
        self, *, system_prompt: str, user_query: str, context: str, history: list[dict] | None = None
    ) -> LLMStreamHandle:
        """Stream the completion as NDJSON deltas from Ollama's chat API."""
        payload = self._payload(
            system_prompt=system_prompt, user_query=user_query, context=context, stream=True,
            history=history,
        )
        request = self._request(payload)
        try:
            response = urllib.request.urlopen(request, timeout=self._timeout)
        except (urllib.error.HTTPError, TimeoutError, urllib.error.URLError, OSError) as exc:
            raise self._wrap_transport_error(exc) from exc

        def token_iter() -> Iterator[str]:
            try:
                with response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError as exc:
                            raise LLMGenerationError("Ollama returned a non-JSON stream line") from exc
                        if not event.get("done"):
                            token = event.get("message", {}).get("content", "")
                            if token:
                                yield token
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                raise LLMUnavailableError(f"Ollama stream failed: {exc}") from exc

        return LLMStreamHandle(token_iter(), provider=self.name, model=self.model)


class GroqLLMClient:
    """Generation via Groq's hosted OpenAI-compatible chat completions API."""

    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = GROQ_DEFAULT_BASE_URL,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> None:
        if api_key is None or not api_key.strip():
            raise LLMConfigurationError("GROQ_API_KEY is not set; cannot build the Groq provider")
        if not model:
            raise LLMConfigurationError("GROQ_MODEL is not set")
        if not base_url:
            raise LLMConfigurationError("GROQ_BASE_URL is not set")
        if temperature < 0:
            raise LLMConfigurationError(f"temperature must be non-negative, got {temperature}")
        #: Kept private: never logged, never included in responses.
        self._api_key = api_key.strip()
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout

    def generate(
        self, *, system_prompt: str, user_query: str, context: str, history: list[dict] | None = None
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *_limited_history(history),
                {"role": "user", "content": build_user_content(context, user_query)},
            ],
            "temperature": self._temperature,
            "stream": False,
        }

        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            _raise_http_error(exc, "Groq")
        except TimeoutError as exc:
            raise LLMTimeoutError("Groq request timed out") from exc
        except urllib.error.URLError as exc:
            raise LLMUnavailableError(f"Groq request failed: {exc.reason}") from exc
        except OSError as exc:
            raise LLMUnavailableError(f"Groq request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMGenerationError("Groq returned a non-JSON response") from exc

        return LLMResponse(
            text=_extract_openai_text(body, "Groq"), provider=self.name, model=self.model
        )

    def stream_generate(
        self, *, system_prompt: str, user_query: str, context: str, history: list[dict] | None = None
    ) -> LLMStreamHandle:
        """Stream the completion as SSE ``chat.completion.chunk`` events from Groq."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *_limited_history(history),
                {"role": "user", "content": build_user_content(context, user_query)},
            ],
            "temperature": self._temperature,
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": USER_AGENT,
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            _raise_http_error(exc, "Groq")
        except TimeoutError as exc:
            raise LLMTimeoutError("Groq request timed out") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailableError(f"Groq request failed: {exc}") from exc

        def token_iter() -> Iterator[str]:
            try:
                with response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        data_payload = _sse_data_payload(line)
                        if not data_payload:
                            continue
                        if data_payload == "[DONE]":
                            return
                        try:
                            event = json.loads(data_payload)
                        except ValueError as exc:
                            raise LLMGenerationError("Groq returned a non-JSON stream event") from exc
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        token = (choices[0].get("delta") or {}).get("content") or ""
                        if token:
                            yield token
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                raise LLMUnavailableError(f"Groq stream failed: {exc}") from exc

        return LLMStreamHandle(token_iter(), provider=self.name, model=self.model)


#: Provider failures that justify trying the next provider in the chain.
_FALLBACK_ERRORS = (LLMUnavailableError, LLMTimeoutError, LLMGenerationError)


class FallbackLLMClient:
    """Try providers in order, transparently failing over on provider errors.

    Configuration errors (for example a rejected API key) propagate
    immediately: they mean the deployment is wrong and must not be masked by a
    fallback. When a later provider succeeds, the response is labeled
    ``<provider>_fallback`` so failover is observable to the RAG layer.
    """

    name = "fallback"

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise LLMConfigurationError("FallbackLLMClient needs at least one provider")
        self._clients = list(clients)

    @property
    def stream_provider_label(self) -> str:
        """Provider label before failover is known (the first client's name)."""
        return self._clients[0].name

    def generate(
        self, *, system_prompt: str, user_query: str, context: str, history: list[dict] | None = None
    ) -> LLMResponse:
        failures: list[str] = []
        for index, client in enumerate(self._clients):
            try:
                response = client.generate(
                    system_prompt=system_prompt, user_query=user_query, context=context,
                    history=history,
                )
            except _FALLBACK_ERRORS as exc:
                failures.append(f"{client.name}: {exc}")
                continue  # provider-level failure -> try the next provider
            if index == 0:
                return response
            # Failover happened; make it observable in the response metadata.
            return LLMResponse(
                text=response.text,
                provider=f"{response.provider}_fallback",
                model=response.model,
            )
        raise LLMUnavailableError("All LLM providers failed: " + "; ".join(failures))

    def stream_generate(
        self, *, system_prompt: str, user_query: str, context: str, history: list[dict] | None = None
    ) -> LLMStreamHandle:
        """Stream from the first provider that starts answering.

        A connection-level failure on ``connect`` falls over to the next
        provider. Once deltas have been emitted to the client, the stream can
        no longer restart cleanly, so a mid-stream provider error is surfaced
        as an error (the API sends a terminal error event).
        """
        failures: list[str] = []
        for index, client in enumerate(self._clients):
            streamer = getattr(client, "stream_generate", None)
            if streamer is None:
                failures.append(f"{client.name}: no streaming support")
                continue
            try:
                handle = streamer(
                    system_prompt=system_prompt, user_query=user_query, context=context,
                    history=history,
                )
            except _FALLBACK_ERRORS as exc:
                failures.append(f"{client.name}: {exc}")
                continue
            if index == 0:
                return handle
            # Failover happened; label the final metadata observably.
            return LLMStreamHandle(
                iter(handle),
                provider=f"{handle.provider}_fallback",
                model=handle.model,
            )
        raise LLMUnavailableError("All LLM providers failed: " + "; ".join(failures))


def _build_ollama(settings: Settings) -> OllamaLLMClient:
    return OllamaLLMClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )


def _build_groq(settings: Settings) -> GroqLLMClient:
    return GroqLLMClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )


def build_llm_client(settings: Settings) -> LLMClient:
    """Create the LLM client chain selected by ``LLM_PROVIDER``.

    - ``auto`` (default): Groq primary, Ollama fallback. Without ``GROQ_API_KEY``
      the Groq provider is skipped with a warning and Ollama is used directly.
    - ``groq`` / ``ollama``: that single provider, no fallback.
    """
    provider = settings.llm_provider.strip().lower()
    if provider == "ollama":
        return _build_ollama(settings)
    if provider == "groq":
        return _build_groq(settings)
    if provider == "auto":
        clients: list[LLMClient] = []
        if settings.groq_api_key:
            clients.append(_build_groq(settings))
        else:
            logger.warning(
                "GROQ_API_KEY is not set; skipping the Groq provider and using Ollama only."
            )
        clients.append(_build_ollama(settings))
        if len(clients) == 1:
            return clients[0]
        return FallbackLLMClient(clients)
    raise LLMConfigurationError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}' (expected 'auto', 'groq' or 'ollama')"
    )
