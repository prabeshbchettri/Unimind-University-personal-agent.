"""Structured application logging.

Provides a single ``setup_logging`` entry point used at application startup.
Logs include structured key/value fields so that later phases (document
processing, retrieval, LLM calls) can be correlated. Sensitive configuration
values are never logged, and user messages / document contents / retrieved
text are never placed in structured fields.

A :class:`RequestIdFilter` injects the current request id (see
``app.core.context``) into every record so all lines of one request —
including service-level metrics — can be correlated.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.config import settings
from app.core.context import get_request_id

# Reserved keys that must never appear in structured output (defence in depth).
_SENSITIVE_SUBSTRINGS = (
    "password",
    "api_key",
    "apikey",
    "token",
    "secret",
    "authorization",
)


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every emitted record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict) and extra:
            payload.update({k: _scrub(v) for k, v in extra.items()})
        return json.dumps(payload, default=str)


class KeyValueFormatter(logging.Formatter):
    """Render log records as readable ``key=value`` lines for local dev."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} [{record.levelname}] {record.name}: {record.getMessage()}"
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict) and extra:
            parts = " ".join(f"{k}={_scrub(v)}" for k, v in extra.items())
            base = f"{base} | {parts}"
        base = f"{base} | request_id={getattr(record, 'request_id', '-')}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def _scrub(value: object) -> object:
    """Best-effort scrubbing of sensitive values appearing in structured fields."""
    if isinstance(value, str):
        lowered = value.lower()
        if any(part in lowered for part in _SENSITIVE_SUBSTRINGS):
            return "***"
    return value


def setup_logging() -> None:
    """Configure the root logger once.

    A root handler writes structured logs to stdout. The format (JSON or
    key/value text) is controlled by ``LOG_FORMAT``, and every record carries
    the current request id.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(KeyValueFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Keep third-party loggers reasonably quiet unless debugging.
    if not settings.debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger for ``name``."""
    return logging.getLogger(name)