# Security

Security posture of the University Academic Assistant (Phase 13). Each
measure is documented with where it lives and how to configure it.

## Secrets

- **Nothing is hard-coded.** All credentials come from environment variables
  or the `.env` file (see `backend/.env.production` — a template with
  placeholders, never real values). `backend/.env*` files are git-ignored.
- **Logging scrubs sensitive fields.** Structured log fields are passed
  through a scrubber (`app/core/logging.py::_scrub`) that redacts anything
  containing `password`, `api_key`, `secret`, `token` or `authorization`.

## Request logging and correlation

- Every HTTP request gets a 12-hex-character id, echoed in the
  `X-Request-ID` response header and injected into **every** log line emitted
  during that request (startup lines use `-`).
- Logs contain only routing metadata (method, path, status, duration) and
  service metrics (retrieval strategy, chunk counts, latencies). **User
  messages, chat history, document contents, retrieved text and answer text
  are never logged.**
- Latencies are measured per stage (`retrieval_ms`, `llm_ms`) in
  `app/services/chat.py`.

## Prompt injection

Retrieved documents and web results are treated as **data, not
instructions**. `SYSTEM_PROMPT` (`app/rag/context.py`) instructs the model
to ignore any instructions, commands or role changes found inside the
context, web results or conversation history. The prompt also enforces
grounding (answer from context, admit insufficient evidence, cite sources),
which structurally limits the impact of a hostile document.

## Rate limiting

- Per-client-IP sliding window: `rate_limit_per_minute` (default 120),
  enforced by the outermost HTTP middleware (`app/main.py`).
- Excess requests get `429 Too Many Requests` with a `Retry-After` header;
  `/health` and the interactive docs are exempt.
- Disabled in the automated test suite via `RATE_LIMIT_ENABLED=false`
  (see `tests/conftest.py`); the limiter is process-local, so front it with
  a shared limiter when scaling horizontally.

## Upload limits

- `max_upload_bytes` (default 20 MB). Oversized uploads are rejected with
  `413 Payload Too Large` **while streaming**, before the content reaches
  memory or disk (`app/api/documents.py::_save_upload`).
- Only `.pdf` files are accepted (anything else: `415`).

## Data exposure

- Chat sessions: no personal memory, personality or preferences; only
  bounded recent history is injected into prompts
  (`app/services/chat.py`).
- Web results are rendered in a separate, clearly attributed block so they
  cannot masquerade as university documents.
- Error responses never include internal details (stacks, config); the
  frontend maps statuses to safe user-facing messages.

## Dependency and transport notes

- The API only listens on configured hosts; CORS is restricted to the
  configured origins (empty list in production unless a browser origin is
  needed — production traffic is same-origin through nginx).
- In Docker, Qdrant/Neo4j/Postgres published ports are for local
  administration; restrict them (`127.0.0.1:` binding) or remove them in
  production. Use TLS on the public endpoint (terminate at nginx).