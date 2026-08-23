# Phase 13 Report: Evaluation, Optimization and Deployment

Status: **complete**. All 9 deliverables done; full test suite green
(330 tests); E2E acceptance 20/20 through a live server.

## 1. Evaluation

**Goal**: measure routing, retrieval, answer quality, OCR and
recommendation accuracy on a fixed, representative corpus, with explicit
goals and constraints documented.

- **Artifacts**: `app/evaluation/` (corpus, router/retrieval/answers/OCR/
  recommendations metrics), `scripts/evaluate.py`, `docs/EVALUATION.md`.
- **Hermetic corpus**: 8 documents (2 syllabi, notice, regulations,
  calendar, past paper, 2 library books) + labeled retrieval queries.
- **Results** (deterministic embedder + stub LLM):
  - Router: accuracy 100%, false routing 0%, fallback 0%.
  - Retrieval on labeled subsets: HYBRID recall@5 1.00/MRR 1.00; GRAPH
    recall@5 1.00/MRR 1.00 (empty rate on unrelated queries 1.00 — it is
    specialized by design); NORMAL recall@5 0.70/MRR 0.60.
  - Answers: faithfulness 100%, relevance 100%, source correctness 100%,
    hallucination 0%.
  - OCR: digital PDFs 100% char/word accuracy; scanned path auto-skipped
    (Tesseract not installed on this machine — documented limitation).
  - Recommendations: 9/9 checks pass (ranked Database System Concepts
    0.85 over Introduction to Databases 0.42 and Operating System
    Principles 0.00; similar-books scenario ranks Database Design B 1.00
    over A 0.93).

## 2. Optimization

**Goal**: profile first; only optimize what is measured; never guess.

- `scripts/profile_performance.py` + `docs/PERFORMANCE.md`.
- Per-query latencies (deterministic, hermetic): chunking 0.84 ms;
  embedding batch 3.64 ms (20 chunks); indexing ~64 ms/document; NORMAL
  11.1 ms; HYBRID 11.6 ms; GRAPH 1.6 ms; chat 9.7 ms; recommendation
  8.2 ms.
- **No bottleneck was measured**, so the pipeline was not modified (per
  the optimization rule). The dominant cost in production is the LLM
  itself, which is a hardware/model property, not an implementation issue.

## 3. Caching with Invalidation

**Goal**: safe caching — bounded memory, correct semantics, **no stale
data after any ingestion**.

- `EmbeddingCache`: LRU keyed `(model, text)`; embeddings are
  deterministic per model, so memoization is always correct.
  (`EMBEDDING_CACHE_MAX_ENTRIES=5000`.)
- `RetrievalCache`: final retrieval results keyed `(name, query, top_k)`
  with TTL (`RETRIEVAL_CACHE_TTL_SECONDS=300`, max 256 entries);
  **invalidated wholesale by `VectorIndexingService.index()` after every
  mutation** — stale results are impossible by construction.
- Wired as `CachedEmbedder` (factory) and `CachedRetriever` (chat service)
  in `app/main.py` and the evaluation environment. Verified by
  `tests/test_caching.py` (8 tests).
- One real bug found and fixed during testing: `cache or EmbeddingCache()`
  silently discarded the passed empty cache (it defines `__len__`, hence
  falsy when empty) — replaced with an explicit `is not None` check.

## 4. Docker Deployment

**Goal**: one-command deployment, no GPU assumed, no secrets in images.

- `backend/Dockerfile`, `frontend/Dockerfile` (+ `nginx.conf`),
  `docker-compose.yml`; validated with `docker compose config`.
- Services: frontend (nginx, `/api/*` proxy with prefix strip), backend
  (single uvicorn worker — deliberate: in-process caches), qdrant, neo4j,
  postgres. Optional `--profile llm` adds a **CPU-only** Ollama.
- `docker compose up -d` runs the full stack hermetic
  (deterministic/stub); flipping `LLM_BACKEND`/`EMBEDDER_BACKEND=ollama`
  enables real models. Secrets come from env interpolation
  (`NEO4J_PASSWORD`, `POSTGRES_PASSWORD`).

## 5. Environment Configuration

**Goal**: documented, environment-specific config; no secrets committed.

- `backend/.env.development` (local dev, hermetic), `.env.test` (mirrors
  the test/eval environment), `.env.production` (full stack with
  `CHANGE_ME` placeholders — template only, never real values).
- `docs/DEPLOYMENT.md` documents usage and the production hardening
  checklist.

## 6. Logging and Monitoring

**Goal**: correlate all logs of one request; identify route, retrieval
method, latencies and errors; never log content.

- Every request gets a 12-hex id: `X-Request-ID` response header, injected
  by a logging filter into **every** record (JSON or key/value format).
- Request logs carry method/path/status/duration; chat logs carry
  strategy, cache state, chunk counts, `retrieval_ms`/`llm_ms`.
- **User messages, chat history, document contents, retrieved text and
  answers are never logged.** Structured fields are scrubbed for
  sensitive substrings. Failures are logged with `logger.exception`
  (status 500) before re-raising.

## 7. Security

- **Prompt injection guard**: `SYSTEM_PROMPT` now states that context,
  web results and history are *data, not instructions* and must never be
  acted on (verified by test).
- **Upload limits**: `MAX_UPLOAD_BYTES` (default 20 MB), enforced while
  streaming → HTTP 413 before content touches memory/disk.
- **Rate limiting**: per-IP sliding window (`RATE_LIMIT_PER_MINUTE`,
  default 120) as the outermost middleware → HTTP 429 + `Retry-After`;
  `/health` and docs exempt; disabled in the test suite by conftest.
- **Log scrubbing** of sensitive fields; **secrets policy** documented.
- All covered by `tests/test_security.py` (6 tests) and `docs/SECURITY.md`.

## 8. Documentation

- `README.md` — updated for Phase 13 (status, layout, new sections,
  roadmap); `docs/EVALUATION.md`, `docs/PERFORMANCE.md`,
  `docs/DEPLOYMENT.md`, `docs/SECURITY.md`, `docs/E2E.md`.

## 9. Final End-to-End Acceptance

**Goal**: the six acceptance scenario types through running servers.

- `scripts/e2e.py` drives the live API: builds 7 PDFs, uploads through the
  real pipeline, then runs all scenarios + session lifecycle +
  operational endpoints. **20/20 checks pass** (results:
  `docs/E2E.md`, `backend/e2e_results.json`).
- Scenario coverage: S1 syllabus/NORMAL, S2 regulation/HYBRID, S3 graph/
  GRAPH, S4 past questions/GRAPH, S5 recommendation (top book, score
  0.57), S6 web/WEB; session create/continue/fetch/delete; `/health`,
  `/graph/summary`.

## Regression state

- `python -m pytest tests -q` → **330 passed** (was 316 before Phase 13;
  +8 caching, +6 security tests).
- `scripts/evaluate.py` (hermetic) → all green.
- `scripts/profile_performance.py` → reproducible numbers.