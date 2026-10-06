# Adaptive University RAG Assistant

A small, explainable retrieval-augmented generation (RAG) system that answers
university students' questions about academic documents (policies, regulations,
course syllabi) and returns grounded answers with citations.

> **Implementation status: Phase 5 (LLM provider abstraction + fallback).**
> Implemented: PDF ingestion → embeddings → Qdrant, vector + BM25 retrieval,
> a deterministic query router (NORMAL | HYBRID), grounded LLM answers with
> citations (`POST /api/chat`), controlled insufficient-evidence responses,
> and a provider-independent LLM layer (Groq primary → Ollama fallback) with
> observable provider metadata.
> Not implemented yet: web search (optional/deferred). The React chat UI is
> connected end to end: answer + sources + strategy + provider are shown for
> every question.

---

## 1. Project purpose

Students need fast, trustworthy answers from official academic documents. A
plain language model tends to invent policy details, which is unacceptable for
academic regulations. This project therefore answers questions **from retrieved
document evidence only**, cites the source of every claim, and says so
explicitly when the available documents do not contain enough information.

The engineering goal is a *simple, reliable, explainable* pipeline rather than a
large stack of technologies.

## 2. Architecture overview

The target architecture (later phases) is:

```
User
  ↓
React frontend
  ↓
FastAPI backend
  ↓
RAG orchestrator
  ↓
Query analyzer
  ↓
Retrieval strategy: NORMAL (vector) | HYBRID (vector + BM25) | WEB (optional)
  ↓
Relevance filtering and context construction
  ↓
LLM gateway: Groq (primary) → Ollama (fallback)
  ↓
Answer + citations + strategy metadata
```

**Implemented today (Phases 1–5):**

```
Ingestion (offline):                      Query (online):
data/documents/*.pdf                      user question
  ↓  app/ingestion.py                       ↓  POST /api/chat (app/main.py)
  extract → clean → chunk → metadata        ↓  app/routing.py     deterministic router
  ↓  app/embeddings.py                      ↓  app/retriever.py   NORMAL | HYBRID (vector + BM25)
  EmbeddingClient (fastembed | Ollama)      ↓                     fused with RRF, deduplicated
  ↓  app/vectorstore.py                     ↓  app/context.py     relevance gate, dedupe, budget
Qdrant (embedded engine, data/qdrant_storage)  ↓  app/llm.py     LLMClient (Ollama)
  ↑  app/bm25.py (lexical index)            ↓  app/rag.py         grounding, citations
                                             ↓  answer + sources + strategy + retrieval metadata
                                               (insufficient evidence → controlled
                                                response, LLM not called)

React frontend (placeholder UI)        FastAPI backend
  App.jsx  ──────────(not yet)──────────►  GET /health, POST /api/chat
```

## 3. Current implementation status

| Area | Status |
| --- | --- |
| Repository structure | Done |
| Backend app + `GET /health` | Done |
| Centralized environment configuration | Done |
| Backend tests | Done |
| Frontend app that starts | Done (connected chat UI) |
| PDF ingestion (extract/clean/chunk/metadata) | Done |
| Embedding interface + local provider | Done |
| Qdrant integration (upsert/search) | Done |
| Ingestion CLI | Done |
| Demo corpus | Done (synthetic, `scripts/generate_demo_corpus.py`) |
| Retrieval API endpoint | Done (`POST /api/chat`) |
| Vector retriever (query side) | Done |
| Context builder (threshold/dedupe/budget) | Done |
| LLM provider (Ollama) behind interface | Done |
| Grounding + citations | Done |
| Insufficient-evidence behavior | Done |
| Deterministic query router (NORMAL/HYBRID) | Done |
| Hybrid retrieval (vector + BM25, RRF fusion) | Done |
| Strategy + reason in API response | Done |
| Groq provider + fallback (LLM gateway) | Done |
| LLM error classification (config/timeout/unavailable/generation) | Done |
| Provider metadata (`groq` / `ollama` / `ollama_fallback`) | Done |
| Web search strategy (WEB) | Not started (optional/deferred) |
| OCR for scanned PDFs | Not started |
| Frontend ↔ backend integration | Done (React chat → `/api/chat` via Vite dev proxy; CORS configurable via `CORS_ALLOWED_ORIGINS`) |

## 4. Backend setup

Requires Python 3.11+ (developed on 3.13).

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash); use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Qdrant runs locally with **no server and no Docker**: the default
`QDRANT_URL=local` uses the Qdrant engine embedded in the Python process,
persisted to `data/qdrant_storage/`. Pointing `QDRANT_URL` at an
`http://...` address switches to a Qdrant server if you ever want one — no
code changes needed.

## 5. Demo corpus and ingestion

Generate the deterministic synthetic demo PDFs (university policies and a
course syllabus — no private or internet-sourced documents):

```bash
cd backend
python ../scripts/generate_demo_corpus.py
```

Then ingest them into Qdrant:

```bash
cd backend
python -m app.ingest --recreate
```

The CLI prints how many chunks were created, which embedding provider ran,
and how many vectors were stored. Re-running it is safe: chunk IDs are
deterministic, so re-ingestion overwrites instead of duplicating.

## 6. Frontend setup

Requires Node.js 20+ (developed on Node 24).

```bash
cd frontend
npm install
```

## 7. Environment configuration

Copy the example file and fill in real values:

```bash
cd backend
cp .env.example .env
```

`.env` is git-ignored and must never be committed. `.env.example` contains
placeholders only.

| Variable | Purpose | Used in Phase 2? |
| --- | --- | --- |
| `APP_ENV` | Runtime environment label | Yes (health response) |
| `API_HOST` / `API_PORT` | Where the API listens | Yes |
| `DOCUMENTS_DIR` | Where ingestion looks for PDFs | Yes |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Character-based chunking | Yes |
| `EMBEDDING_PROVIDER` | `local` (fastembed) or `ollama` | Yes |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | Local model and vector size | Yes |
| `OLLAMA_EMBEDDING_MODEL` | Model when provider is `ollama` | Optional |
| `QDRANT_URL` / `QDRANT_COLLECTION` | Vector store (`local` = embedded engine) | Yes |
| `LLM_PROVIDER` (`auto`/`groq`/`ollama`) / `LLM_TIMEOUT` / `LLM_TEMPERATURE` | Chat generation chain | Yes |
| `GROQ_API_KEY` / `GROQ_MODEL` / `GROQ_BASE_URL` | Primary hosted provider (Groq) | Yes (Phase 5) |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Fallback local provider (Ollama) | Yes |
| `RETRIEVAL_TOP_K` / `MIN_RELEVANCE_SCORE` | Retrieval + evidence threshold | Yes |
| `MIN_BM25_SCORE` | HYBRID-only gate for lexical-only chunks | Yes (Phase 4) |
| `MAX_CONTEXT_CHARS` | Context budget | Yes |

## 8. How to run

Backend (from `backend/`; needs a Groq API key for the hosted provider, and/or
Ollama running with `OLLAMA_MODEL` pulled as the local fallback):

```bash
uvicorn app.main:app --reload
```

Check it:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"Adaptive University RAG Assistant","environment":"development"}
```

Ask a grounded question (after ingesting the demo corpus):

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the minimum attendance requirement?"}'
```

Example response:

```json
{
  "answer": "75 percent [1]",
  "sources": [{"document": "attendance_policy.pdf", "page": 1}, ...],
  "enough_evidence": true,
  "retrieval": {"chunks_retrieved": 5, "chunks_used": 5, "top_score": 0.72},
  "provider": "ollama",
  "model": "llama3.1",
  "strategy": "normal",
  "strategy_reason": "Short conceptual question with no exact identifiers that would need lexical matching."
}
```

Every response exposes the selected retrieval strategy and why it was chosen.
A question containing a course code, acronym, or exact policy term (e.g. "What
are the attendance requirements for ENCT 353?") is routed to `"hybrid"`; the
hybrid response additionally fuses BM25 hits into the evidence and keeps
lexical-only chunks only when their BM25 score reaches `MIN_BM25_SCORE`.

The `provider` field reports who generated the answer: `"groq"`,
`"ollama"`, or `"ollama_fallback"` (Groq failed and Ollama answered instead).
With `LLM_PROVIDER=auto` (the default) Groq is tried first and Ollama is the
fallback; if `GROQ_API_KEY` is missing, Groq is skipped entirely and Ollama is
used directly. Responses never expose API keys or provider-internal error
details.

If the corpus does not contain the answer, the endpoint returns the controlled
message "The available documents do not contain enough information to answer
this question." with empty sources — the LLM is not called at all.

Frontend (from `frontend/`):

```bash
npm run dev
```

Then open http://localhost:5173 (or add `-- --port 5174` if that port is
taken; the Vite config proxies `/api` and `/health` to FastAPI on port 8000,
so no second origin reaches the browser in development).

The UI shows, for every answer: the grounded answer with citations, the
sources exactly as returned by the backend (document + page), the selected
retrieval strategy (`NORMAL`/`HYBRID`) with the router's reason, and the
generating provider (`Groq`, `Ollama fallback`, `—` when no LLM ran). A
status line reflects a real `/health` check — never a static "connected"
badge — and request errors are shown inline instead of fake answers.

## 8. How to run tests

Backend tests run against the real FastAPI application via the test client:

```bash
cd backend
pytest
```

The suite uses deterministic fakes for external services; tests against the
real Qdrant engine always run (embedded, in-process), and a test against a
dedicated Qdrant *server* runs only when `QDRANT_URL` points at one. Tests
cover the deterministic router, the BM25 index, hybrid fusion (dedupe,
metadata, RRF ranking), the context gates (semantic and BM25 thresholds), the
RAG orchestrator (grounding, citations, insufficient evidence, strategy
dispatch) and the `/api/chat` endpoint.

> **Model note:** the default `OLLAMA_MODEL` is `llama3.1` (8B). Measured on
> the demo corpus, 3B-class models (e.g. `llama3.2`) degenerate to one-token
> answers with full-size contexts; 8B stays grounded and cited.

Frontend lint:

```bash
cd frontend
npm run lint
```

## Repository layout

```
.
├── backend/
│   ├── app/
│   │   ├── config.py          # typed settings from environment variables
│   │   ├── main.py            # FastAPI app, GET /health, POST /api/chat
│   │   ├── ingestion.py       # PDF extract, clean, chunk, Chunk metadata
│   │   ├── embeddings.py      # EmbeddingClient protocol + local/Ollama providers
│   │   ├── vectorstore.py     # Qdrant collection, upsert, similarity search
│   │   ├── bm25.py            # Okapi BM25 lexical index (HYBRID strategy)
│   │   ├── retriever.py       # NORMAL vector + HYBRID (RRF fusion) strategies
│   │   ├── routing.py         # deterministic query router (NORMAL | HYBRID)
│   │   ├── context.py         # relevance gates, dedupe, budget
│   │   ├── rag.py             # RAG orchestrator: strategy dispatch, citations
│   │   ├── llm.py             # LLMClient protocol + Groq/Ollama + fallback chain
│   │   └── ingest.py          # pipeline orchestration + CLI
│   ├── tests/                 # unit + integration tests
│   ├── .env.example
│   ├── pytest.ini
│   └── requirements.txt
├── frontend/                  # React + Vite chat UI (answer, sources, strategy, provider)
├── data/
│   └── documents/             # demo corpus (regenerate via scripts/)
├── scripts/
│   └── generate_demo_corpus.py
├── .gitignore
└── README.md
```
