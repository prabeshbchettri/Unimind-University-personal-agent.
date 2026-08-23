# University Academic Assistant

An Adaptive RAG-based conversational academic assistant for university
students and staff. Users can ask questions about syllabi, notices, past
question papers, rules, the academic calendar and library books, backed by
retrieval-augmented generation with Llama 3.1 8B via Ollama.

This repository currently contains **Phases 1-13**: project foundation, the
PDF ingestion/OCR pipeline, document classification plus structure/metadata
extraction, semantic chunking with embeddings indexed into Qdrant, the first
complete RAG pipeline (`POST /chat`) with Llama 3.1 8B via Ollama, hybrid
retrieval (dense vectors + BM25 keywords fused with Reciprocal Rank Fusion),
a focused academic knowledge graph in Neo4j (entities + relationships, kept
separate from the vector store), a query analyzer with an adaptive router
that chooses NORMAL / HYBRID / GRAPH per query (Graph RAG included),
syllabus-aware library book recommendation with transparent topic-coverage
scoring, persistent PostgreSQL chat history (sessions + messages with
bounded conversation context), controlled web search, the React frontend
integration, and Phase 13 evaluation/optimization/deployment: automated
evaluation, performance profiling, caching with invalidation, Docker
deployment, environment templates, structured request logging and security
hardening (rate limiting, upload limits, prompt-injection guard).

## Status

- **Phase 1 (complete)** — backend and frontend scaffolding, configuration,
  structured logging, service interfaces, database boundaries, basic tests.
- **Phase 2 (complete)** — PDF ingestion: automatic text-readable vs scanned
  detection, PyMuPDF text extraction, Tesseract OCR, page-level extraction,
  conservative text cleaning, normalized document output, error handling,
  upload endpoint and CLI.
- **Phase 3 (complete)** — document classification (syllabus, notice, past
  question, rules/regulations, academic calendar, library book, unknown),
  deterministic metadata/structure extraction (semester, subject, year, marks,
  questions, topics, chapters, author, etc.), optional LLM extraction with
  strict schema validation, library books kept independent of the curriculum,
  `/documents/analyze` endpoint and `--analyze` CLI flag.
- **Phase 4 (complete)** — semantic chunking that keeps a topic's definition
  and explanation together (with page/section/topic/subtopic provenance),
  embeddings via a dedicated embedding model (separate from the chat LLM),
  indexing into three Qdrant collections (`university_docs`,
  `past_questions`, `library_books`) routed by document type, and vector
  search returning text + score + metadata.
- **Phase 5 (complete)** — first complete vertical slice: `NormalRetriever`
  (embed query -> search Qdrant -> top-K chunks), a `ContextBuilder` that
  assembles a grounded prompt (source metadata, size limits, clear separation
  from the user query) under a grounding system prompt, an LLM service
  abstraction (Llama 3.1 8B via Ollama, with a deterministic stub), and
  `POST /chat` returning `{answer, sources}`. Questions whose answer is not in
  the database are not confidently answered.
- **Phase 6 (complete)** — hybrid retrieval: an in-memory BM25 sparse index
  (kept in sync during indexing) plus the dense vector index, combined by
  `HybridRetriever` using Reciprocal Rank Fusion (rank-based, scale-independent)
  and a rerank pass that blends the RRF score with normalized dense similarity.
  Every result preserves `document_id`, `document_type`, `title`, `page`,
  `subject`, `semester`, `topic`, `text`, `retrieval_score` and
  `retrieval_method`. `RAG_RETRIEVAL_STRATEGY=hybrid` (default) or `normal`;
  `NormalRetriever` still works unchanged.
- **Phase 7 (complete)** — a focused academic knowledge graph in Neo4j:
  `University -> Program -> Semester -> Subject -> Topic/Subtopic -> PastQuestion`.
  Entities are extracted deterministically from the structured documents
  (no LLM, nothing fabricated), validated against an allowed-relationship
  schema, merged with stable keys (first-seen wins, no duplicate nodes), and
  every node/relationship keeps `document_id`, `page`, `source_type` so the
  graph is traceable to the original PDF. Basic queries are available
  (subjects in a semester, topics/subtopics of a subject/topic, past questions
  about a topic, subjects containing a topic) plus `GET /graph/summary`.
  `GRAPH_BACKEND=memory` (default, no server) or `neo4j`. Qdrant is unchanged:
  vectors stay in Qdrant, explicit relationships in the graph; Phase 8
  combines them through the adaptive router (Graph RAG).
- **Phase 8 (complete)** — query analyzer + adaptive router: every chat query
  is classified into exactly one retrieval strategy — `NORMAL` (dense vector),
  `HYBRID` (dense + BM25, used for exact-match queries: regulation/rule
  numbers, subject codes, dates, years, question numbers, marks/percent) or
  `GRAPH` (structural relationship questions: subjects in a semester, topics/
  subtopics of a subject/topic, past questions about a topic, subjects
  containing a topic) — with a structured, explainable `RetrievalPlan`
  `{strategy, reason, filters, graph_parameters, fallback, ...}`. The
  deterministic `RuleBasedQueryAnalyzer` decides first; the LLM classifier is
  consulted only when rules are inconclusive, and uncertainty falls back to
  HYBRID. Graph RAG is live: GRAPH plans query the knowledge graph directly
  (entities as evidence), with dense vector fallback when the graph has
  nothing. `RAG_RETRIEVAL_STRATEGY=auto` (default) enables routing;
  `normal`/`hybrid` keep the legacy fixed behavior. `POST /router/plan` returns
  the plan without retrieving; `/chat` responses include `strategy`. Routing
  quality is measured on a 36-query eval set (10 NORMAL, 10 HYBRID, 10 GRAPH,
  6 WEB) with accuracy, false-routing and fallback metrics
  (`scripts/evaluate_router.py`).
- **Phase 9 (complete)** — syllabus-aware library book recommendation: no
  hard-coded book-to-subject assignments. A user query maps to the syllabus
  via retrieval evidence (cosine floor or a shared meaningful token; common
  abbreviations like `DBMS` are expanded), the required topics are the union
  of the syllabus chunk topics and the knowledge graph's topics for the
  identified subject, each indexed library book's chapters/subtopics are
  compared against them, and books are ranked by a transparent coverage score
  `0.85 * topic_coverage + 0.15 * subtopic_coverage` where `topic_coverage =
  matched required topics / required topics` (semantic match = cosine ≥ 0.4;
  exact normalized names always match). `POST /recommendations` returns the
  ranked books with `score`, `matched_topics`, `missing_topics` and an
  evidence-grounded explanation (`scripts/sample_recommendation.py`).
- **Phase 10 (complete)** — persistent chat history in PostgreSQL: chat
  sessions (`session_id`, `title`, `created_at`, `updated_at`) and messages
  (`message_id`, `session_id`, `role`, `content`, `created_at`) via
  SQLAlchemy. `POST /chat` creates a session when no `session_id` is given
  (title derived from the first message) or continues an existing one
  (unknown ids return 404); the user message and the grounded answer are
  stored. The bounded recent history (most recent `HISTORY_MAX_MESSAGES`
  messages up to `HISTORY_MAX_CHARS` characters) is injected into the
  generation prompt as a `CONVERSATION:` section alongside the retrieved
  context, so follow-up questions are answered with the ongoing conversation
  in view. `GET /chat/sessions` lists sessions (with message counts),
  `GET /chat/sessions/{id}` returns one with its messages,
  `DELETE /chat/sessions/{id}` removes it, and `/history` lists sessions
  too. `DATABASE_BACKEND=memory` (default, SQLite) or `postgres`
  (`DATABASE_URL`). Conversation persistence only — no long-term personal
  memory, personality or preferences.
- **Phase 11 (complete)** — controlled web search for current/external
  questions. The router gains a fourth strategy, `WEB`: queries with
  current/external signals ("latest", "current", "news", "pricing", "today",
  ...) are answered from web results (title, URL, snippet, retrieval time),
  while university-domain questions (attendance, semester, academic
  calendar, regulations, ...) stay on internal documents — web content never
  overrides university sources. Mixed questions (university context + web
  signal) retrieve both, with university documents rendered in `CONTEXT:`
  and web results in a separate `WEB RESULTS:` block so answers cite URLs
  distinctly. `WEB_SEARCH_BACKEND=stub` (default, deterministic offline) or
  `duckduckgo` (live, no API key); a failing or timing-out web search falls
  back to internal retrieval instead of failing the answer. Empty web
  results produce the explicit "insufficient information" answer, never a
  fabrication. `POST /web/search` exposes the provider directly, and
  `/router/plan` + `/chat` report the `WEB` strategy.
- **Phase 12 (complete)** — React + FastAPI integration. The Vite dev server
  proxies `/api/*` to the backend (`http://localhost:8000`, prefix stripped)
  and the backend allows the frontend origin via CORS (`CORS_ORIGINS`,
  default `http://localhost:5173`). The chat UI has a conversation sidebar
  (new chat, switch, delete), the message thread with per-answer sources
  (university: title/page/document type; web: title/URL/retrieval time),
  book recommendation cards (title, author, coverage score, matched/missing
  topics, evidence, explanation), a book-recommendation mode toggle, friendly
  error banners (no stack traces) for backend/model/index/database/web
  failures and timeouts, non-blocking loading states, and a responsive
  layout (sidebar becomes a drawer on small screens). Book results now carry
  the `author` metadata, and an empty index returns a graceful
  "no syllabus evidence" response instead of a 500.
- **Phase 13 (complete)** — evaluation, optimization and deployment:
  - **Evaluation** (`scripts/evaluate.py`, `docs/EVALUATION.md`): automated
    measurement of routing accuracy, per-strategy retrieval metrics
    (recall/precision/MRR on labeled query subsets, plus the empty-response
    rate on other strategies), answer faithfulness/relevance/source
    correctness/hallucination, OCR accuracy, and book-recommendation checks —
    all on a hermetic corpus (deterministic embedder, stub LLM, in-memory
    backends). Three product defects surfaced by the eval were fixed
    (chunker overlap topic mislabelling, resolver subtopic coverage, book
    profile subtopics).
  - **Profiling** (`scripts/profile_performance.py`, `docs/PERFORMANCE.md`):
    per-query latencies for chunking, embedding, indexing, retrieval
    (NORMAL/HYBRID/GRAPH), chat and recommendations. No bottleneck was
    measured, so the pipeline was not modified.
  - **Caching with invalidation**: `EmbeddingCache` (LRU, keyed by model +
    text) and `RetrievalCache` (query results with TTL) are process-local
    and bounded; the retrieval cache is invalidated wholesale by any
    document indexing, so it can never serve stale data.
  - **Docker** (`docker-compose.yml`): backend, frontend (nginx), Qdrant,
    Neo4j and PostgreSQL services; an optional `llm` profile adds a
    CPU-only Ollama (no GPU assumed).
  - **Environment templates** (`backend/.env.development|.test|.production`):
    documented config sets, no secrets.
  - **Logging/monitoring**: every request gets an id (`X-Request-ID`
    response header) that is injected into all log lines; logs carry route,
    status, per-stage latencies, retrieval strategy and cache state — never
    user messages, document contents or retrieved text.
  - **Security** (`docs/SECURITY.md`): per-IP rate limiting (HTTP 429),
    streaming upload size limit (HTTP 413), a prompt-injection guard in the
    system prompt (context/history are data, not instructions), and log
    scrubbing of sensitive fields.

## Repository layout

```
university-academic-assistant/
├── backend/          FastAPI application (Python)
│   ├── app/
│   │   ├── api/          HTTP routers (health, chat, documents, history)
│   │   ├── config/       centralized environment-based settings
│   │   ├── core/         cross-cutting concerns (structured logging)
│   │   ├── models/       SQLAlchemy ORM models (chat sessions/messages, Phase 10)
│   │   ├── schemas/      Pydantic request/response schemas
│   │   ├── services/     domain service interfaces
│   │   ├── repositories/ data-access boundaries (PostgreSQL/Qdrant/Neo4j)
│   │   ├── ingestion/    PDF type analysis, PyMuPDF + OCR extractors,
│   │   │                 text cleaning, normalized document, pipeline
│   │   ├── structure/    document classifier, deterministic + LLM metadata
│   │   │                 extractors, schema validation, structure pipeline
│   │   ├── chunking/     structure-aware semantic chunking (Phase 4)
│   │   ├── embedding/    embedder interface + deterministic/Ollama backends
│   │   ├── retrieval/    collection routing, search models, NormalRetriever
│   │   ├── rag/          context builder + grounding prompt (Phase 5)
│   │   ├── llm/          LLM client interface + Ollama/stub backends (Phase 5)
│   │   ├── graph/        academic knowledge graph: extraction, schema,
│   │   │                 validation + repository (Neo4j / in-memory, Phase 7)
│   │   ├── web/          web search: provider interface, stub + DuckDuckGo
│   │   │                 backends, WebRetriever (Phase 11)
│   │   ├── recommendation/ book recommendation (Phase 9)
│   │   └── utils/        shared helpers (paths)
│   ├── scripts/       CLI tools (ingest_pdf.py, sample_search.py, sample_chat.py,
│   │                  sample_hybrid.py, sample_graph.py, evaluate_router.py,
│   │                  sample_recommendation.py)
│   ├── tests/         pytest suite
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── .env.example
├── frontend/         React (Vite) chat interface
│   ├── src/
│   │   ├── components/  Sidebar, ChatInterface, MessageBubble, SourceCard,
│   │   │                 RecommendationCard (Phase 12)
│   │   ├── services/    fetch-based API client + chat/session/recommendation
│   │   │                 modules (Phase 12)
│   │   ├── utils/       formatting helpers
│   │   ├── App.jsx      session + message state, API wiring
│   │   └── main.jsx     entry point
│   └── vite.config.js   dev server + /api proxy to the backend
├── data/
│   ├── raw/          uploaded PDFs
│   ├── processed/    extracted text/chunks
│   └── test/         sample documents for testing
├── docker-compose.yml  full-stack deployment (backend + frontend + Qdrant +
│                       Neo4j + Postgres, optional CPU-only Ollama)
├── backend/Dockerfile, frontend/Dockerfile + frontend/nginx.conf
├── docs/             architecture, evaluation, performance, security and
│                     deployment notes
└── README.md
```

## Prerequisites

- Python 3.12+
- Node.js 18+
- **Tesseract OCR 5+** (required for scanned PDFs) — e.g. install on Windows
  with `winget install UB-Mannheim.TesseractOCR`, on Debian/Ubuntu with
  `apt install tesseract-ocr`. Auto-detected, or set `TESSERACT_PATH`.
- (Optional — all phases run without external services via in-memory backends;
  use them for production data: PostgreSQL, Qdrant, Neo4j, Ollama)

## Backend setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # add tests
Copy-Item .env.example .env              # customize as needed
```

Run the API (default http://localhost:8000):

```powershell
uvicorn app.main:app --reload
```

Verify:

```powershell
curl.exe http://localhost:8000/health    # -> {"status":"ok"}
```

Interactive API docs: http://localhost:8000/docs

## PDF ingestion

Upload a PDF through the API (auto-detects text-readable vs scanned):

```powershell
curl.exe -X POST -F "file=@syllabus.pdf;type=application/pdf" http://localhost:8000/documents/upload
```

Or via the CLI (prints the normalized document as JSON):

```powershell
cd backend
python scripts/ingest_pdf.py path/to/document.pdf         # full JSON
python scripts/ingest_pdf.py path/to/document.pdf --summary
```

Text-readable PDFs are extracted with PyMuPDF; scanned PDFs are rendered to
images and transcribed with Tesseract. Both paths return the same normalized
structure (`document_id`, `filename`, `page_count`, `extraction_method`,
`pages[].text`, `metadata`).

## Document structure extraction

Classify and extract metadata/structure from an uploaded PDF in one step:

```powershell
curl.exe -X POST -F "file=@syllabus.pdf;type=application/pdf" http://localhost:8000/documents/analyze
```

Or via the CLI:

```powershell
cd backend
python scripts/ingest_pdf.py path/to/document.pdf --analyze
```

Documents are classified as `syllabus`, `notice`, `past_question`,
`rules_regulations`, `academic_calendar`, `library_book` or `unknown`. Metadata
(semester, subject, year, marks, question number, topics, chapters, author,
etc.) is extracted deterministically by default (`STRUCTURE_EXTRACTOR=
deterministic`); set it to `llm` to route through a model instead. LLM output is
always re-validated against the schema and rejected on any mismatch. Library
books are kept independent — they are never assigned a semester, subject or
program. The default classifier threshold (`DOCUMENT_CLASSIFIER_MIN_SCORE`) and
an optional known-university list (`UNIVERSITY_NAMES`) are configurable.

## Chunking, embeddings and Qdrant indexing

Upload a PDF and index it into Qdrant in one call:

```powershell
curl.exe -X POST -F "file=@syllabus.pdf;type=application/pdf" http://localhost:8000/documents/index
```

Or via the CLI (prints the index summary and a sample search):

```powershell
cd backend
python scripts/ingest_pdf.py path/to/document.pdf --index
```

Search the index:

```powershell
curl.exe "http://localhost:8000/search?q=what%20is%20normalization&top_k=5"
```

A self-contained demo (builds sample documents, indexes them, runs searches)
is available without any external services:

```powershell
cd backend
python scripts/sample_search.py
```

The pipeline is:

```
StructuredDocument -> SemanticChunker -> Embedder -> Qdrant
```

- **Chunking** is structure-aware: a topic heading stays with its definition
  and explanation, past-paper questions become their own chunks, and book
  chapters/subtopics are preserved. Every chunk keeps `document_id`, `page`,
  `section`, `topic`, `subtopic` plus document metadata as its Qdrant payload.
- **Collections** are routed by document type: `university_docs` (syllabi,
  notices, rules, calendar, unknown), `past_questions`, `library_books`.
- **Embeddings** come from a dedicated embedding model, separate from the
  chat LLM. Default `EMBEDDER_BACKEND=auto` uses Ollama `/api/embed` with
  `EMBEDDING_MODEL=bge-m3` (`ollama pull bge-m3`) when a base URL is set;
  `EMBEDDER_BACKEND=deterministic` uses a local token-hash fallback that works
  offline and in tests. Qdrant runs from a server (`QDRANT_URL=:memory:` for
  an embedded in-memory client with no server).
- Search results include `text`, `score` and `metadata`; searching across all
  collections merges and re-ranks by score.

See `scripts/sample_search.py` for sample indexed data and search results.

## RAG chat (`POST /chat`)

Ask a question and receive a grounded answer with its sources:

```powershell
curl.exe -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{\"message\": \"Explain normalization.\"}'
```

```json
{
  "answer": "Normalization reduces data redundancy and improves data integrity ...",
  "sources": [
    {
      "text": "Unit 2: Normalization ...",
      "score": 0.548,
      "metadata": { "document_id": "...", "page": 1, "topic": "Normalization",
                    "subject": "Database Management System" },
      "collection": "university_docs"
    }
  ]
}
```

The pipeline is:

```
User query -> NormalRetriever (embed query -> Qdrant top-K)
           -> ContextBuilder (metadata-preserving, size-limited prompt)
           -> LLM (Llama 3.1 8B via Ollama) -> grounded answer + sources
```

- **NormalRetriever** embeds the query (same embedding model as the index),
  searches all three collections and returns the top-K chunks (text, score,
  metadata). An optional relevance floor (`RAG_MIN_SCORE`, default `0.0` =
  disabled) drops weak matches, which is how the suite demonstrates the
  "no evidence" path with the deterministic embedder.
- **ContextBuilder** numbers the sources, preserves `title`, `document_id`,
  `page`, `subject`, `topic`, respects `CONTEXT_MAX_CHARS` /
  `CONTEXT_MAX_SOURCES`, and clearly separates `CONTEXT:` from
  `User Query:`.
- **Grounding rules** are encoded in the system prompt: answer from the
  retrieved context, do not invent university-specific facts, state when the
  sources are insufficient, preserve uncertainty, prefer retrieved information
  over general knowledge, and cite source/page when available.
- **LLM abstraction**: `LLM_BACKEND=auto` uses Ollama
  (`OLLAMA_MODEL=llama3.1:8b`) when a base URL is configured;
  `LLM_BACKEND=stub` returns a deterministic reply (tests/offline). The LLM is
  deliberately separate from the embedding model (`EMBEDDING_MODEL=bge-m3`).

Offline demo of the full slice (builds sample documents, indexes them, answers
the acceptance queries):

```powershell
cd backend
python scripts/sample_chat.py
```

See `docs/architecture.md` for the Phase 5 design.

## Hybrid retrieval (Phase 6)

`RAG_RETRIEVAL_STRATEGY=hybrid` (default) routes every chat question through
both retrieval subsystems:

```
User query -> Dense (Qdrant cosine) + Sparse (BM25 keywords)
           -> Reciprocal Rank Fusion (ranks, not raw scores)
           -> Rerank (RRF score blended with normalized dense similarity)
           -> Top-K context -> Llama -> answer + sources
```

- **Dense** retrieval is the Phase 5 `NormalRetriever` (Qdrant cosine).
- **Sparse** retrieval is an in-memory Okapi BM25 index
  (`app/retrieval/sparse.py`) that is populated alongside the vector index on
  every `POST /documents/index`. It matches exact tokens — codes (`CSIT 325`),
  regulation numbers (`regulation 12`), dates (`2024-05-01`), academic years
  (`2080`), question numbers (`Q5`) and specific phrases.
- **Fusion** uses Reciprocal Rank Fusion (`1/(k + rank)` summed across lists).
  This is deliberate: dense cosine scores and BM25 scores are on incompatible
  scales and must not be added directly; RRF is scale-independent and robust to
  outlier scores. `HYBRID_RRF_K` controls the constant `k`.
- **Rerank** then blends the rank-based RRF score with the min-max normalized
  dense similarity (`HYBRID_RERANK_RRF_WEIGHT` / `HYBRID_RERANK_DENSE_WEIGHT`)
  so the vector similarity acts as a fine-grained tiebreaker without letting
  BM25 magnitude leak into the final score.
- `NormalRetriever` remains fully functional (`RAG_RETRIEVAL_STRATEGY=normal`).

Compare the two strategies on the representative query set:

```powershell
cd backend
python scripts/sample_hybrid.py
```

The test suite (`tests/test_hybrid_retriever.py`) compares dense vs hybrid on
semantic, subject-code, regulation-number, date, academic-year,
question-number and exact-phrase queries, asserting the correct source appears
in the top results for both, and that hybrid promotes weak exact matches.

## Knowledge graph (Phase 7)

The academic knowledge graph captures the explicit relationships between
important academic entities (not chunks or sentences):

```
University -[:HAS_PROGRAM]-> Program -[:HAS_SEMESTER]-> Semester
Semester -[:HAS_SUBJECT]-> Subject
Subject -[:HAS_TOPIC]-> Topic -[:HAS_SUBTOPIC]-> Subtopic
Subject -[:HAS_QUESTION]-> PastQuestion -[:ABOUT]-> Topic
```

Every `POST /documents/index` runs the graph pipeline alongside the vector
pipeline:

```
StructuredDocument -> Entity Extraction (deterministic)
                   -> Relationship Extraction -> Schema Validation -> Graph
```

- **Extraction is deterministic**: university/program/semester/subject come
  from the validated metadata; topics/subtopics from the structure; past
  questions from the extracted questions (with year, marks, question number).
  No LLM is used and nothing is fabricated.
- **Validation**: only the relationships above are allowed into the graph
  (`app/graph/schema.py`); anything else is rejected before it is written.
- **Duplicate handling**: every entity has a stable key (normalized name,
  `sem:5`, `document_id:question_number`); merges are first-seen-wins, so the
  same subject indexed from two documents yields one node.
- **Source traceability**: every node and relationship stores `document_id`,
  `page`, `source_type` (and `title`), so each entity links back to its PDF.
- **Graph + vector stay separate**: Qdrant keeps semantic similarity, Neo4j
  keeps explicit relationships. Both are populated from the same document in
  the same index call. An unreachable graph never breaks vector indexing.
- **Backend**: `GRAPH_BACKEND=memory` (default, in-process, no server — used
  by tests/offline) or `neo4j` (`NEO4J_URI`, `NEO4J_USERNAME`,
  `NEO4J_PASSWORD`). The in-memory backend mirrors the Neo4j Cypher semantics.

Basic queries (service layer, `app/services/graph.py`): subjects in a
semester, topics of a subject, subtopics of a topic, past questions about a
topic, subjects containing a topic, plus the aggregate summary endpoint:

```powershell
curl.exe http://localhost:8000/graph/summary
```

Offline demo (builds a syllabus + past paper, indexes them, prints the summary
and all five query results):

```powershell
cd backend
python scripts/sample_graph.py
```

## Query analyzer and adaptive router (Phase 8)

`RAG_RETRIEVAL_STRATEGY=auto` (default) routes every chat query through the
adaptive router instead of a fixed retriever:

```
User query -> Query Analyzer -> RetrievalPlan {strategy, reason, filters, ...}
           -> NORMAL (dense) | HYBRID (dense + BM25) | GRAPH (knowledge graph)
           -> top-K context -> Llama -> answer + sources + strategy
```

- **Query Analyzer** (`app/router/analyzers.py`): deterministic rules decide
  first (explainable, no model needed). GRAPH rules fire on structural
  relationship questions (subjects in a semester, topics/subtopics of a
  subject/topic, past questions about a topic, subjects containing a topic);
  HYBRID rules fire on exact-match tokens (regulation/rule numbers, subject
  codes like `CSIT 325`, dates, academic years, question numbers, marks,
  percents) and set filters such as `document_type=rules_regulations`;
  everything else is NORMAL. The optional LLM classifier
  (`ROUTER_CLASSIFIER=llm`/`auto`) double-checks uncertain verdicts.
- **RetrievalPlan** is structured and explainable: strategy, reason, optional
  metadata filters (applied post-retrieval, never emptying the results),
  graph parameters (intent + entity), `reranking_required` and `fallback`.
  Inspect it without retrieving via `POST /router/plan`.
- **Graph RAG** (`app/retrieval/graph_retriever.py`): a GRAPH plan runs the
  knowledge-graph query directly — the entities are the evidence (collection
  `knowledge_graph`, method `graph`) — and falls back to dense vector
  retrieval only when the graph has no answer for the intent.
- **Fallback**: when classification is uncertain (e.g. empty query, LLM
  failure), the router falls back to a safe strategy (`ROUTER_FALLBACK_STRATEGY=hybrid`).
- **Eval set**: 30 representative queries (10 NORMAL, 10 HYBRID, 10 GRAPH) in
  `app/router/eval_set.py` measure routing quality:

```powershell
cd backend
python scripts/evaluate_router.py
```

prints routing accuracy / false-routing rate / fallback rate and the plan for
one query of each strategy (currently 100% / 0% / 0% with the rule analyzer).

## Syllabus-aware book recommendation (Phase 9)

`POST /recommendations` answers queries such as "Which book is best for
normalization?", "Recommend a book for DBMS." and "Which library book covers
most of my syllabus?" without any hard-coded book-to-subject assignment:

```
User query -> TopicResolver (syllabus evidence)
           -> required topics (syllabus chunk topics + graph topics)
           -> book profiles (chapter topics / subtopics from the index)
           -> CoverageCalculator (transparent score) -> ranked books
           -> evidence-grounded explanation
```

```powershell
curl.exe -X POST http://localhost:8000/recommendations `
  -H "Content-Type: application/json" `
  -d '{\"query\": \"Which book is best for normalization?\"}'
```

- **TopicResolver** (`app/recommendation/resolver.py`) maps the query to the
  syllabus. A syllabus chunk is evidence when its cosine similarity is at or
  above `RECOMMENDATION_MIN_SCORE` **or** it shares at least one meaningful
  token with the query (this keeps short syllabi evidence even when the crude
  token-hash embedder scores them near zero). Common course abbreviations are
  expanded (`DBMS` -> `Database Management System`). Once a subject is
  identified, all of that subject's syllabus chunks count as evidence and the
  required topics are the union of their `topic` payloads with the knowledge
  graph's topics for the subject (the graph is authoritative for syllabus
  topics). No syllabus evidence -> no recommendation.
- **Coverage score** (`app/recommendation/coverage.py`) is fully transparent
  and documented: `score = topic_weight * topic_coverage + subtopic_weight *
  subtopic_coverage` (defaults `RECOMMENDATION_TOPIC_WEIGHT=0.85`,
  `RECOMMENDATION_SUBTOPIC_WEIGHT=0.15`) where `topic_coverage` is the share
  of required topics covered by the book and `subtopic_coverage` the share of
  required subtopics (0 when the syllabus defines none). A topic matches
  exactly (normalized names) or semantically (embedding cosine at or above
  `RECOMMENDATION_SEMANTIC_THRESHOLD=0.4`). Ties break by more matched topics,
  then title.
- **Response** per book: `book_title`, `author`, `score` (0-1), `matched_topics`,
  `missing_topics` (and subtopic equivalents), plus `subject`, `required_topics`
  and a grounded `explanation`. The LLM may explain the ranking but never
  invents book contents: the score always comes from the retrieved evidence,
  and with the stub LLM the explanation degrades to a deterministic summary.
- Library book profiles come from the index: a book's topics are the distinct
  `topic` payloads of its chunks (chapter titles), subtopics the distinct
  `subtopic` payloads. Chapter headings start their own chunks, so a book's
  topic payloads reflect every chapter, not just the first.

Offline demo (builds a syllabus + three books with strong/partial/poor
coverage and runs all three acceptance queries):

```powershell
cd backend
python scripts/sample_recommendation.py
```

## Chat history (Phase 10)

Conversations are persisted in PostgreSQL: a session owns an ordered list of
messages, and the user message + assistant answer of every `POST /chat` are
stored. This is conversation persistence only — no long-term personal
memory, personality or preferences.

```
User (anonymous; session_id identifies the conversation)
  └── ChatSession (session_id, title, created_at, updated_at)
        ├── Message (message_id, session_id, role, content, created_at)
        └── ...
```

- **Backend**: `DATABASE_BACKEND=memory` (default — in-process SQLite, used
  by tests and offline dev) or `postgres` (PostgreSQL via `DATABASE_URL`,
  psycopg 3). `auto` picks postgres unless the URL is an in-memory sentinel.
  Tables are created automatically at startup; `pip install psycopg[binary]`
  comes from `requirements.txt`.
- **`POST /chat`** — with no `session_id` a new session is created (its title
  is derived from the first user message, truncated); with a known
  `session_id` the conversation continues; an unknown id returns `404`.
- **Context**: when continuing a session, the most recent
  `HISTORY_MAX_MESSAGES` messages that fit in `HISTORY_MAX_CHARS` characters
  are rendered as a `CONVERSATION:` section between the retrieved
  `CONTEXT:` and the `User Query:` — the entire history is never sent blindly
  when it exceeds the limit.
- **Endpoints**: `GET /chat/sessions` (most recent first, with
  `message_count`), `GET /chat/sessions/{session_id}` (with all messages),
  `DELETE /chat/sessions/{session_id}` (cascades messages), and
  `GET /history` (session list, kept from the Phase 1 placeholder).

```powershell
curl.exe -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{\"message\": \"Explain normalization.\"}'
# -> { "answer": "...", "session_id": "776c...", ... }

curl.exe -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{\"message\": \"And transactions?\", \"session_id\": \"776c...\"}'

curl.exe http://localhost:8000/chat/sessions/776c...
curl.exe -X DELETE http://localhost:8000/chat/sessions/776c...   # 204
```

## Web search (Phase 11)

Questions that ask for **current or external information** — "What is the
latest Python version?", "What happened in today's AI news?", "What is the
current OpenAI API pricing?" — are answered from web search instead of (or
alongside) the university knowledge base. University questions keep their
priority: "What is the attendance requirement?", "What subjects are in
semester 5?", "When does the academic calendar start?" stay on the internal
documents, and web content is never allowed to override university documents
for university-specific facts.

```
User query
  -> RuleBasedQueryAnalyzer
       GRAPH / HYBRID rules first (university priority)
       -> WEB signals (latest, current, news, pricing, today, ...)
            university context + signal  -> WEB (mixed: internal + web)
            signal only                  -> WEB (web only)
       -> else NORMAL
  -> WebRetriever -> provider (stub | duckduckgo)
       timeout / failure -> fall back to internal retrieval (never an error)
  -> ContextBuilder: CONTEXT: (university) ... WEB RESULTS: (title + URL +
     retrieval time + snippet) ... User Query ... Answer
```

- **Backend**: `WEB_SEARCH_BACKEND=stub` (default — deterministic canned
  results, offline) or `duckduckgo` (live search over the public DuckDuckGo
  HTML endpoint, no API key, `WEB_SEARCH_TIMEOUT` per request).
- **Attribution**: every web result preserves `title`, `url`, `snippet` and
  `retrieved_at`; the prompt renders them as `[Web N] title (url), retrieved
  <time>` and the API exposes them on each source (`kind: "web"`, `url`,
  `retrieved_at`).
- **Safety**: web content is treated as untrusted supplementary evidence.
  The grounding prompt requires URL citation, says web results are not
  automatically trustworthy, and keeps university documents authoritative
  for university facts. Empty web results produce the explicit
  "insufficient information" answer — never a fabricated reply.
- **Failure handling**: a web provider that errors or exceeds
  `WEB_SEARCH_TIMEOUT` is caught by the router (`plan.fallback = true`),
  which falls back to internal retrieval; the answer always completes.
- **Endpoints**: `POST /web/search` (raw provider results for verification),
  `POST /router/plan` (reports `strategy: "WEB"` and
  `web_parameters.mixed`), and `POST /chat` (returns `strategy: "WEB"` with
  web sources marked `kind: "web"`).

```powershell
curl.exe -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{\"message\": \"What is the latest Python version?\"}'
# -> { "strategy": "WEB", "sources": [{ "kind": "web", "url":
#     "https://www.python.org/downloads/", "retrieved_at": "..." }], ... }
```

## Frontend setup

```powershell
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Start the backend first (see "Backend setup"); Vite proxies every `/api/*`
request to `http://localhost:8000` with the `/api` prefix stripped (override
with `VITE_BACKEND_URL`). Requests from the browser also need the backend
CORS origins to include the frontend origin (default already does). For a
production build set `VITE_API_BASE` to the absolute backend URL
(e.g. `http://localhost:8000`) and make sure `CORS_ORIGINS` includes the
deployed origin.

The UI is a ChatGPT-style assistant: a conversation sidebar (new chat,
switch/delete sessions), the message thread, sources under each answer
(university documents and web results with their URLs), book recommendation
cards, a "Book recommendations" toggle in the composer, typing and error
states, and a responsive layout (the sidebar becomes a drawer on small
screens). All backend failures surface as friendly messages — never stack
traces.

```powershell
cd frontend
npm run build        # production bundle in frontend/dist
```

## Phase 13: evaluation, optimization and deployment

See the dedicated documents for the full details:

- `docs/EVALUATION.md` — automated evaluation (hermetic corpus + metrics),
  with the three product defects found and fixed.
- `docs/PERFORMANCE.md` — per-query profiling results.
- `docs/SECURITY.md` — security measures and how to configure them.
- `docs/DEPLOYMENT.md` — Docker and production deployment guide.

Quick overview:

- **Run the automated evaluation** (hermetic, no services needed):

  ```powershell
  cd backend
  $env:QDRANT_URL=":memory:"; $env:EMBEDDER_BACKEND="deterministic"; `
  $env:LLM_BACKEND="stub"; $env:ROUTER_CLASSIFIER="rules"; `
  $env:GRAPH_BACKEND="memory"; $env:DATABASE_BACKEND="memory"
  python scripts/evaluate.py
  ```

- **Profile per-query performance** (same hermetic environment):

  ```powershell
  python scripts/profile_performance.py
  ```

- **Caching**: embeddings are memoized per `(model, text)` (LRU,
  `EMBEDDING_CACHE_MAX_ENTRIES`) and retrieval results per query with a TTL
  (`RETRIEVAL_CACHE_MAX_ENTRIES`, `RETRIEVAL_CACHE_TTL_SECONDS`). The
  retrieval cache is invalidated by every document index operation, so
  cached results can never go stale after an ingestion.
- **Docker**: `docker compose up -d` starts the full stack with hermetic
  backends; add `--profile llm` for a CPU-only local Ollama and set
  `LLM_BACKEND=ollama EMBEDDER_BACKEND=ollama` for real model quality.
- **Environment templates**: copy `backend/.env.development` /
  `.env.production` to `.env` for the matching environment; test config lives
  in `backend/.env.test`.
- **Logging**: every response carries an `X-Request-ID` header; all log lines
  of that request share the id, with route, status, strategy, cache state and
  per-stage latencies (`retrieval_ms`, `llm_ms`) — never content.
- **Security**: rate limiting (429, `RATE_LIMIT_ENABLED`,
  `RATE_LIMIT_PER_MINUTE`), upload cap (413, `MAX_UPLOAD_BYTES`,
  default 20 MB), prompt-injection guard in the system prompt.

## Tests

```powershell
cd backend
pytest -q
```

The suite covers health/config/startup (Phase 1), cleaning, analyzer, error
handling, the PyMuPDF pipeline, the OCR backend (mocked), and a scanned-PDF OCR
end-to-end test that skips automatically when Tesseract is not installed
(Phase 2). Phase 3 adds document classification, per-type deterministic
extraction, LLM output parsing/validation (with fake generators), the
ingestion-to-structure pipeline, and the `/documents/analyze` endpoint.
Phase 4 adds semantic chunking, embedding backends, the in-memory Qdrant
repository, the indexing/search service and the `/documents/index` +
`/search` endpoints (all hermetic — no external services required).
Phase 5 adds the LLM layer (Ollama + stub), the `NormalRetriever`, the
context builder (prompt construction and grounding rules), the end-to-end
`ChatService` and `POST /chat` (including the "answer not in database" case),
all hermetic with the stub LLM and deterministic embedder.
Phase 6 adds the in-memory BM25 sparse index, the `HybridRetriever` (dense +
sparse + RRF fusion + rerank), and dense-vs-hybrid comparison tests across
semantic and exact-match query types, all hermetic.
Phase 7 adds the academic knowledge graph: deterministic entity/relationship
extraction, schema validation, duplicate handling, source traceability, the
in-memory graph repository (mirroring Neo4j semantics), the five basic graph
queries, vector+graph integration during indexing, and graceful degradation
when the graph is unreachable — all hermetic (the Neo4j driver path is tested
without a server).
Phase 8 adds the query analyzer and adaptive router: rule-based + optional LLM
classification into NORMAL/HYBRID/GRAPH, structured explainable plans, filters
applied post-retrieval, the GraphRetriever (graph evidence + dense fallback),
the routing eval set and metrics, the `/router/plan` endpoint and the
`strategy` field on `/chat` — all hermetic with the deterministic embedder,
stub LLM and in-memory graph backend.
Phase 9 adds syllabus-aware book recommendation: the topic resolver (syllabus
evidence by cosine or shared token, alias expansion, graph merge), the
transparent coverage formula (topic/subtopic weights), book profiles from
index payloads, the `POST /recommendations` endpoint and the
evidence-grounded explanation path — all hermetic.
Phase 10 adds chat history persistence: the SQLAlchemy session/message
models, the `ChatHistoryRepository` (SQLite memory backend in tests, the
same code path as PostgreSQL), session-aware `POST /chat` (create/continue,
unknown session 404), bounded `CONVERSATION:` history injection into the
generation prompt, the `/chat/sessions` CRUD endpoints and the `/history`
list — all hermetic.
Phase 11 adds web search: the `WEB` router strategy with university-priority
rules and mixed-question detection, the web provider interface (deterministic
stub + DuckDuckGo backend with hard timeout), the `WebRetriever` with
graceful failure/timeout fallback, the `WEB RESULTS:` prompt block with
URL/retrieval-time attribution, the safety-grounded system prompt, the
36-query eval set (now including WEB), and the `/web/search` endpoint — all
hermetic with the stub provider and a canned-HTML parser test.
Phase 12 adds the React frontend integration: CORS middleware (settings
`CORS_ORIGINS`), the `/api` dev proxy with prefix rewrite, the book `author`
metadata flow (chunk → profile → coverage → API), graceful handling of an
empty index by the recommendation resolver, and API-level guarantees the UI
relies on (session create/continue/delete, web + university sources) — all
hermetic; the frontend itself is covered by a production build plus the
backend suite behind the `/api` proxy.
Phase 13 adds caching (embedding LRU + retrieval cache with invalidation),
structured request logging (request ids, per-stage latencies, no content),
and security tests (rate limiting 429, upload 413, injection guard,
`X-Request-ID`) — all hermetic (`tests/test_caching.py`,
`tests/test_security.py`).

## Configuration

All configuration is read from environment variables (see
`backend/.env.example`). Defaults allow the app to boot with no external
services. Secrets must never be committed; keep them in `.env`.
Phase 12 adds `CORS_ORIGINS` (comma-separated browser origins allowed to
call the API; the React dev server origin is allowed by default).

## Known limitations

- `GET /documents` returns `501 Not Implemented`; a document listing repository
  is still pending.
- OCR quality depends on Tesseract and source image quality (see `docs/`).
- Structure extraction is heuristic; unusual layouts may yield partial
  metadata. The LLM extractor requires the Ollama LLM.
- The `deterministic` embedder is for tests/offline dev only; production
  retrieval quality requires a real embedding model (Ollama `bge-m3`). Its
  inflated cosine similarities also mean `RAG_MIN_SCORE` should stay `0.0`
  (disabled) in production and the grounding prompt is the primary guard
  against confident answers with no evidence.
- The BM25 index and the in-memory graph backend are rebuilt per process (like
  the embedded Qdrant client); they are not persisted across restarts.
- The knowledge graph covers the focused academic model (university, program,
  semester, subject, topics/subtopics, past questions). Graph queries that
  need vector context beyond the entities (e.g. full past-paper text) use the
  dense fallback.
- Rule-based routing covers the academic query patterns above; exotic
  phrasings fall back to HYBRID (safe, not wrong). LLM classification
  (`ROUTER_CLASSIFIER=llm`/`auto`) refines this when a model is available.
- Book coverage uses the extracted chapter/subtopic structure; books whose
  text yields no chapter headings get no profile. Semantic topic matching
  needs a real embedding model (`deterministic` is token-based and weak).
- Chat history is conversation persistence: no authentication (anonymous
  sessions), no user profiles, no long-term memory. `DATABASE_BACKEND=memory`
  does not persist across restarts (in-process SQLite); use `postgres`.
- Web search uses external content that is not automatically trustworthy and
  can change over time; answers must cite URLs, and the grounding prompt
  keeps university documents authoritative for university facts. The stub
  backend returns only canned results; live search requires
  `WEB_SEARCH_BACKEND=duckduckgo` (public HTML endpoint, results may be
  limited) and network access.
- The frontend has no automated browser tests (the backend suite plus the
  Vite production build guard the integration). CORS only allows the
  configured `CORS_ORIGINS`; the dev proxy assumes the backend is on
  `http://localhost:8000` (`VITE_BACKEND_URL` overrides). Recommendations
  display `author` only when the book's extracted metadata contains one
  (plain-text PDFs usually do not).
- Phase 13: the caches and the rate limiter are process-local (fine for one
  uvicorn worker; scale out horizontally with a shared limiter/cache in
  front). The evaluation uses the deterministic embedder and stub LLM —
  numbers are regression baselines, not production-quality claims. Docker
  images need network access at build time (`pip install`, `npm ci`).

## Roadmap

| Phase | Scope |
|-------|-------|
| 1 | Project foundation and architecture (complete) |
| 2 | PDF ingestion and OCR (complete) |
| 3 | Document structure and metadata extraction (complete) |
| 4 | Chunking, embeddings and Qdrant (complete) |
| 5 | Basic Normal RAG (complete) |
| 6 | Hybrid RAG (BM25) (complete) |
| 7 | Neo4j academic knowledge graph (complete) |
| 8 | Query analyzer and adaptive router (complete) |
| 9 | Library book recommendation (complete) |
| 10 | PostgreSQL chat history (complete) |
| 11 | Web search (complete) |
| 12 | React + FastAPI integration (complete) |
| 13 | Evaluation, optimization and deployment (complete) |
