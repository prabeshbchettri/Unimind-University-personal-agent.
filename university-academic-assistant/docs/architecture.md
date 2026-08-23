# Architecture Notes

## System context

```
USER -> React UI -> FastAPI -> Query Analyzer/Router
                                  -> Normal RAG (Qdrant)
                                  -> Hybrid RAG (Qdrant + BM25)
                                  -> Graph RAG (Neo4j + Qdrant)
                                  -> Web Search (external)
       Context -> Llama 3.1 8B (Ollama) -> Answer
       PostgreSQL stores chat history
```

## Backend module boundaries

| Module          | Responsibility                                        | Phase |
|-----------------|-------------------------------------------------------|-------|
| `api/`          | HTTP surface; thin adapters over services             | 1     |
| `config/`       | Environment-driven settings singleton                 | 1     |
| `core/`         | Logging, middleware, error handling                   | 1     |
| `services/`     | Orchestration logic per domain                        | 1     |
| `repositories/` | Data access for PostgreSQL / Qdrant / Neo4j           | 1     |
| `ingestion/`    | PDF type analysis, PyMuPDF + OCR, cleaning, pipeline  | 2     |
| `structure/`    | document classifier, extractors, validation, pipeline| 3     |
| `chunking/`     | structure-aware semantic chunking                     | 4     |
| `embedding/`    | embedder interface + deterministic/Ollama backends    | 4     |
| `retrieval/`    | collection routing/models (4); NormalRetriever (5); BM25 sparse + HybridRetriever (6); graph-queries interface (7) | 4   |
| `rag/`          | context builder + grounding system prompt                    | 5     |
| `llm/`          | LLM client interface + Ollama/stub backends, factory         | 5     |
| `graph/`        | knowledge graph: models, schema validation, deterministic extractor | 7 |
| `recommendation/` | evidence-based book recommendation                  | 9     |
| `web/`          | web search: provider interface, stub + DuckDuckGo, WebRetriever | 11 |
| `models/`       | SQLAlchemy ORM models (chat history etc.)             | 10    |

## Ingestion pipeline (Phase 2)

```
PDF
 │  validate (magic bytes) + open (pymupdf)
 ▼
PDFAnalyzer  ── probe text per page, measure quality ──► decide
   │  text_ratio >= threshold                       │ < threshold
   ▼                                                ▼
PyMuPDFExtractor (per-page text)         OCRPDFExtractor (render page -> image -> OCR)
   └───────────────┬───────────────────────────────┘
                   ▼
           TextCleaner (conservative)
                   ▼
            NormalizedDocument
   document_id / filename / page_count / extraction_method / pages / metadata
```

- Detection is configurable (`PDF_ANALYZER_*`) and never relies on filename.
- The OCR backend is an interface (`OCRBackend`); `TesseractOCRBackend` is the
  Phase 2 implementation and can be swapped for PaddleOCR later.
- Per-page failures are captured in `metadata.page_errors` and never abort the
  document.
- Both pipelines emit the identical `NormalizedDocument` structure so
  downstream phases do not care where the text came from.

## Structure extraction (Phase 3)

```
NormalizedDocument (Phase 2 output)
   │
   ▼
DocumentClassifier ── weighted keyword/regex feature scoring ──► document_type
   │   syllabus | notice | past_question | rules_regulations
   │   | academic_calendar | library_book | unknown  (score >= min, else unknown)
   ▼
MetadataExtractor (interface)
   ├── DeterministicMetadataExtractor (default)  ── heuristics.py regex helpers
   └── LLMMetadataExtractor (optional)           ── prompt -> parse_llm_json
   │                                                  -> validate_llm_metadata
   │                                                  (LLMExtractionError on bad JSON)
   ▼
StructuredDocument
   metadata (document_type, title, university, program, semester, subject,
            subject_code, academic_year, year, question_number, marks, topic,
            subtopic, author)
   structure (topics, subtopics, questions, chapters)
   pages (embedded from NormalizedDocument)
```

- Classification scores each type with overlapping features (e.g. `Q5.` +
  `Marks` → past_question, `Unit`/`Module` + `Credit` → syllabus, `Chapter` +
  `ISBN` → library_book) and never reads the filename.
- `document_classifier_min_score` (default 3) gates acceptance; anything below
  becomes `unknown`, so confident-but-wrong guesses are avoided.
- Extraction is deterministic by default; no values are fabricated — missing
  fields stay `null`. Every extractor implements the same `MetadataExtractor`
  interface, so the LLM path can be enabled with `STRUCTURE_EXTRACTOR=llm`.
- The LLM extractor asks the model to return strict JSON (schema included in the
  prompt). Output is re-parsed and re-validated with Pydantic; any invalid or
  hallucinated JSON raises `LLMExtractionError` and is never persisted.
- Library books are classified and extracted independently: they never receive a
  university, program, semester or subject, and instead keep title, author,
  chapters and subtopics.
- Phase 2 ingestion is unchanged; `StructurePipeline.process()` consumes its
  output. `POST /documents/analyze` runs ingestion + structure in one request.

## Chunking, embeddings and Qdrant (Phase 4)

```
StructuredDocument (Phase 3 output)
   │
   ▼
SemanticChunker ── structure-aware blocks, no mid-paragraph splits
   │   past_question -> per-question blocks
   │   library_book  -> chapter / subtopic blocks
   │   syllabus      -> unit / module / topic blocks
   │   else          -> paragraph blocks
   │   blocks grouped into chunks (max_chars) with block-level overlap
   ▼
DocumentChunk  (chunk_id, document_id, page, section, topic, subtopic,
                semester, subject, subject_code, ..., text)
   │
   ▼
Embedder  (dedicated embedding model, separate from the chat LLM)
   ├── OllamaEmbedder (BGE via /api/embed)   ── EMBEDDER_BACKEND=ollama
   └── DeterministicEmbedder (offline/tests) ── EMBEDDER_BACKEND=deterministic
   ▼
QdrantRepository  (in-memory or remote client, COSINE distance)
   │  collection_for(document_type):
   │    university_docs <- syllabus/notice/rules/calendar/unknown
   │    past_questions  <- past_question
   │    library_books   <- library_book
   ▼
VectorIndexingService.index()  ->  upsert points (stable UUID per chunk)
VectorIndexingService.search(query, collection, top_k) -> text + score + metadata
```

- Chunking never splits a paragraph arbitrarily: topic headings stay with their
  definition/explanation, and only an oversized block is ever split (at sentence
  boundaries). A trailing overlap carries context into the next chunk.
- Qdrant holds **only** vector retrieval data (chunks + payload). Chat history
  belongs to PostgreSQL (Phase 10) and knowledge-graph relationships to Neo4j
  (Phase 7) — they are never stored in Qdrant.
- Embeddings are produced by a model that is deliberately distinct from the
  answer-generation LLM (`EMBEDDING_MODEL` vs `OLLAMA_MODEL`).
- `QDRANT_URL=:memory:` selects an embedded in-memory client (no server) for
  tests and offline development; set a real URL for production.
- `POST /documents/index` runs ingestion -> structure -> chunk -> embed -> upsert
  in one request; `GET /search?q=...` returns `text`, `score`, `metadata`.

## Basic Normal RAG (Phase 5)

```
User Query
   │
   ▼
NormalRetriever  ── embed query (same embedding model as the index)
   │  search all collections (university_docs, past_questions, library_books)
   │  merge + rank by cosine score; optional min_score floor (rag_min_score)
   ▼
list[SearchResult]  (text, score, metadata, collection)
   │
   ▼
ContextBuilder.build(query, chunks)  ->  (system_prompt, prompt)
   │  numbers sources ([Source 1]...) and preserves title/document/page/subject/topic
   │  respects context_max_chars / context_max_sources (whole sources kept)
   │  separates CONTEXT: from "User Query:" (no query leakage into context)
   │  empty evidence -> explicit "no retrieved context" note
   ▼
LLMClient.complete(system, prompt)   (Llama 3.1 8B via Ollama, temp 0.2)
   ├── OllamaLLMClient  ── /api/generate, model = OLLAMA_MODEL (LLM_BACKEND=ollama)
   └── StubLLMClient    ── deterministic reply (LLM_BACKEND=stub, tests/offline)
   ▼
ChatResult(answer, sources)
   ▼
POST /chat  ->  {"answer": ..., "sources": [{"text","score","metadata","collection"}]}
```

- **Retrieval** (`NormalRetriever`) is the only Phase 5 strategy; there is no
  query router yet, so all collections are searched and merged by score. Hybrid
  (BM25) and graph retrieval plug in behind the same interface later.
- **Grounding system prompt** instructs the model to: answer from the supplied
  context only; not invent university-specific facts; explicitly state when the
  sources are insufficient; preserve uncertainty; prefer retrieved university
  information over general knowledge; mention source/page when available.
- **Relevance floor**: `RAG_MIN_SCORE` (default `0.0` = disabled) drops chunks
  below a similarity threshold. It is the mechanism the deterministic test
  suite uses to demonstrate the "no evidence" path (zero-overlap queries score
  `0.0` with the deterministic embedder). Production relies primarily on the
  grounding prompt, since real cosine scores are not as cleanly separated.
- **Anti-hallucination acceptance**: "Explain normalization.", "What is DBMS?",
  "Explain this syllabus topic." are answered from the index; a question absent
  from the database returns no usable sources and an explicit
  "insufficient information" answer rather than a confident fabrication.
- The LLM used for generation is separate from the embedding model; either can
  be swapped via `LLM_BACKEND` / `EMBEDDER_BACKEND`.

## Hybrid retrieval (Phase 6)

```
User Query
   │
   ├──────────────────────────────┬─────────────────────────────┐
   ▼                             ▼                             │
Dense (vector)              Sparse (BM25)                     │
NormalRetriever ─────────>   BM25Index                        │
Qdrant cosine                 in-memory keyword index         │
   │  (text+score+metadata)     │  (text+score+metadata)      │
   └───────────────┬────────────┘                             │
                   ▼                                          │
          fuse_results (RRF)                                  │
           score = Σ 1/(k + rank)   [rank-based, scale-free]  │
                   │                                          │
                   ▼                                          │
          rerank_results                                      │
   final = rrf_weight*rrf_norm + dense_weight*dense_norm      │
                   │                                          │
                   ▼                                          │
          Top-K SearchResult (method="hybrid") ───────────────┘
                   │
                   ▼
          ContextBuilder -> Llama -> Answer
```

- **BM25Index** (`app/retrieval/sparse.py`) — in-memory Okapi BM25
  (k1=1.5, b=0.75). It is populated by `VectorIndexingService.index()` on every
  `POST /documents/index`, keeping the keyword index in sync with Qdrant.
  Tokenization keeps alphanumeric tokens (codes, numbers, years, phrases) and
  drops stopwords, so exact terms like `CSIT 325`, `regulation 12`, `2080`,
  `Q5` and `2024-05-01` match directly.
- **HybridRetriever** (`app/retrieval/hybrid_retriever.py`) exposes
  `search_dense`, `search_sparse`, `fuse_results` and `rerank_results`, then
  `retrieve` orchestrates dense -> sparse -> fuse -> rerank. It is not coupled
  to the API layer; `ChatService` depends on the same `retrieve(query, top_k)`
  interface as `NormalRetriever`.
- **Fusion — Reciprocal Rank Fusion.** Dense cosine (≈[-1,1]) and BM25
  (unbounded, corpus-dependent) scores are not on a comparable scale and must
  not be summed raw. RRF uses only each result's *rank* in each list
  (`1/(k+rank)`, k=60), which is scale-independent, requires no calibration,
  and is robust to outlier scores from either subsystem.
- **Rerank.** RRF is rank-only, so the rerank pass blends the RRF score (rank
  signal) with the min-max normalized dense similarity (quality/tiebreaker
  signal): `final = 0.7*rrf_norm + 0.3*dense_norm`. BM25 magnitude never
  enters the final score.
- **Metadata** is preserved end-to-end: every result carries `document_id`,
  `document_type`, `title`, `page`, `subject`, `semester`, `topic` (in
  `metadata`), plus `text`, `score` (retrieval score) and `method`
  (`dense` / `sparse` / `hybrid`).
- **Strategy switch.** `RAG_RETRIEVAL_STRATEGY=hybrid` (default) or `normal`.
  `NormalRetriever` remains available and unchanged, so dense-only retrieval
  keeps working.
- The BM25 index is in-memory (like the embedded Qdrant client) and rebuilt on
  process start; persistence is left to a later phase.

## Knowledge graph (Phase 7)

```
POST /documents/index
   │
   ▼
StructuredDocument ─────────────► SemanticChunker -> Qdrant (vectors, unchanged)
   │
   ▼
GraphExtractor (deterministic, app/graph/extractor.py)
   │  University/Program/Semester/Subject <- metadata
   │  Topic/Subtopic                       <- structure (+ metadata pair)
   │  PastQuestion (year, marks, number)   <- structure.questions
   ▼
GraphExtraction (nodes + relationships, stable keys)
   │
   ▼
Validation (app/graph/schema.py)
   │  only the allowed relationship triples pass; everything else is dropped
   ▼
GraphRepository.merge_nodes / merge_relationships (first-seen wins)
   ├── InMemoryGraphRepository  ── GRAPH_BACKEND=memory  (tests / offline)
   └── Neo4jGraphRepository     ── GRAPH_BACKEND=neo4j   (MERGE on stable keys)
   ▼
Basic queries (subjects in semester, topics of subject, subtopics of topic,
              past questions about topic, subjects containing topic)
   ▼
GET /graph/summary  (status, backend, per-entity counts)
```

Graph model — a focused academic model, never chunks or sentences:

```
University -[:HAS_PROGRAM]-> Program -[:HAS_SEMESTER]-> Semester
Semester -[:HAS_SUBJECT]-> Subject
Subject -[:HAS_TOPIC]-> Topic -[:HAS_SUBTOPIC]-> Subtopic
Subject -[:HAS_QUESTION]-> PastQuestion -[:ABOUT]-> Topic
```

- **Deterministic extraction, no fabrication.** Entities come only from the
  validated Phase 3 metadata/structure. Syllabus topics are linked to the
  subject (`HAS_TOPIC`); past-paper topics are *not* linked to the subject (a
  paper must not pollute the subject's topic list) — they exist as nodes so
  `PastQuestion -[:ABOUT]-> Topic` can reference them. If an LLM extractor is
  added later, `validate_relationships` gates its output identically.
- **Schema validation.** `ALLOWED_RELATIONSHIPS` in `app/graph/schema.py`
  defines the seven supported triples. `KnowledgeGraphService.index()` splits
  extracted relationships into valid/invalid and only writes the valid ones,
  so unsupported relationships can never enter the graph.
- **Stable keys and duplicate avoidance.** `node_key()` produces normalized
  keys: lowercase names for University/Program/Subject/Topic/Subtopic,
  `sem:5` for Semester, `document_id:question_number` for PastQuestion.
  Merging is `MERGE ... ON CREATE` (first-seen wins) in both backends, so the
  same subject arriving from two documents yields exactly one node.
- **Source traceability.** Every node and relationship stores `document_id`,
  `page`, `source_type` and `title`; pages are resolved by scanning the
  document pages for the entity text, so the graph links back to the PDF.
- **Graph + vector stay separate.** Qdrant remains the only semantic-similarity
  store; Neo4j holds only explicit relationships. `VectorIndexingService.index`
  feeds both from the same document. If the graph is unreachable
  (`GraphUnavailableError`), vector indexing still succeeds and the result
  reports `graph.status = "unavailable"` — vector functionality is unchanged.
- **Backends.** `GRAPH_BACKEND=memory` (default) uses `InMemoryGraphRepository`,
  which mirrors the Neo4j merge semantics and query results, so the entire
  suite is hermetic. `GRAPH_BACKEND=neo4j` uses the `neo4j` driver with the
  same Cypher-equivalent queries (MERGE by key, MATCH-based reads). The driver
  is lazy: an unreachable Neo4j surfaces as `ping() == False` /
  `GraphUnavailableError`, never a boot failure.
- **Integration point for later phases.** The graph queries live on
  `GraphRepository` and are delegated by `KnowledgeGraphService`; a future
  GraphRAG router can query the graph (exact answers: "subjects in semester
  5") and combine them with hybrid vector context through the same
  `SearchResult` shape used by `NormalRetriever`/`HybridRetriever`. This is
  implemented in Phase 8 (see below).

## Query analyzer and adaptive router (Phase 8)

The router makes retrieval adaptive: instead of a fixed retriever, every chat
query is classified and executed with the best strategy.

```
User query
  -> RuleBasedQueryAnalyzer (deterministic rules, explainable)
       |-> GRAPH  (structural relationships, e.g. "subjects in semester 5")
       |-> HYBRID (exact tokens: regulation 12, CSIT 325, 2080, Q5, 10 marks)
       |-> NORMAL (semantic/conceptual, confidence 0.7 = default)
       |-> unknown -> LLMQueryAnalyzer (optional) -> still unknown -> fallback
  -> RetrievalPlan {strategy, query, reason, filters, top_k, graph_parameters,
                    reranking_required, fallback}
  -> strategy retrieval -> SearchResults -> ContextBuilder -> Llama -> answer
```

- **Rule-first, LLM-second.** `RuleBasedQueryAnalyzer` is deterministic and
  dependency-free: GRAPH rules fire on structural intents (subjects in a
  semester, topics/subtopics of a subject/topic, past questions about a
  topic, subjects containing a topic — the `GRAPH_INTENTS` in
  `app/router/plan.py`); HYBRID rules fire on exact-match tokens
  (regulation/rule number, subject code, date, year, question number, marks,
  percent). Entity captures are guarded (`_clean_entity`) so a bare number or
  `Q5` can never be treated as an entity name ("What is question 5 about?"
  is HYBRID, not GRAPH). `LLMQueryAnalyzer` is consulted only when rules are
  inconclusive (`ROUTER_CLASSIFIER=auto`, default) or to double-check
  low-confidence verdicts (`=llm`); its JSON output is validated and failures
  are treated as unknown, never as a strategy.
- **RetrievalPlan is data, not code.** The router returns a structured,
  explainable plan: strategy, reason, optional metadata filters (e.g.
  `document_type=rules_regulations` for regulation queries), graph parameters
  (intent + entity), `reranking_required` and `fallback`. `POST /router/plan`
  exposes it without retrieving; `/chat` echoes the selected strategy in
  `response.strategy`.
- **Filters applied post-retrieval, never emptying results.** Plan filters
  are applied to the retrieved results; if the filter would empty the list the
  unfiltered results are kept (over-filtering is worse than none).
- **Graph RAG.** `GraphRetriever` (`app/retrieval/graph_retriever.py`) is the
  first Graph RAG flow: a GRAPH plan maps its intent 1:1 to a
  `KnowledgeGraphService` query, converts the returned entities into
  `SearchResult`s (collection `knowledge_graph`, method `graph`, human-readable
  evidence text with source/page) and — only when the graph has no answer —
  falls back to the dense retriever for vector context.
- **Fallback on uncertainty.** Empty queries, LLM failures and unconverged
  classification route to `ROUTER_FALLBACK_STRATEGY` (default `hybrid`) with
  `plan.fallback = True`, so chat always answers.
- **Drop-in interface.** `AdaptiveRouter.retrieve(query, top_k)` matches the
  retriever interface, so `ChatService` is unchanged. `RAG_RETRIEVAL_STRATEGY`
  accepts `auto` (router, default), `normal` or `hybrid` (fixed legacy
  behavior).
- **Eval set and metrics.** `app/router/eval_set.py` holds 30 representative
  queries (10 NORMAL, 10 HYBRID, 10 GRAPH) and `evaluate_router()` computes
  routing accuracy, false-routing rate and fallback rate.
  `scripts/evaluate_router.py` runs it (currently 100% / 0% / 0% with the rule
  analyzer) and prints one example plan per strategy.
- **Import discipline.** The router package imports only lightweight modules;
  `app/router/__init__.py` stays minimal to avoid import cycles, and
  `GraphRetriever` imports `SearchResult` from `app.retrieval.models` (never
  the package) so `app.retrieval.__init__` can import it safely.

## Phase 9 — syllabus-aware library book recommendation

`POST /recommendations` ranks library books by how well their chapters cover
the syllabus topics relevant to a query. The design rule from the task is
enforced: **no book is ever assigned to a semester/subject by hand** — every
mapping comes from retrieval evidence over the indexed documents.

```
User query
  -> TopicResolver.identify(query)
       expand aliases (DBMS -> Database Management System)
       embed -> search university_docs (document_type=syllabus)
       evidence: cosine >= RECOMMENDATION_MIN_SCORE  OR  >=1 shared content token
       dominant subject from evidence payloads
       required topics = union(chunk topic payloads, graph topics_of_subject(subject))
  -> RecommendationService._book_profiles()
       distinct chapter topics / subtopics per library_book from payloads
  -> CoverageCalculator.rank(profiles, required topics)
       score = 0.85 * topic_coverage + 0.15 * subtopic_coverage
       topic match: exact normalized name  OR  cosine >= 0.4 (semantic)
  -> ranked books + matched/missing topics + grounded explanation
```

Design decisions:

- **Evidence over rules for topic identification.** The resolver accepts a
  syllabus chunk as evidence by cosine floor *or* by sharing at least one
  meaningful token with the query (stopwords removed, `book`/`recommend`
  removed). This is deliberate: the deterministic test embedder scores large
  merged chunks near zero even when they share a token, so a pure cosine gate
  would wrongly report "no syllabus". Query tokens are the boundary — a query
  like "quantum physics research" shares nothing and still gets no evidence.
- **Subject expansion.** Once a subject is identified from the evidence
  chunks, *all* syllabus chunks of that subject count as evidence, so the
  required topics cover the whole subject (all units), not just the token
  overlap with the query. The knowledge graph then contributes the subject's
  official topic list (`topics_of_subject`); when payload topics exist they
  are merged (deduplicated). The graph is authoritative when chunk payloads
  are topic-less (e.g. one merged blob).
- **Aliases are general academic shorthand**, not university metadata:
  `dbms -> Database Management System`, `dsa -> Data Structures and
  Algorithms`. Expandable in `app/recommendation/resolver.py`.
- **Transparent coverage score.** `CoverageCalculator` is a pure function:
  `score = topic_weight * topic_coverage + subtopic_weight * subtopic_coverage`
  (defaults 0.85 / 0.15, configurable). `topic_coverage` = matched required
  topics / required topics; `subtopic_coverage` = matched required subtopics /
  required subtopics (0 when the syllabus defines none, so subtopic weight is
  inert). A required topic matches when the normalized name equals a book
  chapter topic (exact wins always) or the embedding cosine is at or above
  `RECOMMENDATION_SEMANTIC_THRESHOLD` (0.4). Ties: more matched topics first,
  then alphabetical title. The formula is documented in the module docstring
  and surfaced in the response (`subject`, `required_topics`, `score`,
  `matched_topics`, `missing_topics`).
- **Book profiles come from the index**, not from structure files: each
  `library_book` document's distinct chunk `topic` payloads are its chapters,
  distinct `subtopic` payloads its subtopics. A chunker change (unit/chapter
  headings always start a chunk) guarantees every chapter heading is its own
  chunk, so a small book's topics are not collapsed to the first chapter.
- **Grounded explanation.** The LLM receives the computed rankings (scores,
  matched/missing topics) and is told to explain only that evidence; with the
  stub LLM (or an LLM failure) the service falls back to a deterministic
  summary ("The top book covers the required topics." / "no evidence").
- **Degradation.** No syllabus evidence -> `no_syllabus_evidence=true` and an
  empty list; an unreachable graph is treated as no graph topics; empty query
  -> 422.
- **Import discipline.** `app/recommendation/__init__.py` is minimal;
  `TopicResolver` and `RecommendationService` import `QdrantRepository` only
  under `TYPE_CHECKING` (eager import would cycle through
  `app.retrieval.*` -> `app.repositories.qdrant`). Likewise
  `app/retrieval/normal_retriever.py` imports `QdrantRepository` only under
  `TYPE_CHECKING` so `app.repositories.qdrant` can be imported first by any
  consumer (scripts, tests) without partial-init failures.

## Phase 10 — PostgreSQL chat history

Chat history adds conversation persistence on top of the answer pipeline: a
session keeps its message thread, and the prompt builder injects a bounded
window of that thread as conversation context. This phase is **persistence
only** — there is no long-term memory, user personality or preference
learning; everything besides the bounded conversation window is ignored.

```
POST /chat  {session_id: null | "..."}
  -> ChatService.answer(message, top_k, session_id)
       session_id == null -> ChatHistoryRepository.create_session
                             title = derive_title(first message)
       known session      -> ChatHistoryRepository.get_messages (window)
       unknown session    -> SessionNotFoundError -> 404
       user message added (always, before retrieval)
       retrieve -> ContextBuilder.build(query, chunks, history=window)
       prompt: CONTEXT ... CONVERSATION (window) ... User Query ... Answer
       assistant message added -> response {answer, sources, session_id}
GET /chat/sessions                -> list (id, title, created_at, message_count)
GET /chat/sessions/{session_id}   -> detail + full messages
DELETE /chat/sessions/{session_id}-> 204 (cascade deletes messages)
```

Data model (`app/models/chat.py`):

```
ChatSession(session_id: str PK, title: str(120), created_at, updated_at)
ChatMessage(message_id PK, session_id FK -> ChatSession CASCADE,
            role: user | assistant | system, content: Text, created_at)
```

Design decisions:

- **Repository mirrors the graph pattern.** `ChatHistoryRepository` wraps a
  sync SQLAlchemy engine and runs every call through `asyncio.to_thread`.
  `DATABASE_BACKEND=memory` (default) uses `sqlite://` with a `StaticPool`
  (per-process, one engine per `:memory:` DB), `postgres` uses `DATABASE_URL`
  via psycopg, and `auto` treats a `sqlite://`-prefixed URL as memory and
  anything else as postgres. `create_all` runs on init (`create_tables=False`
  skips connecting, for dialect-only tests).
- **Bounded conversation context.** Only the most recent
  `HISTORY_MAX_MESSAGES` (12) messages are loaded, and `_render_history`
  walks them backwards, keeping lines while `HISTORY_MAX_CHARS` (2000) allows
  — the most recent line is always included. The window is injected as a
  `CONVERSATION:` block between `CONTEXT:` and `User Query:`, so retrieved
  knowledge stays primary and the history never exceeds the budget.
- **Explicit session lifecycle.** `POST /chat` without `session_id` creates a
  session whose title is derived from the first user message
  (`SESSION_TITLE_MAX_CHARS=60`, punctuation stripped); an unknown id is a 404
  (client error), never a silent new session. The user message is persisted
  before retrieval, so even a failed generation leaves the thread intact.
- **Reads and deletes are self-contained.** `get_messages(limit)` returns the
  most recent `limit`, oldest-first, for a bounded window; `list_sessions`
  outer-joins message counts; `delete_session` cascades via FK
  `ON DELETE CASCADE`. Sessions are anonymous (no auth) — identity is the
  `session_id` itself.
- **Wiring.** `main.py` owns the repository (`app.state.chat_history`),
  passes it to `ChatService`, logs the resolved backend, and disposes the
  engine on shutdown. `history=None` keeps the legacy stateless behavior, so
  every earlier phase test still passes unchanged. `GET /history` reuses the
  same repository and is no longer a `501` placeholder.

## Phase 11 — controlled web search

The router gains a fourth strategy, `WEB`, for questions that need current or
external information. The core rule is **university priority**: internal
strategies are checked first, university-domain vocabulary keeps a query on
the university documents, and web content never overrides authoritative
university documents for university-specific facts.

```
User query
  -> RuleBasedQueryAnalyzer
       GRAPH / HYBRID rules first (university priority)
       -> WEB signals (latest, current, news, pricing, today, what happened, ...)
            + university context  -> WEB {mixed: true}  (internal + web)
            no university context -> WEB {}              (web only)
       -> else NORMAL
  -> WEB plan -> WebRetriever.retrieve(query, top_k)
       provider.search under asyncio.wait_for(WEB_SEARCH_TIMEOUT)
       success -> SearchResult[collection="web", method="web",
                                metadata={title, url, retrieved_at}]
       failure / timeout -> WebRetrievalError -> plan.fallback = true
                            -> internal (hybrid) retrieval instead
  -> ChatService splits web vs university sources
       ContextBuilder: CONTEXT: (university) ... WEB RESULTS: (title + URL +
       retrieval time + snippet) ... CONVERSATION: ... User Query ... Answer
```

Design decisions:

- **University priority via rule order.** GRAPH and HYBRID exact-match rules
  fire before web detection, so "Which subjects are in semester 5?" and
  "What does regulation 12 say about attendance?" can never become web
  queries. Then `_analyze_web` triggers only on explicit current/external
  signals (`latest`, `current`, `today`, `news`, `pricing`, `price`,
  `upcoming`, `recent`, `updated`, `forecast`, `weather`, `what happened`).
  University-domain vocabulary (`attendance`, `semester`, `syllabus`,
  `academic calendar`, `exam`, `marks`, `regulation`, ...) makes a query
  internal (no signal) or **mixed** (signal present): mixed plans run
  internal hybrid retrieval *and* web, with university documents rendered
  first in `CONTEXT:` and web results in their own `WEB RESULTS:` block.
- **Controlled, attributed sources.** `WebRetriever` wraps the provider
  behind the shared `retrieve(query, top_k)` interface and the shared
  `SearchResult` shape (`collection="web"`, `method="web"`). Every result
  preserves `title`, `url`, `snippet` and an ISO-8601 `retrieved_at`; the
  prompt cites them as `[Web N] title (url), retrieved <time>` and the API
  exposes `kind: "university" | "web"`, `url` and `retrieved_at` per source,
  so clients (and the LLM) can always tell web from university evidence.
- **Web content is untrusted, supplementary evidence.** The grounding prompt
  requires URL citation, states web results are not automatically
  trustworthy and may change over time, and makes university documents
  authoritative for university-specific facts. Empty web results produce the
  explicit "insufficient information" answer — never a confident
  fabrication. No web fallback is triggered from an empty *internal*
  retrieval (that would turn "What is the capital of France?" — a Phase 5
  acceptance case — into a web answer).
- **Graceful failure and timeout.** Providers run under
  `asyncio.wait_for(WEB_SEARCH_TIMEOUT)`. A `WebSearchError` or timeout is
  caught in the router (`plan.fallback = true`), which falls back to
  internal hybrid retrieval; the answer always completes. Empty results are
  *not* a failure — they surface as no evidence.
- **Backends.** `WEB_SEARCH_BACKEND=stub` (default) is a deterministic,
  offline provider with canned results and a fixed retrieval timestamp —
  hermetic for tests. `duckduckgo` performs live search over the public
  DuckDuckGo HTML endpoint with only the standard library (no API key),
  parses result links/snippets and resolves redirect URLs; its HTML parser
  is unit-tested against canned markup. `POST /web/search` exposes the
  provider directly for verification.

## Phase 12 - React + FastAPI integration

The web UI is a React (Vite) single-page application that talks to the
existing FastAPI backend over HTTP/JSON. No new backend query paths were
added; the frontend consumes the Phase 1-11 API surface.

### Browser-to-backend plumbing

- **Dev proxy.** `frontend/vite.config.js` proxies every `/api/*` request to
  `http://localhost:8000` (override `VITE_BACKEND_URL`) with the `/api`
  prefix stripped, so the UI can use clean `/api/...` paths while the backend
  routes stay at the root (`/chat`, `/recommendations`, ...). Stripping the
  prefix (rather than moving backend routes under `/api`) keeps the ~316
  existing endpoint tests untouched.
- **CORS.** `app/main.py` adds `CORSMiddleware` whose origins come from the
  `CORS_ORIGINS` setting (comma-separated, default
  `http://localhost:5173,http://127.0.0.1:5173`). `allow_credentials` stays
  off; methods and headers are open.
- **API client.** `frontend/src/services/apiClient.js` is the single fetch
  wrapper: JSON serialization, an AbortController timeout (60 s), and an
  error mapper that converts HTTP statuses and transport failures into
  `ApiError` instances with user-facing messages. `503` → "model / search
  index / database unavailable", `404` → "conversation not found", `422` →
  invalid request, `501` → not implemented, timeouts and network failures →
  retryable hints. Stack traces never reach the browser.

### UI structure

- `App.jsx` owns the state: the session list, the active session id, the
  message thread, and the pending flag. `Sidebar` lists sessions (title,
  message count, relative time) with new-chat and per-session delete; the
  active session loads its stored messages via `GET /chat/sessions/{id}`.
- `ChatInterface` renders the welcome screen (suggested prompts), the
  message thread, and the composer (Enter to send, Shift+Enter for a new
  line). A "Book recommendations" toggle switches the send target between
  `POST /chat` and `POST /recommendations`.
- `MessageBubble` renders user/assistant messages; assistant messages can
  carry sources, the routing strategy badge (with fallback marker), book
  recommendations, or an inline error banner. The typing indicator is a
  non-blocking animation — the composer stays usable for follow-ups.
- `SourceCard` renders one source. University sources show title, page and
  document type from the payload metadata; web sources show title, the
  clickable URL and the retrieval time.
- `RecommendationCard` renders one book: title, author, coverage score,
  matched/missing topic chips, subtopic counts, indexed chapters and the
  evidence list; the top-level explanation is shown above the cards.

### Author metadata flow

`DocumentChunk` and the chunker now carry `author` (from the document
metadata), so indexed payloads include it. `BookProfile.author` and
`BookCoverage.author` propagate it through the recommendation service to
`BookRecommendationSchema.author`, which the UI displays under the title.
PDFs without author metadata yield `author: null` (the UI omits the line).

### Empty-index behavior

The recommendation resolver previously surfaced `Collection ... not found`
as a 500 when nothing was indexed yet. `TopicResolver.identify` now treats a
missing collection as "no syllabus evidence" and returns the graceful empty
recommendation (`no_syllabus_evidence: true`, no books) — matching the
"backend not ready" error-handling contract the UI relies on.

### Responsive layout

Desktop shows a fixed 264 px sidebar. Below 720 px the sidebar becomes a
slide-in drawer toggled by the header menu button, the composer wraps, and
message bubbles take nearly full width — usable on tablets and phones.

## Design principles

- Ingestion is independent from query answering.
- Retrieval is independent from generation.
- No hard-coded university subjects/topics; metadata comes from the documents.
- No fabricated metadata; source page numbers preserved.
- Components expose stable interfaces so backends can be swapped later.
- The backend remains a single FastAPI application with internal modules
  (no unnecessary microservices).
