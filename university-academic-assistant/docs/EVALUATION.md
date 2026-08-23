# Evaluation Methodology and Results (Phase 13)

This document defines how the system is evaluated and reports the current
results. Everything here runs **hermetically** (no network, no external
services) so the numbers are reproducible on any machine:

```bash
cd backend
QDRANT_URL=":memory:" EMBEDDER_BACKEND="deterministic" LLM_BACKEND="stub" \
ROUTER_CLASSIFIER="rules" GRAPH_BACKEND="memory" DATABASE_BACKEND="memory" \
python scripts/evaluate.py
```

On Windows PowerShell the variables must be set in the same command:

```powershell
$env:QDRANT_URL=":memory:"; $env:EMBEDDER_BACKEND="deterministic"; $env:LLM_BACKEND="stub"; $env:ROUTER_CLASSIFIER="rules"; $env:GRAPH_BACKEND="memory"; $env:DATABASE_BACKEND="memory"; python scripts/evaluate.py
```

The environment (`app/evaluation/environment.py`) is the real pipeline wired to
in-memory implementations: Qdrant `:memory:`, a deterministic token-hash
embedder, the stub LLM, an in-memory knowledge graph and BM25. Nothing is
mocked at the API boundary — the same services the server uses are evaluated.

---

## 1. Router evaluation

**Method.** `app/router/eval_set.py` contains 36 labeled queries (10 NORMAL,
10 HYBRID, 10 GRAPH, 6 WEB). Each query's expected strategy is fixed in the
data; the router's plan is compared against it.

- **Routing accuracy** — fraction of queries routed to the expected strategy.
- **False routing rate** — fraction routed to an unexpected strategy.
- **Fallback rate** — fraction that fell back to NORMAL although the query had
  an expected strategy.

**Result (run with rules classifier, offline).**

| metric | value |
|---|---|
| routing accuracy | 100.00% |
| false routing rate | 0.00% |
| fallback rate | 0.00% |

## 2. Retrieval evaluation

**Method.** `app/evaluation/corpus.py` defines a synthetic academic corpus of
7 documents covering every category the system must handle:

- `syllabus-dbms`, `syllabus-dsa` — two subject syllabi (semester 5)
- `notice-attendance` — attendance notice
- `regulations-exam` — examination regulations
- `calendar-2024` — academic calendar
- `pastpaper-dbms-2080` — past question paper with extracted questions
- `book-dbms-concepts`, `book-os-principles` — library books with
  author/chapter/subtopic metadata

`RETRIEVAL_QUERIES` labels 11 queries with their relevant documents
(`expected_docs`) and their expected strategy. Three strategies are compared:

- **NORMAL** — dense vector search (deterministic embedder)
- **HYBRID** — dense + BM25 combined with RRF
- **GRAPH** — knowledge-graph relationships (topics of subject, past
  questions about a topic, subjects of a semester)

Metrics per strategy, computed over the queries labeled for that strategy:
Recall@1/3/5, Precision@3/5, MRR (`app/evaluation/metrics.py`).

> **Why per-strategy subsets?** The graph retriever is *specialized*: it
> legitimately returns no rows for semantic queries (the chat pipeline falls
> back to dense retrieval for those). Averaging graph metrics over all queries
> would penalize it for correctly staying silent. `empty_rate_on_other_queries`
> guards the opposite failure — a strategy producing rows for queries labeled
> for another strategy.

**Result.**

| strategy | recall@1/3/5 | precision@3/5 | MRR | empty on other |
|---|---|---|---|---|
| NORMAL | 0.20 / 0.70 / 0.70 | 0.40 / 0.28 | 0.60 | 0.00 |
| HYBRID | 1.00 / 1.00 / 1.00 | 0.39 / 0.30 | 1.00 | 0.00 |
| GRAPH | 0.83 / 1.00 / 1.00 | 0.78 / 0.75 | 1.00 | 1.00 |

HYBRID beats pure dense on exact-token administrative facts (attendance,
regulations, calendar) and is therefore the chat default. GRAPH answers its
structural questions perfectly and never pollutes other queries.

## 3. Answer evaluation

**Method.** The chat pipeline (router -> retrieval -> context assembly ->
LLM) is run against a *grounded stub* LLM that answers by quoting the
strongest retrieved source verbatim. Because the generator is deterministic,
the metrics measure the **pipeline** (retrieval feeding the generator,
context assembly, grounding discipline), not model quality. With a real
Ollama model the identical methodology applies — the metrics are computed
from the final answer text and the retrieved sources:

- **Faithfulness** — fraction of answer content (whitespace-trimmed 4-grams
  over meaningful tokens) that appears in the retrieved sources. 1.0 = fully
  grounded.
- **Relevance** — retrieval hit rate: the fraction of queries whose corpus
  has evidence that ran with non-empty evidence.
- **Source correctness** — every source referenced by the answer
  (`[Source N]` markers) exists in the retrieved source list.
- **Hallucination rate** — fraction of answers containing substantive
  n-grams found in no retrieved source; a response that asserts a specific
  fact while quoting nothing is also flagged.

**Result.**

| metric | value |
|---|---|
| mean faithfulness | 100.00% |
| relevance (evidence) | 100.00% |
| source correctness | 100.00% |
| hallucination rate | 0.00% |

## 4. OCR evaluation

**Method** (`app/evaluation/ocr.py`). Two documents are generated in-memory
and extracted with the real pipeline:

- **digital** — a PDF with extractable text; extracted with pymupdf.
- **scanned** — an image-only PDF (each page rendered to an image); extracted
  with pytesseract when the Tesseract binary is available, otherwise skipped
  (machine limitation, method still defined).

Metrics: character accuracy and word recall against the known source text.

**Result.**

| scenario | method | char accuracy | word recall |
|---|---|---|---|
| digital | pymupdf | 100.00% | 100.00% |
| scanned | tesseract | skipped — Tesseract OCR binary not found on this machine | |

## 5. Library recommendation evaluation

**Method** (`app/evaluation/recommendations.py`). Each scenario builds its own
hermetic mini-corpus (syllabus + books) so coverage behavior is exercised
independently. Matching is exact normalized-name matching
(`semantic_threshold=1.0`): the crude deterministic embedder would otherwise
match loosely-related subtopics above the default 0.4 threshold. Semantic
synonym matching is a real-model behavior validated by the model-backed
integration tests instead.

The documented formula is verified per book:
`score = 0.85 * topic_coverage + 0.15 * subtopic_coverage`.

**Result.** 9/9 checks pass:

| check | meaning |
|---|---|
| high_coverage_scores_highest | the book covering all required topics ranks first |
| high_coverage_matches_all_topics | its matched topics equal the required set |
| medium_below_high | a partial book scores between strong (0.42) and low (0.00) |
| low_coverage_scores_zero | an unrelated book scores 0.00 with no matched topics |
| missing_topic_reported | a required topic absent from a book appears in `missing_topics` |
| score_follows_formula | score equals 0.85*topic_cov + 0.15*subtopic_cov |
| similar_books_both_ranked | both candidates are returned |
| similar_books_richer_subtopics_first | the book with more required subtopics ranks first |
| similar_books_formula_matches | scores on the similar-books scenario follow the formula |

---

## Findings and fixes surfaced by the evaluation

The evaluation was not decorative — it found three real defects that were
fixed in the product code:

1. **Chunk topic mislabeling with overlap.** `_group_blocks` prepended the
   overlap carry in front of the new chunk, so a chunk could inherit the
   *previous* chapter's heading as its topic payload (the last chapter of a
   book was labeled with the previous chapter's topic). Fix: `_overlap_tail`
   excludes pure heading lines, and `_build_chunk` labels a chunk from its
   own heading block. (`app/chunking/chunker.py`)
2. **Subtopic term of the coverage formula could never fire.** Syllabus
   chunks never carry subtopic payloads and book chapters merged into single
   chunks lost subsection payloads, so `required_subtopics` was always empty
   and `0.15 * subtopic_coverage` was always 0. Fixes: the resolver now also
   takes subtopics from the knowledge graph for the identified subject's
   topics (`app/recommendation/resolver.py`), and library-book chunk payloads
   now carry the book's structural subtopics
   (`app/services/vector_indexing.py`), which the recommendation profile
   reads (`app/services/recommendation.py`).
3. **Stub answer generator stopped at the first paragraph**, quoting only
   the source header. The grounded responder now quotes the full source
   body. (`app/evaluation/answers.py`)

All 316 backend tests still pass after these fixes.

## Running the numbers again

```powershell
cd backend
$env:QDRANT_URL=":memory:"; $env:EMBEDDER_BACKEND="deterministic"; $env:LLM_BACKEND="stub"; $env:ROUTER_CLASSIFIER="rules"; $env:GRAPH_BACKEND="memory"; $env:DATABASE_BACKEND="memory"; python scripts/evaluate.py
```

The report prints router, retrieval, answer, OCR and recommendation sections;
the script exits non-zero if any recommendation check fails.
