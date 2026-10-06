# Evaluation Report — Adaptive University RAG Assistant

## Test Environment

| Aspect | Configuration |
| --- | --- |
| Date | 2026-10-06 |
| Python | 3.13.5 (backend virtualenv `backend/.venv`) |
| Embeddings | local `fastembed` — `BAAI/bge-small-en-v1.5` (dim 384) |
| Vector store | Qdrant embedded engine (`QDRANT_URL=local`), collection `university_docs` |
| Corpus | 4 synthetic PDFs → 5 chunks (`scripts/generate_demo_corpus.py`) |
| LLM | `LLM_PROVIDER=auto` — Groq `openai/gpt-oss-20b` primary, Ollama `llama3.1` fallback; `GROQ_API_KEY` configured |
| Retrieval | `RETRIEVAL_TOP_K=5`, `MIN_RELEVANCE_SCORE=0.45`, `MIN_BM25_SCORE=1.0`, `MAX_CONTEXT_CHARS=4000` |

The runner (`scripts/run_evaluation.py`) executes the **real** production
pipeline (query router → retrieval → context builder → LLM gateway) in-process.
Nothing is mocked. This is a manual engineering evaluation, explicitly **not** a
scientific benchmark.

> **Run-to-run variability:** the LLM answers are non-deterministic, so the
> pass count moves by a case or two between runs (an earlier run scored 13/18;
> this run scored 12/18). The routing/retrieval signals are stable; the LLM-path
> cases are not. Treat the numbers as a qualitative check, not a fixed score.

## Result (latest run)

| Metric | Value |
| --- | --- |
| Cases | 18 |
| **Pass** | **12** |
| Fail | 6 |
| Routing matched expectation | 16 / 18 |
| Evidence gate behaved as expected | 15 / 18 |
| Grounded answers with citations | 14 |
| Controlled decline via retrieval gate | 1 (`none-16`) |
| Strategy distribution | normal 7 · hybrid 11 |
| Latency (min / median / max) | 10 ms / 770 ms / 35 268 ms |

## Routing

The deterministic router selected NORMAL for 7 queries and HYBRID for 11;
routing matched the dataset's expectation in 16 of 18 cases.

- **Correct NORMAL** (short conceptual/factual, no exact identifiers):
  `fact-01/02/03`, `sem-05`, `sem-06`, `none-16`.
- **Correct HYBRID** (course codes, policy terminology, multi-part, ambiguous):
  `term-08`, `course-09/10/11`, `multi-12/13`, `multisrc-14/15`, `none-17`,
  `amb-18`.
- **Mismatches** (judgment calls; answers still correct and grounded):
  - `sem-04` ("What happens if a student cannot attend classes because of
    illness?") → HYBRID (default) but expected NORMAL; "what happens" is not a
    recognised conceptual prefix.
  - `term-07` ("What is the re-evaluation procedure …?") → NORMAL (the "what
    is …" prefix) but expected HYBRID; "re-evaluation"/"procedure" are not in
    the terminology list.

## Retrieval

- The relevant source was surfaced as the top-1 document for nearly every
  grounded question (attendance → `attendance_policy.pdf`, library →
  `library_policy.pdf`, CS201 → `cs201_syllabus.pdf`, exam rules →
  `examination_rules.pdf`).
- Hybrid fusion (vector + BM25 + RRF) correctly retains lexical-only matches and
  their metadata (unit-tested in `tests/test_hybrid.py`).
- **Weakness:** the corpus is only 5 chunks, so `top_k=5` always returns the
  whole corpus and top-3 relevance is trivially satisfied. Retrieval
  discrimination can only be measured meaningfully on a larger corpus.

## Grounding

- 14 answers were grounded and cited.
- `sem-06` and `none-17` declined honestly with the controlled
  insufficient-evidence message — the retrieved text was related but did not
  answer the question, so the LLM declined rather than invent. `multi-12` also
  declined this run (the CS201 assessment split *is* in the corpus, so this is
  model conservatism rather than a pipeline bug).
- **Weakness (`course-11`):** "What are the attendance requirements for ENCT
  353?" — ENCT 353 does not exist in the corpus. The system answered by applying
  the general "75 percent of all scheduled sessions in every registered course"
  rule to ENCT 353. This is a plausible but over-generalised answer; the honest
  behaviour is to decline.

## Citations

- Grounded answers included `[n]` citation markers resolving to the source
  metadata (document + page) returned in `sources`.
- The CJK-bracket normalisation (`【1】` → `[1]`, for Groq `gpt-oss-20b`) held
  (see `test_cjk_style_citations_are_normalized`).

## Failure Handling

Failure modes are covered by the unit/integration suite (no paid API or
always-running external service is required; network failures are mocked with
local HTTP stubs):

- Qdrant / vector store unavailable → `503` (`test_api_chat.py`, `test_health.py`).
- Groq unavailable (429/5xx/unreachable) → typed error / fallback (`test_groq.py`).
- Ollama unavailable → typed error (`test_llm.py`).
- Both providers unavailable → clean `LLMUnavailableError` (`test_fallback.py`).
- Empty / whitespace / over-long query → `422` (`test_api_chat.py`).
- No relevant documents → controlled insufficient-evidence, LLM not called.
- Model timeout → fallback / typed error (`test_groq.py`, `test_fallback.py`).
- Malformed request body → `422` (`test_api_chat.py`).
- Invalid configuration (bad provider, missing key) → typed config error.

Client-facing errors are generic messages; stack traces, API keys and
environment values are never exposed.

## Provider Fallback

Fallback was exercised live: two cases fell back to Ollama
(`provider=ollama_fallback`, model `llama3.1`), the rest used Groq.

- `course-09`, `multisrc-15` → `ollama_fallback`.

This confirms the Groq → Ollama fallback chain works end-to-end, not only under
unit test.

## Known Weaknesses

- The corpus (5 chunks) is too small to measure retrieval quality meaningfully.
- The deterministic router's small prefix/terminology lists route some surface
  forms differently than a human classifier would expect (`sem-04`, `term-07`).
- Course-code queries for a course absent from the corpus can be answered from
  general policy wording (`course-11`) rather than declining.
- The runner's `evidence_ok` check counts only retrieval-gate declines; honest
  LLM-path declines (`sem-06`, `none-17`) are currently marked FAIL even though
  the behaviour is correct.
- LLM-path pass/fail varies run to run because generation is non-deterministic.

## Future Improvements (out of scope for Phase 7/8)

- A "course-specific evidence" gate so a course-code query whose course is
  absent from the corpus declines instead of over-generalising.
- A larger, real (non-synthetic) corpus for meaningful retrieval metrics.
- Expanding the router's prefix/terminology tables, or a small LLM-assisted router.
- Per-category relevance thresholds.
- Web-search strategy and OCR for scanned PDFs (already deferred).
