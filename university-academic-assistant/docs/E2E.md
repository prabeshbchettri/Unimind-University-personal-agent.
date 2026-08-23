# End-to-End Acceptance Test (Phase 13)

The six acceptance scenario types were exercised through a **live server**
(`uvicorn app.main:app` on `127.0.0.1:8000`, hermetic environment:
deterministic embedder, stub LLM, rules router, in-memory Qdrant/graph/history)
over the real HTTP API. `scripts/e2e.py` builds PDFs from scratch, uploads
them through the full `/documents/index` pipeline (classify -> structure ->
chunk -> embed -> index -> graph), then runs every scenario.

Result: **20/20 checks passed** (1.47 s). Raw output: `backend/e2e_results.json`.

## Corpus (built through the real pipeline)

| Uploaded PDF | Classified as | Chunks | Collection |
|--------------|---------------|--------|------------|
| syllabus-dbms.pdf | syllabus | 4 | university_docs |
| syllabus-dsa.pdf | syllabus | 5 | university_docs |
| regulations-exam.pdf | rules_regulations | 1 | university_docs |
| pastpaper-dbms-2080.pdf | past_question | 1 | past_questions |
| book-dbms-concepts.pdf | library_book | 4 | library_books |
| notice-attendance.pdf | notice | 1 | university_docs |
| calendar-2024.pdf | academic_calendar | 1 | university_docs |

Knowledge graph after ingestion (`GET /graph/summary`): 2 Subjects, 7 Topics,
6 PastQuestions, 1 Semester; 7 `HAS_TOPIC` and 6 `HAS_QUESTION` relationships.

## Scenario results

| # | Scenario | Strategy | Check | Result |
|---|----------|----------|-------|--------|
| S1 | "Explain normalization." (syllabus) | NORMAL | 5 sources returned | PASS |
| S2 | "What does rule 8 say about grading?" (regulation) | HYBRID | source = Examination Regulations 2024 | PASS |
| S3 | "Which subjects are in semester 5?" (structural) | GRAPH | graph evidence + dense fallback (5 sources) | PASS |
| S4 | "Which past questions are about normalization?" | GRAPH | sources returned | PASS |
| S5 | "Which book is best for normalization?" (recommendation) | — | top = Database System Concepts, score 0.57 | PASS |
| S6 | "What is the latest Python version?" (current info) | WEB | 1 web source (kind=web) | PASS |

Recommendation detail: subject Database Management System, required topics
[Normalization, Introduction, Transactions], matched [Normalization,
Transactions], missing [Introduction], `topic_coverage` 0.67, score 0.57.

## Operational checks (all PASS)

- `GET /health` -> 200 `{"status": "ok"}`.
- Chat session lifecycle over HTTP: create (`session_id` returned),
  continue (same id, 200), fetch (4 messages persisted), delete (204),
  fetch again (404).
- `GET /graph/summary` -> 200 with node/relationship counts.
- All indexing ran through the real pipeline and the retrieval cache was
  invalidated by each index operation.

## Notes

- With the stub LLM the answer text is the safe grounding default
  ("I could not find sufficient information..."); the checks assert routing,
  sources and persistence rather than answer prose. Real-model quality is
  covered by `docs/EVALUATION.md` (hermetic) and the Ollama profile.
- Rerunning the script against the same server re-indexes the documents
  under new ids (graph merges are first-seen-wins, so counts grow); run it
  against a fresh server for clean repeatability.
- Reproduction: start the backend (`uvicorn app.main:app`, hermetic env from
  `scripts/evaluate.py`), then `python scripts/e2e.py`.