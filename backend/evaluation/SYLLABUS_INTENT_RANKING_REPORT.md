# Document-Type Intent Ranking (Syllabus Fix)

Targeted, low-risk fix: when a query asks for a **syllabus** (or course outline),
the course syllabus document is ranked above lecture slides — without changing
vector search, BM25 or RRF, and without hard-coding any course or filename.

---

## 1. Root cause

The ranking considered **content only** and never looked at *what kind of
document* a chunk came from:

- **BM25 indexes only chunk text** (`BM25Index.add` → `tokenize(chunk.text)`).
  The document name/folder is not searchable, so the word "syllabus" and the
  course name in the filename (`ENCT352SOFTWAREENGINEERING…`) never contribute a
  lexical signal.
- **Vector similarity** favors slide/lecture text, which repeats the course
  vocabulary; the syllabus's first page is administrative text (objectives,
  assessment scheme, module list) with less of that vocabulary.
- **RRF** fuses the two ranked lists purely by *position* — it has no notion of
  document type either.
- The stored payload contains no document-type/category field
  (`chunk_id, document_name, source_path, page, chunk_index, text`), and nothing
  in retrieval used `document_name`/`source_path` at ranking time.

So for `"syllabus of the software engineering"` the slide chunks out-ranked the
actual syllabus. Reproduced at the pre-change `top_k=5` (see §4).

---

## 2. Files changed

| File | Change |
|---|---|
| `backend/app/document_intent.py` | **New.** Keyword intent detector, document classifier, course matcher, and the post-retrieval re-ranking. |
| `backend/app/retriever.py` | `RetrievedChunk` gains `source_path` and `intent_boost` (defaults; no behavior change); `VectorRetriever`/`HybridRetriever` now carry `source_path`. |
| `backend/app/bm25.py` | `LexicalHit` carries `source_path` (metadata only). |
| `backend/app/rag.py` | Applies `apply_intent_ranking()` right after retrieval, before context construction (both the normal and streaming paths). |
| `backend/tests/test_document_intent.py` | **New.** 13 regression tests. |
| `backend/tests/conftest.py` | `make_retrieved()` accepts an optional `source_path`. |
| `scripts/inspect_ranking.py` | **New.** Prints before/after ranking with scores for any query. |

No new dependency, no extra LLM call, no change to vector search / BM25 / RRF.

---

## 3. Ranking logic added

**Intent detection** (deterministic regex, no model):

```
syllabus | syllabi | course outline(s) | course content(s) | curriculum |
course structure | course description | outline of the course
```

**Document classification** from filename + folder (metadata already stored):

| Type | Rule |
|---|---|
| `syllabus` | filename starts with a course code (e.g. `ENCT352…`) |
| `slide` | filename contains `slide` |
| `chapter` | filename contains `chapter` |
| `past_paper` | filename contains `past_question`/`past paper` |
| `notice` / `rules` | filename or folder |
| `other` | fallback |

**Course matching**: the course is derived from the folder
(`software_engineering` → `{software, engineering}`) with a filename fallback.
A document matches when every course word appears in the query.

**Boost** applied after retrieval (only when syllabus intent is detected):

```
final = 1/(rank+1) + boost          # rank = position in the retrieved list

syllabus of the named course   → +2.0   (strong)
syllabus, no course named      → +1.0   (generic "give me the syllabus")
other official doc of the course → +0.3 (past paper / chapter)
lecture slides / notices / rules → 0.0
```

Guards that keep it intent- and course-aware:

- No syllabus intent → ranking returned **unchanged** (same objects/order).
- The query names a course and a matching syllabus is present → that syllabus is
  boosted; **other courses' syllabi get no boost**.
- The query names a course we cannot match (e.g. Computer Networks, absent from
  the corpus) → **no boost at all**, rather than surfacing an unrelated syllabus.

---

## 4. Before / after ranking (Software Engineering)

Measured on the real corpus with `scripts/inspect_ranking.py`. The `BEFORE`
below is at the pre-change `top_k=5` and matches the reported problem exactly.

```
Query: syllabus of the software engineering      (hybrid, top_k=5)
BEFORE:
  1. ch_1_slide.pdf · p6     sem=0.838
  2. ch_3_slide.pdf · p2     bm25=11.372
  3. ENCT352SOFTWAREENGINEERING_2026_04_25_10_01_01.pdf · p1   sem=0.803
  4. ch_1_slide.pdf · p2     bm25=10.265
  5. ch_3_slide.pdf · p1     sem=0.801
AFTER:
  1. ENCT352SOFTWAREENGINEERING_2026_04_25_10_01_01.pdf · p1   intent_boost=+2.0  final=3.000
  2. ch_1_slide.pdf · p6     intent_boost=+0.0  final=0.500
  3. ch_3_slide.pdf · p2     intent_boost=+0.0  final=0.333
  4. ch_1_slide.pdf · p2     intent_boost=+0.0  final=0.250
  5. ch_3_slide.pdf · p1     intent_boost=+0.0  final=0.200
```

Other cases (hybrid, `top_k=7`):

| Query | Intent | Result |
|---|---|---|
| `software engineering syllabus` | syllabus | syllabus moves 4 → **1** (boost +2.0) |
| `software engineering chapter 1` | none | **unchanged** (chapter/slide content first) |
| `explain software testing in software engineering` | none | **unchanged** (lecture content first) |
| `syllabus of computer networks` | syllabus | **unchanged** — no Computer Networks doc, so no boost and no unrelated syllabus surfaced |

---

## 5. Tests passed

- New: `tests/test_document_intent.py` — **13 tests**:
  classification, intent detection (positive/negative), syllabus query ranks
  syllabus first, content query leaves ranking identical (identity), chapter
  query does not prioritize syllabus, course-specific query picks the matching
  course, unmatched-course query boosts nothing, generic syllabus query boosts
  any syllabus, boosted chunk keeps its retrieval scores, empty retrieval no-op,
  and a pipeline integration test asserting the returned citation order.
- Full backend suite: **203 passed, 1 skipped** (`pytest` exit 0).
- Frontend: unchanged by this work.

---

## 6. Limitations

- Classification and course detection are **keyword/metadata heuristics**. They
  fit the current corpus (folders + `ENC…` syllabus filenames). A corpus without
  folder structure would fall back to the filename heuristic, which is weaker.
- The boost is a **rank adjustment**, not a learned re-ranking model. It is
  intentionally simple, deterministic and explainable.
- If the requested course's syllabus is not in the corpus, no syllabus is
  boosted (by design); the query keeps its content ranking.
- The boost magnitudes (`2.0 / 1.0 / 0.3`) were chosen relative to the
  position weight `1/(rank+1)` so a matching syllabus reliably reaches the top
  while ordinary ordering is preserved; they are constants in
  `document_intent.py` and easy to tune for a demo.
