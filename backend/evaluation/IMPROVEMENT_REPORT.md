# Adaptive University RAG Assistant — Targeted Improvement Report

Scope: smallest high-impact, low-risk changes after the full RAGAS baseline run.
No architecture change, no new dependency, no dataset/metric/methodology change.

Baseline (full 95-question run, `evaluation_results.json`):

| Metric | Value | n |
|---|---|---|
| Faithfulness | 0.594 | 73 |
| Answer Relevancy | 0.684 | 77 |
| Context Recall | 0.651 | 74 |
| Context Precision | 0.747 | 78 |
| Abstention | 13/17 | — |
| False abstentions | 10 | 78 |

---

## 1. Problems found (Phase 1 analysis)

**Faithfulness failures (0.594) had two distinct causes:**

1. **Unsupported specifics.** The lowest-faithfulness non-declined answers added
   detail the sources did not contain — page ranges, chapter numbers, extra
   sub-sections (e.g. `term-11`, `term-30`, `chap-39`, `chap-16`, `cmp-61`,
   `sum-67`). This is genuine over-generation from a small model.
2. **Model-side abstentions scored as 0.** Ten answerable questions were answered
   with the decline sentence (`declined=True`) even though chunks passed the
   evidence gate; RAGAS gives those faithfulness 0. Several of these were
   retrieval misses (the right document/paragraph was never in the kept context).
   Context was **not** truncated: kept context stayed within the 4000-char budget.

**Abstention failures (reported 13/17) were mostly a measurement artifact:**

- `none-46`, `none-76`, `none-03` **did abstain** — but paraphrased the decline
  instead of emitting the exact controlled sentence, so the harness's exact-string
  `declined` check missed them.
- `none-18` ("minimum passing mark for the Engineering Economics final exam") is
  the **only genuine hallucination**: the model read the mark-distribution table
  and inferred "half of the total = 30", inventing a rule that was never stated.

Real abstention behavior was therefore ~**16/17**; the reported 13/17 understates it.

**Recall failures (0.651) were retrieval coverage, not context sizing:**

- A retrieval-only probe (no LLM) shows the gold document is in the candidate set
  for **82.1%** of answerable questions at `top_k=5`, **85.9%** at `k=7`,
  **89.7%** at `k=10` — monotonic, with **no losses** at larger k.
- Many low-recall / false-abstention cases (`hol-01`, `boot-01`, `rules-01`,
  `term-09`, `chap-66`) had the correct source outside the top-5.

---

## 2. Changes made

| # | File | Change |
|---|---|---|
| 1 | `backend/app/rag.py` | Strengthened `SYSTEM_PROMPT`: sources are the ONLY facts; never invent page/chapter numbers, dates, names, figures, requirements; **never derive/compute** a fact the sources don't state (explicitly: don't compute a pass mark from a table); use brief/indirect/scattered evidence rather than declining; answer the question directly first and avoid padded background. |
| 2 | `backend/app/rag.py` | Added `_canonical_decline()`: a short, clearly-declining **paraphrase** from the model is normalized to the controlled `INSUFFICIENT_EVIDENCE_ANSWER` (applied on both the non-streaming path and the stream's terminal `done` answer). |
| 3 | `backend/app/llm.py` | `build_user_content` now states the positive rule plainly: if **any** source contains the requested information — even briefly, indirectly, or worded differently — answer from it; only say so if evidence truly lacks it. |
| 4 | `backend/app/config.py`, `backend/.env.example`, `architecture.md` | `RETRIEVAL_TOP_K` 5 → **7** (small, measured recall gain). |
| 5 | `backend/tests/test_generation.py` | Two unit tests: paraphrased decline is normalized; a substantive answer is left intact. |

**Deliberately NOT changed:** `MIN_RELEVANCE_SCORE` (0.45), `MIN_BM25_SCORE` (1.0),
`MAX_CONTEXT_CHARS` (4000). Raising the gate would have *increased* the 10 false
abstentions, and the one genuine hallucination (`none-18`) was a reasoning error,
not weak retrieval — so a threshold change would have been speculative and harmful.

---

## 3. Test results

- **Backend suite:** `190 passed, 1 skipped` (was 188+1; +2 new tests). Includes
  RAG, streaming, LLM, context, routing, API and RAGAS-runner tests.
- **Frontend:** `oxlint` 0 errors (3 pre-existing warnings in
  `scripts/verify_unimind_ui.mjs`, unrelated); `vite build` succeeds.
- **Targeted development validation** (14 questions = 10 answerable + 4
  unanswerable, chosen from prior failures) — *behavioural*, via the real
  pipeline (`traces_dev.json`):
  - `none-18` now **declines** instead of fabricating "30" ✅
  - `none-03`, `none-46`, `none-76` decline cleanly and are detected ✅
  - previously false-abstained questions now answer: `rules-01` (75%),
    `hol-01` (Foundation Day), `boot-01`, `detail-53`, `chap-39`, `term-11` ✅
  - subset abstention: **4/4** unanswerable declined ✅
  - remaining variance: `fact-01` and `chap-66` occasionally decline (both
    borderline; `chap-66`'s gold source is outside the retrieved set).
- **Manual API checks** (streaming endpoint): answerable question correct (75%);
  not-in-corpus question abstains; weak-evidence question does **not** invent a
  pass mark; streaming works; **no raw HTML** in any answer; multi-turn history
  resolves follow-ups ("What are its advantages?").
- **RAGAS subset: NOT run to completion.** A 14-question judge pass on local
  Ollama exceeded the remaining time budget. **This is not a benchmark and no new
  RAGAS scores are claimed.**

**Caveats (be transparent in the presentation):**
- Mid-validation the **Groq daily token quota was exhausted (HTTP 429 TPD 200000)**,
  so most live requests exercised the **Ollama fallback**, whose wording differs
  from Groq's. Provider behaviour is not identical; this is exactly what the
  fallback exists for.
- The model is a small 8B model; borderline questions (syllabus lookups) flip
  between answering and declining across runs.

---

## 4. Expected impact

| Change | Expected impact |
|---|---|
| Grounded generation rules (no invented specifics, no derivation) | Faithfulness ↑ |
| Consistent decline wording + normalization | Abstention ↑ (measurable) / Faithfulness ↑ |
| `RETRIEVAL_TOP_K` 5 → 7 | Context Recall ↑ (82.1% → 85.9% gold-doc hit rate) |
| Direct-first answers, no padding | Answer Relevancy ↑ |

These are **expected** directions, not measured RAGAS improvements. The only
figure measured directly is the retrieval gold-document hit rate.

---

## 5. Presentation talking points

1. **Baseline is a real, end-to-end RAGAS run**: 95 questions (78 answerable, 17
   unanswerable), Faithfulness 0.59, Answer Relevancy 0.68, Recall 0.65,
   Precision 0.75, Abstention 13/17.
2. **Faithfulness was the weakest metric**, and the analysis separates two causes:
   unsupported detail added by the model, and answerable questions that were
   declined (which score 0).
3. **Abstention was better than reported**: of the 4 "hallucinated" answers, 3
   were actually correct abstentions phrased differently; only one — inferring a
   pass mark from a mark-distribution table — was a real hallucination.
4. **The fixes target the real causes**: stricter grounding (no invented
   numbers/derivations), a consistent abstention sentence, and a small retrieval
   widening (5→7) justified by a measured recall gain.
5. **We deliberately kept the architecture simple**: no agents, graphs, rerankers
   or new models — the evidence showed retrieval coverage and generation
   discipline, not architecture, were the levers.
6. **How insufficient evidence is handled**: a deterministic relevance gate
   decides first; if no chunk is relevant, the LLM is never called and the
   controlled decline is returned. When evidence exists but the model still
   judges it insufficient, its decline is normalized to the same sentence.
7. **Conversation context**: the client sends history back each turn; the LLM
   uses it to resolve references, while retrieval uses a standalone query built
   from the latest turn — so follow-ups keep their lexical anchor.
8. **Evaluation integrity**: the full RAGAS benchmark is untouched; a small
   14-question development check was used only to confirm the known failure
   cases, clearly labelled as non-benchmark.
