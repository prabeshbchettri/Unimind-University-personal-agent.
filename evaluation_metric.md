# RAG Evaluation Metrics

This document describes the RAGAS-based evaluation of the **Adaptive University
RAG Assistant** as it is actually implemented in this repository. It documents
the real dataset, the real pipeline, the real metric selection, and the real
configuration — not a generic RAGAS tutorial.

Evaluation code: `scripts/run_ragas_evaluation.py`
Machine-readable results: `backend/evaluation/evaluation_results.json`
Dataset (unchanged): `backend/evaluation/dataset.json`

Two run modes are distinguished throughout:

- **FULL final evaluation** (all 95 questions) — the official numbers.
- **FAST development evaluation** (`--fast`) — a clearly-labelled representative
  subset for iteration; **not** equivalent to the full run.

The evaluation is split into two **stages**: trace generation (the real pipeline)
and RAGAS judging. See §3 and §6.

---

## 1. Evaluation Objective

The goal is to obtain **quantitative signals** for three questions about the
system's behavior:

1. **Retrieval quality** — did the system retrieve the information needed to
   answer the question, and did it rank useful passages ahead of noise?
2. **Generation quality** — is the generated answer relevant to the user's
   question?
3. **Groundedness / faithfulness** — is the generated answer supported by the
   retrieved university material, rather than invented?

The evaluation measures the **actual production code path**. For every question
it calls `app.rag.answer_question(...)` — the same router, retriever, context
builder, relevance gate and LLM gateway the API server uses — and captures the
retrieved contexts (`RAGResult.retrieved`) and the generated answer
(`RAGResult.answer`). Nothing is mocked, and retrieval behavior is not changed
for the sake of the scores.

RAGAS produces **evaluation signals**, not proof of correctness. A high score
means the system behaved well *on this dataset under this judge model*; it does
not establish general correctness.

---

## 2. Evaluation Dataset

The dataset already present in the project is used as-is. **No question was
modified and no new evaluation dataset was generated.**

Source: `backend/evaluation/dataset.json` (`dataset_name =
"Adaptive University RAG Evaluation Dataset"`, `version = "1.1-notes-format"`).

| Property | Actual value |
| --- | --- |
| Questions in file | **95** (the in-file `total_questions` field still says `78` and is stale) |
| Answerable questions | **78** (`insufficient_evidence_expected = false`) |
| Unanswerable questions | **17** (`insufficient_evidence_expected = true`) |
| Courses | Simulation and Modeling (27), Software Engineering (27), Engineering Economics (19), Cross-document (7), Bootcamp Notice (6), Rules and Regulations (5), Holiday Notice (4) |
| Question types (`category`) | 20 distinct, e.g. specific detail (11), exact term (10), definition (8), chapter lookup (6), comparison (6), no answer in corpus (17), semantic paraphrase (5), summary (4), multi-part (4) |
| Difficulty | medium (53), easy (26), hard (16) |
| `relevant_source` | One or more expected source documents (or `chapter_N`) per question |
| Reference answers | **None.** There is no `reference_answer` field. |

### Dataset fields → RAGAS fields

The dataset is richer than a plain RAGAS record; it carries routing and
source expectations as well. Mapping is done by the runner:

| Dataset field | Role in evaluation | RAGAS field it feeds |
| --- | --- | --- |
| `question` | The user input | `user_input` |
| `answer_must_mention_any` | Required-information keywords (the only ground-truth content signal) | **derived `reference`** (see below) |
| `insufficient_evidence_expected` | Whether the corpus contains the answer | decides answerable vs. abstention evaluation |
| `relevant_source` | Expected source document(s) | recorded, not used by RAGAS |
| `expected_strategy` | Router expectation | recorded, not used by RAGAS |
| `course`, `category`, `difficulty` | Grouping keys for aggregation | recorded, drives per-group means |
| `id` | Stable question id | traceability in `per_question` |

### `reference_answer` gap and the derived reference

RAGAS `ContextRecall` and reference-based `ContextPrecision` require a
`reference` (ground-truth answer) that this dataset does not provide. To avoid
generating a new dataset or editing the questions, the runner builds a
**derived reference at runtime** from the dataset's own
`answer_must_mention_any` list:

```
A correct answer must include the following information: <keyword>; <keyword>; ...
```

The dataset file is never modified, and the derived string is stored per
question in the results (`reference_derived`) for full transparency. This is a
**weaker reference than a human-written answer** and the resulting
Context Recall / Context Precision scores must be read with that in mind (see
§9). Unanswerable questions have no keywords and therefore no reference — they
are excluded from reference-based metrics by construction.

---

## 3. Evaluation Pipeline

```
backend/evaluation/dataset.json (95 existing questions)
        │
        ▼  STAGE 1 — trace generation (in-process, production code path)
app.rag.answer_question()
   routing (routing.py)  →  retrieval (retriever.py: vector / vector+BM25+RRF)
   →  context gate + dedupe + size (context.py)  →  LLM gateway (llm.py)
        │
        ▼  captured per question (also saved to backend/evaluation/traces.json)
   question, generated_answer, retrieved_contexts (RAGResult.retrieved),
   sources, enough_evidence, strategy, provider, latency, reference_derived
        │
        ▼  STAGE 2 — RAGAS judging (saved traces can be re-judged without Stage 1)
        ▼  mapped to RAGAS inputs
   user_input = question
   response   = generated_answer
   retrieved_contexts = [chunk.text ...]     (actual retrieved contexts)
   reference  = derived from answer_must_mention_any   (answerable questions)
        │
        ▼  official RAGAS 0.4.3 collections metrics (LLM-as-judge = local Ollama)
   Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
   (metric calls per question run concurrently under one global limiter)
        │
        ▼  aggregate per question → overall / per-course / per-category / per-difficulty
        ▼  write backend/evaluation/evaluation_results.json (atomic, per-question checkpoint)
```

The reference answer is **never** treated as the generated answer: the
generated answer always comes from the running RAG system.

---

## 4. Selected RAGAS Metrics

RAGAS version: **0.4.3**, using the **current** API
`ragas.metrics.collections` (the legacy `ragas.metrics.<Name>` classes are
deprecated in this version and are *not* used). Selected metrics are exactly the
four below — the smallest set that covers retrieval, ranking, groundedness and
relevance.

### 4.1 Faithfulness

- **What it measures** — the fraction of factual statements in the generated
  answer that can be inferred from the retrieved context.
- **Why selected** — this is the core groundedness guarantee of the system: the
  answer must be supported by the university sources, not outside knowledge.
- **Inputs** — `user_input` (question), `response` (generated answer),
  `retrieved_contexts`.
- **Conceptual calculation** — an LLM decomposes the answer into standalone
  statements, then judges each statement `1` (supported by the context) or `0`
  (not supported); the score is supported statements / total statements.
- **Interpretation** — `1.0` means every answer claim is grounded; lower values
  mean part of the answer is unsupported by the retrieved evidence.

### 4.2 Answer Relevancy

- **What it measures** — how well the generated answer addresses the actual
  question.
- **Why selected** — a grounded answer can still miss the question; this metric
  covers relevance independently of groundedness.
- **Inputs** — `user_input` (question), `response` (generated answer); also uses
  the evaluation embedding model.
- **Conceptual calculation** — an LLM generates several questions *from the
  answer*; the metric averages the cosine similarity between those generated
  questions and the original question. Evasive/noncommittal answers are scored
  `0`.
- **Interpretation** — `1.0` means the answer is on-topic and committal; low
  values mean it is off-topic, vague, or evasive.

### 4.3 Context Recall

- **What it measures** — the share of the information required to answer the
  question that is present in the retrieved contexts.
- **Why selected** — it directly answers "did retrieval surface the information
  necessary to answer?" and separates retrieval failures from generation
  failures.
- **Inputs** — `user_input`, `retrieved_contexts`, `reference` (derived from
  `answer_must_mention_any` for answerable questions).
- **Conceptual calculation** — an LLM splits the reference into statements and
  marks each as attributable (`1`) or not (`0`) to the retrieved contexts; the
  score is attributed statements / total statements.
- **Interpretation** — `1.0` means the retrieved context contains the required
  information; low values point at retrieval misses (or at a thin derived
  reference).

### 4.4 Context Precision

- **What it measures** — whether the retrieved chunks that are useful rank ahead
  of the noisy ones (ranking-aware retrieval quality).
- **Why selected** — recall alone does not say whether relevant evidence is
  buried under irrelevant chunks; precision captures ranking quality.
- **Inputs** — `user_input`, `reference` (derived), `retrieved_contexts`
  (variant `ContextPrecisionWithReference` / `ContextPrecision`, the default).
- **Conceptual calculation** — an LLM judges each retrieved context `1`
  (useful for the reference) or `0`, then RAGAS computes average precision over
  the ranked list.
- **Interpretation** — `1.0` means useful contexts appear first; low values mean
  useful evidence is ranked poorly or the top results are mostly noise.

> A reference-free variant, `ContextPrecisionWithoutReference`, is available via
> `--context-precision without-reference`. It judges each context against the
> generated answer instead of a reference. It is **not selected by default** to
> preserve consistency with Context Recall (both use the same reference) and to
> avoid scoring precision twice.

### Metrics intentionally NOT selected

| Metric | Why not selected |
| --- | --- |
| `AnswerCorrectness` | Requires a genuine reference answer for semantic correctness against ground truth. Only a keyword-derived reference is available, which is not a valid correctness target. It is also largely redundant here with Faithfulness + Answer Relevancy + Context Recall. |
| `FactualCorrectness` | Same reference problem; additionally expensive (claim decomposition) for marginal extra signal. |
| `ContextEntityRecall` | Reference-based; entity overlap against a keyword list is noisy and would not add information beyond Context Recall. |
| `NoiseSensitivity` | Requires a reference; the precision-related signal it adds is already covered by Context Precision. |
| `AnswerSimilarity` / `SemanticSimilarity` | Reference-based; redundant with the selected answer metrics. |
| `ResponseGroundedness` (NV) | Redundant with Faithfulness. |
| `AspectCritic`, `RubricsScore`, `DomainSpecificRubrics` | Custom rubric criteria, not part of the concise core four; adds subjective evaluation surface without addressing the four stated questions. |
| `ContextUtilization` (`ContextPrecisionWithoutReference`) | An alternative precision variant, not an additional axis; exposed as a flag instead. |

This keeps the evaluation to four meaningful metrics and avoids over-evaluating.

---

## 5. Metric-to-System Mapping

| Metric | Evaluation Target | Required Data | Metrics it feeds on the system | Interpretation |
| --- | --- | --- | --- | --- |
| Context Recall | Retrieval | Question, retrieved contexts, reference (derived) | `retriever.py` (vector / hybrid) + `context.py` gate | Did retrieval contain the required information? |
| Context Precision | Retrieval ranking | Question, reference (derived), retrieved contexts | `retriever.py` ranking + RRF fusion | Is useful context ranked ahead of noise? |
| Faithfulness | Grounded generation | Question, retrieved contexts, generated answer | `llm.py` generation under the grounding system prompt | Is the answer supported by the retrieved material? |
| Answer Relevancy | Generation relevance | Question, generated answer | `llm.py` generation + intent handling | Does the answer address the question? |

Unanswerable questions (`insufficient_evidence_expected = true`) are **not**
scored with the four metrics. Because no correct answer exists, the only correct
behavior is to abstain, so they are evaluated separately as
`abstention_correct` in the results (see §7). Their per-question RAGAS scores
remain `null`.

---

## 6. Evaluation Procedure

The evaluation is split into two stages so the expensive production pipeline and
the RAGAS judging can be run independently.

- **Stage 1 — trace generation**: run the real pipeline over the dataset and
  save per-question traces (question, generated answer, retrieved contexts,
  sources, evidence flag, strategy, provider, latency, derived reference).
- **Stage 2 — RAGAS judging**: score saved traces. No pipeline is run, so RAGAS
  settings (judge, strictness, concurrency, metric variant) can be iterated
  cheaply against the *same* answers and contexts.

`--stage all` (default) runs both, i.e. the complete end-to-end evaluation.

All commands run from `backend/` (so `.env` and `.models/` resolve).

### Full final evaluation (the official numbers)

```bash
# Complete end-to-end run over all 95 questions (78 answerable + 17 unanswerable)
python ../scripts/run_ragas_evaluation.py --concurrency 2
```

### Fast development evaluation

```bash
# Clearly-labelled representative subset (10 stratified answerable + all 17
# unanswerable), AnswerRelevancy strictness 1
python ../scripts/run_ragas_evaluation.py --fast
```

Fast runs are written to the results file with `run.mode = "fast"` and a
`run.mode_warning`. **They are never equivalent to the full evaluation and must
not be reported as such.** Fast mode changes only *which questions* and
*strictness*; the metrics, judge, dataset questions, answers and contexts are
unchanged.

### Two-stage workflow (recommended when iterating on RAGAS itself)

```bash
# Stage 1: run the real pipeline once, save traces
python ../scripts/run_ragas_evaluation.py --stage traces --save-traces evaluation/traces.json

# Stage 2: judge the saved traces (repeat freely; no pipeline call)
python ../scripts/run_ragas_evaluation.py --stage judge --traces evaluation/traces.json
python ../scripts/run_ragas_evaluation.py --stage judge --traces evaluation/traces.json \
    --answer-relevancy-strictness 3     # same traces, stricter metric
```

### Subsets and filters

Filters apply identically in both stages:

```bash
python ../scripts/run_ragas_evaluation.py --limit 10
python ../scripts/run_ragas_evaluation.py --ids fact-01,def-48,rules-01
python ../scripts/run_ragas_evaluation.py --course "Software Engineering"
python ../scripts/run_ragas_evaluation.py --category definition
python ../scripts/run_ragas_evaluation.py --difficulty hard
python ../scripts/run_ragas_evaluation.py --answerable-only
python ../scripts/run_ragas_evaluation.py --unanswerable-only
```

### Resume and checkpointing

- Results and traces are written **atomically** (temp file + replace), so an
  interrupt never corrupts `evaluation_results.json`.
- A checkpoint is written after **every** completed question.
- On the next run, questions already scored under the **same judging
  configuration** are skipped and their scores reused; questions whose pipeline
  output matches the current pipeline configuration skip Stage 1 entirely.
- `--no-resume` forces a full recomputation; `--no-cache` disables the disk cache.

A run that stops at 40/95 therefore continues with the remaining questions
instead of restarting from zero.

### Caching

- Judge prompt/response cache and embedding cache live under
  `backend/.ragas_cache/`.
- The cache is **namespaced** by `judge_model` + a hash of the judging
  configuration (RAGAS version, metric set, context-precision variant,
  strictness) + a short pipeline-configuration hash. Results from different
  judge models, answers, contexts, references, or metric settings therefore
  cannot silently mix. The namespace is recorded in `run.cache_namespace`.
- Cache keys are the RAGAS prompts themselves, which embed the question, the
  generated answer, the retrieved contexts and the reference — so a changed
  answer or context is a different key automatically.

### Concurrency and the local judge bottleneck

RAGAS 0.4.x makes several judge calls per answerable question. Current call
counts with this configuration (contexts = top-`k` retrieved chunks, ≤ 5):

| Metric | Judge calls per answerable question | Notes |
| --- | --- | --- |
| Faithfulness | 2 | statement extraction + NLI verdict |
| Answer Relevancy | `strictness` (3 full / 1 fast) | question generation per answer |
| Context Recall | 1 | statement attribution |
| Context Precision | = number of retrieved contexts (≤ 5) | one usefulness verdict per context |
| **Total (full, strictness 3)** | **~11** | ~858 calls over 78 answerable questions |
| **Total (fast, strictness 1)** | **~9** | ~90 calls over 10 answerable questions |

Metric calls within a question run concurrently and questions run concurrently,
all bounded by a single global `--concurrency` limiter (max in-flight judge
requests).

**Measured judge-concurrency benchmark** (3 fresh judge calls per
autoconfiguration, local `llama3.1` on a GTX 1650 4 GB / 16 GB RAM machine):

| Concurrency | 3 calls | Per call |
| --- | --- | --- |
| 1 | 13.4 s | 4.5 s |
| 2 | 12.7 s | 4.2 s |
| 4 | 13.6 s | 4.5 s |

**Interpretation:** the local Ollama server processes requests effectively
serially, so raising `--concurrency` above 2 does **not** reduce wall time on
this machine; it only risks GPU/RAM pressure. The recommended default is
`--concurrency 2` (safe, and beneficial if `OLLAMA_NUM_PARALLEL` is later
increased). Because concurrency does not parallelise the judge, the dominant
speed levers are **fewer calls** (fast mode strictness), **caching**, and
**not re-running the pipeline** (two-stage + resume).

**Real metric calls are much heavier than the micro-benchmark.** The metric
prompts carry the retrieved contexts and use complex structured-output schemas,
which the local 8B model handles slowly (observed ~1.5–4 minutes per judge call
on the same machine). A full run is therefore dominated by its 700–860 serial
judge calls, not by scheduling. This is the reality the optimizations target:
reduce the number of calls and never pay for the same work twice.

Approximate workload per configuration:

| Configuration | Questions | Answerable scored | Judge calls (est.) |
| --- | --- | --- | --- |
| Full, `--answer-relevancy-strictness 3` (RAGAS default) | 95 | 78 | ~858 |
| Full, strictness 1 | 95 | 78 | ~702 |
| Fast (`--fast`, strictness 1) | 27 (10 answerable + 17 unanswerable) | 10 | ~90 |
| Re-run (unchanged config) | same | 0 | 0 (resume + cache) |
| Stage 2 only on changed strictness | same | 0 | only the changed metric's calls |

> Verified end-to-end on this project: a real answerable question produced all
> four metrics (e.g. Faithfulness `1.0`, Answer Relevancy `0.98`,
> Context Recall `0.5`, Context Precision `0.25`), and an unanswerable question
> was correctly excluded from the metrics and marked as a correct abstention.

### What each step does

1. Load the existing dataset and apply `--fast` and/or the filters.
2. Stage 1: build the same pipeline objects as the API server
   (`build_pipeline`) and run each question through `answer_question(...)`,
   capturing the trace.
3. Stage 2: build the RAGAS judge (local Ollama via an OpenAI-compatible async
   client) and the embedding adapter (the project's local fastembed model).
4. Score each **answerable** question with the four metrics concurrently
   (bounded by `--concurrency`); score each **unanswerable** question for
   abstention correctness (no judge calls).
5. Aggregate and write the results file, checkpointing after every question.

Useful flags: `--stage`, `--fast`, `--fast-size`, `--save-traces`, `--traces`,
`--ids`, `--limit`, `--course`, `--category`, `--difficulty`, `--answerable-only`,
`--unanswerable-only`, `--answer-relevancy-strictness`, `--context-precision`,
`--concurrency`, `--judge-model`, `--judge-base-url`, `--judge-timeout`,
`--no-cache`, `--no-resume`, `--run-notes`.

Notes:

- The judge is the project's **local Ollama** model (`OLLAMA_MODEL`, default
  `llama3.1`) — no paid API is introduced. Generation still uses the system's
  configured provider (`LLM_PROVIDER=auto` → Groq primary, Ollama fallback), so
  the pipeline is measured as deployed.
- The Qdrant embedded engine takes an exclusive file lock: **do not run the API
  server** while Stage 1 executes. Stage 2 (`--stage judge`) does not touch
  Qdrant beyond loading the embedding model, but is normally run without the
  server too.

---

## 7. Aggregation

`backend/evaluation/evaluation_results.json` contains:

- **`run`** — evaluation configuration: RAGAS version, selected metrics,
  judge/Ollama model, embedding model, generation provider, retrieval settings
  (`top_k`, `min_relevance_score`, `min_bm25_score`, `max_context_chars`),
  dataset path/version, filters, timestamp.
- **`overall`** — mean of each metric over **answerable** questions (with `_n`
  coverage counts). `null` where a metric had no valid values.
- **`by_course`**, **`by_category`**, **`by_difficulty`** — the same metric means
  computed within each group, so weak areas can be located.
- **`abstention`** — `unanswerable_questions`, `correct_abstentions`,
  `hallucinated_answers` (unanswerable questions that were answered), plus
  `answerable_questions` and `false_abstentions` (answerable questions the
  system declined).
- **`per_question`** — one row per question: id, course, category, difficulty,
  answerable, question, `generated_answer`, `retrieved_context_count`,
  `retrieved_contexts`, `sources`, `strategy`, `provider`, `enough_evidence`,
  `declined`, `reference_derived`, per-metric `scores`, and `metric_errors`.
- **`failures`** — questions with pipeline errors, metric errors, or a
  hallucinated answer to an unanswerable question.

Means ignore `null`/`NaN` values and are reported with their sample count so a
low-coverage mean is not mistaken for a confident one.

---

## 8. Result Interpretation

There are **no official RAGAS score thresholds** and none are claimed here. The
bands below are **project-specific, heuristic** guidance for reading the numbers,
not universal standards.

| Band | Heuristic reading |
| --- | --- |
| ≈ 0.85 – 1.00 | Strong signal for that metric on this dataset |
| ≈ 0.60 – 0.85 | Mixed: some failures worth inspecting in `per_question` |
| < 0.60 | Weak: investigate retrieval, gate, and generation for those questions |

How to read combinations:

- **Low Context Recall** → retrieval/gating did not surface the required
  information. Inspect `retrieved_contexts` — likely a corpus/index or threshold
  issue.
- **Low Context Precision with acceptable Recall** → relevant evidence exists but
  is ranked below noise (fusion/ranking issue).
- **Low Faithfulness** → answers contain claims not supported by the context
  (hallucination risk).
- **Low Answer Relevancy** → answers are off-topic or evasive; check whether
  question types are being handled as intended.
- **High faithfulness + low recall** → the system is conservative: it stays
  grounded but misses information it could have retrieved.

Abstention is judged separately: `correct_abstentions /
unanswerable_questions` should be high, and `hallucinated_answers` should be 0.
Answerable questions counted under `false_abstentions` score poorly on the
generation metrics and should be inspected (an honest-looking decline that is
actually a retrieval/gate false negative).

---

## 9. Limitations

- **LLM-as-a-judge.** Faithfulness, Context Recall, Context Precision, and the
  question-generation step of Answer Relevancy are model judgments, not ground
  truth. Scores depend on the judge model and can vary between runs.
- **Local judge quality.** The default judge is the local Ollama `llama3.1`
  model, chosen to keep evaluation free and offline. It is smaller and less
  consistent than a frontier hosted model, so scores carry more noise. The judge
  is configurable via `--judge-model` / `--judge-base-url`.
- **Derived references.** Context Recall and Context Precision use a reference
  derived from `answer_must_mention_any`, not a human-written reference answer.
  Keyword-shaped references limit how nuanced those two metrics can be; treat
  them as directional, and prefer `--context-precision without-reference` when a
  reference-independent precision signal is desired.
- **Dataset scope.** The corpus is a small set of synthetic/course PDFs. A
  handful of `retrieval_top_k = 5` chunks can cover a large share of a small
  corpus, which can inflate Context Precision; these numbers are not a
  large-corpus retrieval benchmark.
- **Reference answers absent.** Because the dataset has no `reference_answer`,
  no semantic *correctness* metric can be computed. Adding real reference answers
  later would enable `AnswerCorrectness` and strengthen Context Recall/Precision.
- **Evaluator bias / non-determinism.** Generation (`LLM_TEMPERATURE = 0.1`) and
  the judge are non-deterministic; pass counts and score decimals drift slightly
  between runs.
- **Cost / latency.** Local judging is CPU/GPU-bound and makes roughly 9–11
  judge calls per answerable question (Faithfulness 2, Answer Relevancy
  `strictness`, Context Recall 1, Context Precision ≤ `top_k`). A full
  78-answerable-question run is long. The namespaced disk cache, the two-stage
  trace/judge split, per-question checkpointing/resume, and `--fast` are the
  mitigations; local concurrency does not parallelise judging (§6).
- **Not a correctness proof.** These are quantitative signals about retrieval and
  generation quality, not evidence that the system is correct in general.

---

## 10. Reproducibility

| Item | Value |
| --- | --- |
| RAGAS version | **0.4.3** (`ragas.metrics.collections` API) |
| RAGAS dependency pin | `langchain-community==0.3.31` (ragas 0.4.3 imports `langchain_community.chat_models.vertexai`, removed in 0.4.x), `openai==1.109.1` |
| Judge LLM | local Ollama, `OLLAMA_MODEL` (default `llama3.1`), OpenAI-compatible endpoint `OLLAMA_BASE_URL/v1` |
| Embeddings (Answer Relevancy) | project local fastembed `BAAI/bge-small-en-v1.5` (dim 384) |
| Generation provider (measured system) | `LLM_PROVIDER` (default `auto`: Groq `openai/gpt-oss-20b` → Ollama `llama3.1` fallback) |
| Retrieval config | `RETRIEVAL_TOP_K=5`, `MIN_RELEVANCE_SCORE=0.45`, `MIN_BM25_SCORE=1.0`, `MAX_CONTEXT_CHARS=4000` |
| Dataset | `backend/evaluation/dataset.json` (v1.1-notes-format, 95 questions) |
| Results | `backend/evaluation/evaluation_results.json` |
| Stage-1 traces | `backend/evaluation/traces.json` (default for `--save-traces`) |
| Judge/embedding cache | `backend/.ragas_cache/<cache_namespace>/` — namespaced by judge model + judging-config hash + pipeline-config hash |
| Run fingerprints | `run.pipeline_fingerprint` and `run.judging_fingerprint` in `evaluation_results.json`; changed configs invalidate resume/cache reuse |
| **Full evaluation command** | `cd backend && python ../scripts/run_ragas_evaluation.py --concurrency 2` |
| Fast dev command | `cd backend && python ../scripts/run_ragas_evaluation.py --fast` |
| Stage 1 only | `python ../scripts/run_ragas_evaluation.py --stage traces --save-traces evaluation/traces.json` |
| Stage 2 only | `python ../scripts/run_ragas_evaluation.py --stage judge --traces evaluation/traces.json` |

Reproduce a run with the same configuration by using the defaults above; the
exact resolved configuration (including fingerprints, cache namespace, mode,
concurrency, strictness and question count) is embedded in the `run` section of
`evaluation_results.json` for every run. Re-running with a changed judge model,
answers, contexts, references or metric configuration is a different cache
namespace and a different fingerprint, so results cannot silently mix.
