"""Evaluate the real Adaptive RAG pipeline with official RAGAS metrics.

This runner measures the system **as it exists**: for every question in the
existing evaluation dataset it calls the same in-process pipeline the API
server uses (router -> retrieval -> context/gate -> LLM gateway), captures the
retrieved contexts and the generated answer, and scores them with the official
RAGAS 0.4.x *collections* metrics. Nothing is mocked and the retrieval behavior
is not changed for the sake of the numbers.

Metrics (official RAGAS only -- see ``evaluation_metric.md`` for the rationale):
``Faithfulness``, ``AnswerRelevancy``, ``ContextRecall``, ``ContextPrecision``.

Two-stage workflow
------------------
The expensive parts are separated so RAGAS settings can be iterated without
re-running the production RAG pipeline:

- **Stage 1 (traces)**: run the real pipeline over the dataset and save the
  per-question traces (question, generated answer, retrieved contexts, sources,
  evidence flag, strategy, provider, latency, derived reference).
    ``python ../scripts/run_ragas_evaluation.py --stage traces --save-traces evaluation/traces.json``
- **Stage 2 (judge)**: score saved traces with RAGAS (no pipeline run).
    ``python ../scripts/run_ragas_evaluation.py --stage judge --traces evaluation/traces.json``

``--stage all`` (default) runs both, i.e. the complete end-to-end evaluation.

Reference handling
------------------
``ContextRecall`` and ``ContextPrecision`` require a RAGAS ``reference``. This
dataset has no ``reference_answer`` field, so a **derived reference** is built at
runtime from ``answer_must_mention_any`` (dataset file never modified; the
derivation is stored per question under ``reference_derived``).

Unanswerable questions (``insufficient_evidence_expected == true``) have no
ground truth: they are excluded from the four metrics and evaluated separately
for correct abstention (``abstention_correct``).

Performance notes (see ``evaluation_metric.md`` for the benchmark table)
------------------------------------------------------------------------
- Judge = local Ollama via an OpenAI-compatible async client; no paid API.
- Judge prompts/embeddings are cached on disk, namespaced by judge model and
  evaluation configuration, so results from different models/configs never mix.
- Bounded concurrency (``--concurrency``) caps in-flight judge requests; metric
  calls within a question run concurrently, and questions run concurrently.
- Traces and results are checkpointed atomically and runs are resumable.
- ``--fast`` runs a clearly-labelled representative subset for development.

The Qdrant embedded engine takes an exclusive file lock, so the API server must
not be running while Stage 1 executes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Windows consoles default to cp1252; model output can contain any Unicode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings  # noqa: E402
from app.rag import INSUFFICIENT_EVIDENCE_ANSWER, answer_question  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
DATASET = _ROOT / "backend" / "evaluation" / "dataset.json"
DEFAULT_OUT = _ROOT / "backend" / "evaluation" / "evaluation_results.json"
DEFAULT_TRACES = _ROOT / "backend" / "evaluation" / "traces.json"
CACHE_ROOT = _ROOT / "backend" / ".ragas_cache"

#: Metric keys, in report order. Values are the official RAGAS collection metric
#: names lower-cased for a stable JSON schema. The four metrics are fixed.
METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")

#: Fields captured by Stage 1 and reused by Stage 2. Everything a metric needs.
TRACE_FIELDS = (
    "id", "course", "category", "difficulty", "answerable", "question",
    "expected_strategy", "generated_answer", "retrieved_contexts",
    "retrieved_context_count", "sources", "enough_evidence", "declined",
    "strategy", "provider", "model", "latency_ms", "reference_derived",
    "answer_must_mention_any", "pipeline_error",
)


# --------------------------------------------------------------------------- #
# Caching (namespaced so different models/configs never share entries)
# --------------------------------------------------------------------------- #
def _fingerprint(data: dict) -> str:
    """Short stable hash of a configuration dict."""
    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def pipeline_fingerprint(settings) -> str:
    """Hash of everything that can change Stage-1 pipeline outputs."""
    return _fingerprint(
        {
            "generation_provider": settings.llm_provider,
            "groq_model": settings.groq_model,
            "ollama_model": settings.ollama_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "min_relevance_score": settings.min_relevance_score,
            "min_bm25_score": settings.min_bm25_score,
            "max_context_chars": settings.max_context_chars,
            "qdrant_collection": settings.qdrant_collection,
            "embedding_model": settings.embedding_model,
        }
    )


def judging_fingerprint(judge_model: str, mode: str, strictness: int) -> str:
    """Hash of everything that can change the RAGAS scores for fixed traces."""
    return _fingerprint(
        {
            "ragas_version": _ragas_version(),
            "judge_model": judge_model,
            "context_precision_mode": mode,
            "answer_relevancy_strictness": strictness,
            "metrics": list(METRIC_KEYS),
        }
    )


def _cache_dir_for(namespace: str, enabled: bool) -> Path | None:
    """Namespaced cache directory, or None when caching is disabled."""
    if not enabled:
        return None
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", namespace)
    return CACHE_ROOT / safe


# --------------------------------------------------------------------------- #
# RAGAS judge / embeddings wiring
# --------------------------------------------------------------------------- #
def _build_judge(model: str, base_url: str, cache_dir: Path | None, request_timeout: float):
    """Build the RAGAS judge LLM and a live judge-call counter.

    RAGAS 0.4.x collection metrics call ``llm.agenerate`` internally, so the
    judge needs an *async* client. Ollama exposes an OpenAI-compatible chat API
    at ``/v1``, so the same ``llm_factory`` path works for a local judge with no
    paid service.

    Returns ``(llm, counter)`` where ``counter`` is a mutable dict whose
    ``calls`` key counts real HTTP requests to the judge (cache hits do not
    count). The counter is best-effort: if the client cannot be instrumented it
    simply stays at 0.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(base_url=base_url, api_key="ollama", timeout=request_timeout)
    counter = {"calls": 0}
    try:
        original = client.chat.completions.create

        async def _counted_create(*args, **kwargs):
            counter["calls"] += 1
            return await original(*args, **kwargs)

        client.chat.completions.create = _counted_create  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001 -- counting is observability, never fatal
        pass

    kwargs: dict = {"client": client}
    if cache_dir is not None:
        from ragas.cache import DiskCacheBackend

        kwargs["cache"] = DiskCacheBackend(cache_dir=str(cache_dir))
    return llm_factory(model, **kwargs), counter


def _make_embedding_adapter(client, cache_dir: Path | None):
    """RAGAS embedding adapter around the project's own embedding client.

    Answer Relevancy embeds generated questions and the original question to
    measure cosine similarity. Reusing the local fastembed model keeps
    evaluation consistent with retrieval and avoids any external embedding API.
    The cache (when enabled) means identical texts are embedded once.
    """
    from ragas.embeddings.base import BaseRagasEmbedding

    cache = None
    if cache_dir is not None:
        from ragas.cache import DiskCacheBackend

        cache = DiskCacheBackend(cache_dir=str(cache_dir / "embeddings"))

    class LocalEmbeddingRagas(BaseRagasEmbedding):
        def __init__(self, embed_client) -> None:
            super().__init__(cache=cache)  # wraps embed_text/aembed_text with the cache
            self._client = embed_client

        def embed_text(self, text: str, **kwargs) -> list[float]:
            return [float(value) for value in self._client.embed_query(text)]

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            # fastembed is CPU-bound and synchronous; run it off the event loop.
            return await asyncio.to_thread(self.embed_text, text)

        def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
            return await asyncio.to_thread(
                lambda: [self.embed_text(text) for text in texts]
            )

    return LocalEmbeddingRagas(client)


def _build_metrics(judge, embeddings, *, strictness: int, context_precision_mode: str):
    """Instantiate the selected official RAGAS collections metrics."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextPrecisionWithoutReference,
        ContextRecall,
        Faithfulness,
    )

    precision_cls = (
        ContextPrecisionWithoutReference
        if context_precision_mode == "without-reference"
        else ContextPrecision
    )
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge, embeddings=embeddings, strictness=strictness
        ),
        "context_recall": ContextRecall(llm=judge),
        "context_precision": precision_cls(llm=judge),
    }


# --------------------------------------------------------------------------- #
# Pipeline execution + reference derivation
# --------------------------------------------------------------------------- #
def build_pipeline(settings):
    """Build the same shared objects the API server uses (see app/main.py)."""
    from app.bm25 import build_bm25_index
    from app.embeddings import build_embedding_client
    from app.llm import build_llm_client
    from app.retriever import HybridRetriever, VectorRetriever
    from app.routing import QueryRouter
    from app.vectorstore import QdrantVectorStore

    embedding = build_embedding_client(settings)
    store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=embedding.dim,
    )
    vector_retriever = VectorRetriever(
        store=store, embedding=embedding, top_k=settings.retrieval_top_k
    )
    return {
        "embedding": embedding,
        "retriever": vector_retriever,
        "hybrid_retriever": HybridRetriever(
            vector=vector_retriever,
            bm25=build_bm25_index(store),
            top_k=settings.retrieval_top_k,
        ),
        "llm": build_llm_client(settings),
        "router": QueryRouter(),
    }


def derive_reference(case: dict) -> str | None:
    """Build a RAGAS reference from the dataset's own required-information list.

    The dataset has no ``reference_answer`` field; its ground-truth signal is
    ``answer_must_mention_any`` (the facts a correct answer must contain). We
    turn that into a short, decomposable reference the RAGAS judge can reason
    about. Returns ``None`` when there is nothing to derive from.
    """
    keywords = case.get("answer_must_mention_any") or []
    if not keywords:
        return None
    joined = "; ".join(str(item) for item in keywords)
    return f"A correct answer must include the following information: {joined}."


def run_pipeline(settings, pipeline, case: dict) -> dict:
    """Run one dataset question through the real pipeline and capture outputs."""
    question = case["question"]
    started = time.perf_counter()
    answer = ""
    retrieved_texts: list[str] = []
    sources: list[dict] = []
    enough_evidence = False
    strategy = "error"
    provider = "none"
    model = "none"
    error: str | None = None
    try:
        result = answer_question(
            question,
            settings=settings,
            retriever=pipeline["retriever"],
            llm=pipeline["llm"],
            router=pipeline["router"],
            hybrid_retriever=pipeline["hybrid_retriever"],
        )
        answer = result.answer
        retrieved_texts = [chunk.text for chunk in result.retrieved]
        sources = [{"document": s.document, "page": s.page} for s in result.sources]
        enough_evidence = result.enough_evidence
        strategy = result.strategy
        provider = result.provider
        model = result.model
    except Exception as exc:  # noqa: BLE001 -- record and continue the batch
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "id": case["id"],
        "course": case["course"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "answerable": not case["insufficient_evidence_expected"],
        "question": question,
        "expected_strategy": case.get("expected_strategy"),
        "generated_answer": answer,
        "retrieved_contexts": retrieved_texts,
        "retrieved_context_count": len(retrieved_texts),
        "sources": sources,
        "enough_evidence": enough_evidence,
        "declined": (not enough_evidence) or (answer.strip() == INSUFFICIENT_EVIDENCE_ANSWER),
        "strategy": strategy,
        "provider": provider,
        "model": model,
        "latency_ms": round(latency_ms),
        "reference_derived": derive_reference(case),
        "answer_must_mention_any": case.get("answer_must_mention_any") or [],
        "pipeline_error": error,
    }


# --------------------------------------------------------------------------- #
# RAGAS scoring (concurrent)
# --------------------------------------------------------------------------- #
def _clean_value(result) -> float | None:
    """Extract a finite float from a RAGAS MetricResult (NaN/None -> None)."""
    value = getattr(result, "value", None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


async def _run_metric(key, coro, limiter, scores, errors, fatal):
    """Await one metric call under the global limiter, recording the outcome."""
    async with limiter:
        try:
            scores[key] = _clean_value(await coro)
        except Exception as exc:  # noqa: BLE001 -- one metric must not sink the run
            errors[key] = f"{type(exc).__name__}: {exc}"
            fatal.append(key)


async def score_record(record: dict, metrics: dict, mode: str, limiter: asyncio.Semaphore) -> None:
    """Score one pipeline record in place with the selected RAGAS metrics.

    The four metric calls are independent and run **concurrently** (each still
    bounded by the shared ``limiter``), so a single question no longer pays the
    sum of four judge round-trips. Unanswerable questions are scored only for
    abstention (no ground truth), recorded under ``abstention_correct``.
    """
    record["scores"] = {key: None for key in METRIC_KEYS}
    record["metric_errors"] = {}
    record["scored"] = False
    if record.get("pipeline_error"):
        return

    if not record["answerable"]:
        # No ground truth exists: the only correct behavior is to abstain.
        record["abstention_correct"] = bool(record["declined"])
        record["scored"] = True
        return

    question = record["question"]
    answer = record["generated_answer"]
    contexts = record["retrieved_contexts"]
    reference = record["reference_derived"]
    errors: dict[str, str] = {}
    fatal: list[str] = []
    jobs = []

    if answer and contexts:
        jobs.append(
            ("faithfulness", metrics["faithfulness"].ascore(
                user_input=question, response=answer, retrieved_contexts=contexts))
        )
    elif not contexts:
        errors["faithfulness"] = "no retrieved contexts"

    if answer:
        jobs.append(
            ("answer_relevancy", metrics["answer_relevancy"].ascore(
                user_input=question, response=answer))
        )

    if reference and contexts:
        jobs.append(
            ("context_recall", metrics["context_recall"].ascore(
                user_input=question, retrieved_contexts=contexts, reference=reference))
        )
        if mode == "without-reference":
            jobs.append(
                ("context_precision", metrics["context_precision"].ascore(
                    user_input=question, response=answer, retrieved_contexts=contexts))
            )
        else:
            jobs.append(
                ("context_precision", metrics["context_precision"].ascore(
                    user_input=question, reference=reference, retrieved_contexts=contexts))
            )
    elif not reference:
        errors["context_recall"] = "no derivable reference"
    elif not contexts:
        errors["context_recall"] = "no retrieved contexts"

    await asyncio.gather(
        *(_run_metric(key, coro, limiter, record["scores"], errors, fatal)
          for key, coro in jobs)
    )
    record["metric_errors"] = errors
    # A record counts as fully scored only if no judge call raised. Deterministic
    # skips (no contexts / no reference) still count as scored so resume does not
    # retry them forever.
    record["scored"] = not fatal


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _metric_block(records: list[dict]) -> dict:
    """Mean + coverage per metric for a group of records."""
    block: dict = {}
    for key in METRIC_KEYS:
        values = [
            r.get("scores", {}).get(key)
            for r in records
            if r.get("scores", {}).get(key) is not None
        ]
        block[key] = _mean(values)
        block[f"{key}_n"] = len(values)
    return block


def _group_by(records: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get(field)), []).append(record)
    return {name: _metric_block(rows) for name, rows in sorted(groups.items())}


def aggregate(records: list[dict]) -> dict:
    """Overall, per-course, per-category and per-difficulty aggregate scores."""
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    abstention = {
        "unanswerable_questions": len(unanswerable),
        "correct_abstentions": sum(1 for r in unanswerable if r.get("abstention_correct")),
        "hallucinated_answers": sum(
            1 for r in unanswerable if r.get("abstention_correct") is False
        ),
        "answerable_questions": len(answerable),
        "false_abstentions": sum(1 for r in answerable if r.get("declined")),
    }

    return {
        "overall": _metric_block(answerable),
        "by_course": _group_by(answerable, "course"),
        "by_category": _group_by(answerable, "category"),
        "by_difficulty": _group_by(answerable, "difficulty"),
        "abstention": abstention,
    }


def collect_failures(records: list[dict]) -> list[dict]:
    """Per-question error/failure rows, for the ``failures`` section."""
    failures = []
    for record in records:
        problems = []
        if record.get("pipeline_error"):
            problems.append(f"pipeline: {record['pipeline_error']}")
        for key, message in record.get("metric_errors", {}).items():
            problems.append(f"{key}: {message}")
        if not record["answerable"] and record.get("abstention_correct") is False:
            problems.append("unanswerable question was answered (possible hallucination)")
        if problems:
            failures.append({"id": record["id"], "course": record["course"], "problems": problems})
    return failures


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _is_answerable(row: dict) -> bool:
    """Answerability of a dataset case or a trace record (both shapes)."""
    if "insufficient_evidence_expected" in row:
        return not row["insufficient_evidence_expected"]
    return bool(row.get("answerable", True))


def _select_cases(cases: list[dict], args) -> list[dict]:
    """Apply the CLI filters, in a fixed, documented order.

    Works on both dataset cases and Stage-1 trace records, so the same filters
    apply whether the pipeline runs or saved traces are judged.
    """
    selected = cases
    if getattr(args, "ids", ""):
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        selected = [c for c in selected if c["id"] in wanted]
    if getattr(args, "course", ""):
        selected = [c for c in selected if c["course"] == args.course]
    if getattr(args, "category", ""):
        selected = [c for c in selected if c["category"] == args.category]
    if getattr(args, "difficulty", ""):
        selected = [c for c in selected if c["difficulty"] == args.difficulty]
    if getattr(args, "answerable_only", False):
        selected = [c for c in selected if _is_answerable(c)]
    if getattr(args, "unanswerable_only", False):
        selected = [c for c in selected if not _is_answerable(c)]
    if getattr(args, "limit", 0):
        selected = selected[: args.limit]
    return selected


def select_fast_subset(cases: list[dict], answerable_size: int = 10) -> list[dict]:
    """Deterministic representative subset for ``--fast``.

    Answerable questions are sampled round-robin across (course, difficulty)
    strata so every course and difficulty is represented; **all** unanswerable
    questions are included because they make no judge calls and their abstention
    evaluation is free. Selection is deterministic (stable across runs) so
    fast-mode runs are comparable.
    """
    answerable = [c for c in cases if _is_answerable(c)]
    unanswerable = [c for c in cases if not _is_answerable(c)]

    groups: dict[tuple, list[dict]] = {}
    for case in answerable:
        groups.setdefault((case["course"], case["difficulty"]), []).append(case)

    keys = sorted(groups)
    selected: list[dict] = []
    index = 0
    while len(selected) < answerable_size and any(groups[k] for k in keys):
        key = keys[index % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop(0))
        index += 1
    return selected + unanswerable


# --------------------------------------------------------------------------- #
# Traces / results I/O
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, payload: dict) -> None:
    """Write JSON atomically (temp file + replace) so an interrupt never corrupts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_traces(path: Path, meta: dict, records: list[dict]) -> None:
    """Stage-1 output: pipeline traces without any RAGAS scores."""
    traces = [ {k: r[k] for k in TRACE_FIELDS if k in r} for r in records ]
    _atomic_write(path, {
        "traces_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **meta,
        "records": traces,
    })


def load_traces(path: Path) -> tuple[dict, list[dict]]:
    """Load a Stage-1 traces file; returns (meta, records)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    return {k: v for k, v in data.items() if k != "records"}, records


def _load_existing_results(path: Path) -> dict | None:
    """Load a previous results/checkpoint file, or None when absent/invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Scoring driver
# --------------------------------------------------------------------------- #
async def _score_all(records, metrics, mode, concurrency, checkpoint=None) -> None:
    """Score every record with a single global concurrency limiter.

    The limiter bounds **in-flight judge requests** across all questions and all
    metric calls, so concurrency scales with the machine rather than the number
    of questions. Records are checkpointed as they complete.
    """
    limiter = asyncio.Semaphore(max(1, concurrency))

    async def worker(record):
        await score_record(record, metrics, mode, limiter)

    tasks = [asyncio.create_task(worker(r)) for r in records]
    done = 0
    for task in asyncio.as_completed(tasks):
        await task
        done += 1
        if checkpoint is not None:
            checkpoint(done)


def _estimate_judge_calls(records: list[dict], strictness: int) -> int:
    """Upper-bound estimate of judge calls for reporting (not a guarantee)."""
    total = 0
    for record in records:
        if not record["answerable"]:
            continue
        contexts = len(record.get("retrieved_contexts") or [])
        if record.get("generated_answer") and contexts:
            total += 2  # Faithfulness: statements + NLI verdict
        if record.get("generated_answer"):
            total += max(1, strictness)  # Answer Relevancy: question generation
        if record.get("reference_derived") and contexts:
            total += 1  # Context Recall
            total += contexts  # Context Precision: one verdict per context
    return total


def _ragas_version() -> str:
    try:
        import ragas

        return getattr(ragas, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON results path")
    parser.add_argument("--stage", choices=["all", "traces", "judge"], default="all",
                        help="all = run pipeline and judge; traces = pipeline only; judge = judge saved traces")
    parser.add_argument("--save-traces", default=None, help="write Stage-1 traces to this path")
    parser.add_argument("--traces", default=None, help="read Stage-1 traces from this path (skips the pipeline)")
    parser.add_argument("--fast", action="store_true",
                        help="FAST/DEVELOPMENT mode: representative subset, lower strictness")
    parser.add_argument("--fast-size", type=int, default=10,
                        help="answerable questions in the --fast subset (all unanswerable are always included)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    parser.add_argument("--ids", default="", help="comma-separated question ids to evaluate")
    parser.add_argument("--course", default="", help="evaluate only one course")
    parser.add_argument("--category", default="", help="evaluate only one question category")
    parser.add_argument("--difficulty", default="", help="evaluate only one difficulty (easy/medium/hard)")
    parser.add_argument("--answerable-only", action="store_true", help="only answerable questions")
    parser.add_argument("--unanswerable-only", action="store_true", help="only unanswerable questions")
    parser.add_argument("--judge-model", default=None,
                        help="Ollama model acting as the RAGAS judge (default: OLLAMA_MODEL)")
    parser.add_argument("--judge-base-url", default=None,
                        help="OpenAI-compatible base URL for the judge (default: OLLAMA_BASE_URL + /v1)")
    parser.add_argument("--context-precision", choices=["with-reference", "without-reference"],
                        default="with-reference",
                        help="official RAGAS variant: reference-based (default) or reference-free")
    parser.add_argument("--answer-relevancy-strictness", type=int, default=None,
                        help="questions AnswerRelevancy generates per answer (default 3; --fast default 1)")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="max in-flight judge requests (benchmark recommends 2 on a 4GB GPU)")
    parser.add_argument("--judge-timeout", type=float, default=300.0, help="judge request timeout (s)")
    parser.add_argument("--no-cache", action="store_true", help="disable the judge/embedding disk cache")
    parser.add_argument("--no-resume", action="store_true", help="ignore previous results/traces and recompute")
    parser.add_argument("--run-notes", default="", help="free-form label stored in the run metadata")
    return parser


def _resolve_selection(dataset_cases: list[dict], args) -> list[dict]:
    """Combine --fast with the explicit filters into the final question set."""
    cases = dataset_cases
    if args.fast:
        cases = select_fast_subset(cases, args.fast_size)
    cases = _select_cases(cases, args)
    return cases


def _run_stage1(settings, cases, args, existing_by_id: dict) -> list[dict]:
    """Run the RAG pipeline over the selected cases, reusing matching traces."""
    records: list[dict] = []
    to_run = []
    for case in cases:
        prior = existing_by_id.get(case["id"])
        if prior is not None and not args.no_resume:
            # Reuse captured pipeline output; the pipeline fingerprint already
            # matched (checked by the caller) so this is the same configuration.
            reused = {k: prior.get(k) for k in TRACE_FIELDS}
            reused.setdefault("course", case["course"])
            records.append(reused)
        else:
            to_run.append(case)

    if to_run:
        print(f"Building pipeline (provider={settings.llm_provider}, store={settings.qdrant_url})...")
        pipeline = build_pipeline(settings)
        print(f"Running {len(to_run)} question(s) through the real RAG pipeline...")
        for index, case in enumerate(to_run, start=1):
            record = run_pipeline(settings, pipeline, case)
            status = "declined" if record["declined"] else f"{record['strategy']}/{record['provider']}"
            print(f"  [{index:>3}/{len(to_run)}] {record['id']:<10} {status:<22} "
                  f"{record['retrieved_context_count']} ctx, {record['latency_ms']} ms")
            records.append(record)
    if len(records) != len(cases):
        # Keep the requested order.
        order = {c["id"]: i for i, c in enumerate(cases)}
        records.sort(key=lambda r: order.get(r["id"], 1_000_000))
    return records


def main() -> int:
    args = _build_parser().parse_args()

    try:
        import ragas  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"RAGAS is not importable: {exc}\nInstall it with: pip install ragas==0.4.3")
        return 2

    get_settings.cache_clear()
    settings = get_settings()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = _resolve_selection(dataset["questions"], args)
    if not cases:
        print("No questions matched the filters; nothing to do.")
        return 1

    judge_model = args.judge_model or settings.ollama_model
    judge_base_url = args.judge_base_url or (settings.ollama_base_url.rstrip("/") + "/v1")
    strictness = args.answer_relevancy_strictness
    if strictness is None:
        strictness = 1 if args.fast else 3

    pipe_fp = pipeline_fingerprint(settings)
    judge_fp = judging_fingerprint(judge_model, args.context_precision, strictness)
    # Cache namespace: judge model + judging config + pipeline config.
    cache_namespace = f"{judge_model}-{judge_fp}-{pipe_fp[:8]}"
    out_path = Path(args.out)
    traces_path = Path(args.traces) if args.traces else (Path(args.save_traces) if args.save_traces else DEFAULT_TRACES)

    mode = "fast" if args.fast else "full"
    label = "FAST/DEVELOPMENT" if args.fast else "FULL"
    if args.stage == "judge":
        print(f"Stage:   judge saved traces [{label} mode]")
    else:
        print(f"Dataset: {DATASET.name} (v{dataset.get('version')}) -> {len(cases)} question(s) [{label} mode]")
    print(f"Judge:   {judge_model} @ {judge_base_url}  "
          f"(context_precision={args.context_precision}, strictness={strictness}, concurrency={args.concurrency})")

    # ---------------------------------------------------------------- selection
    records: list[dict] = []
    existing_by_id: dict[str, dict] = {}
    existing_scored: dict[str, dict] = {}

    # Reuse from a previous results file (resume) when compatible.
    previous = None if args.no_resume else _load_existing_results(out_path)
    if previous and previous.get("run", {}).get("pipeline_fingerprint") == pipe_fp:
        for record in previous.get("per_question", []):
            existing_by_id[record["id"]] = record
        if previous["run"].get("judging_fingerprint") == judge_fp:
            for record in previous.get("per_question", []):
                if record.get("scored"):
                    existing_scored[record["id"]] = record

    # ------------------------------------------------------------------ stage 1
    if args.stage in ("all", "traces"):
        if args.stage == "traces" and args.traces:
            print("--stage traces ignores --traces; running the pipeline instead.")
        records = _run_stage1(settings, cases, args, existing_by_id)
        if args.save_traces or args.stage == "traces":
            save_traces(traces_path, {
                "dataset": str(DATASET),
                "dataset_version": dataset.get("version"),
                "pipeline_fingerprint": pipe_fp,
            }, records)
            print(f"Saved {len(records)} trace(s) to {traces_path}")
        if args.stage == "traces":
            print("Stage 'traces' complete (no RAGAS judging performed).")
            return 0
    else:  # stage == "judge"
        if args.traces and Path(args.traces).exists():
            meta, records = load_traces(Path(args.traces))
            if meta.get("pipeline_fingerprint") and meta["pipeline_fingerprint"] != pipe_fp:
                print("WARNING: traces were produced with a different pipeline configuration; "
                      "RAGAS will score those traces as-is (pipeline fingerprint mismatch).")
            if args.fast:
                records = select_fast_subset(records, args.fast_size)
            records = _select_cases(records, args)
            if not records:
                print("No traces matched the filters; nothing to do.")
                return 1
            print(f"Loaded {len(records)} trace(s) from {args.traces}; pipeline will not run.")
        elif existing_by_id:
            records = [existing_by_id[c["id"]] for c in cases if c["id"] in existing_by_id]
            print(f"No --traces file; scoring {len(records)} question(s) from the previous results file.")
        else:
            print("Stage 'judge' needs --traces (or a compatible previous results file).")
            return 1

    # ------------------------------------------------------------------ stage 2
    from app.embeddings import build_embedding_client  # noqa: E402

    cache_dir = _cache_dir_for(cache_namespace, not args.no_cache)
    embeddings = _make_embedding_adapter(build_embedding_client(settings), cache_dir)
    judge, counter = _build_judge(judge_model, judge_base_url, cache_dir, args.judge_timeout)
    metrics = _build_metrics(
        judge, embeddings, strictness=strictness, context_precision_mode=args.context_precision
    )

    def base_payload(partial: bool = False, scored: int | None = None) -> dict:
        run_meta = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metric_framework": "RAGAS",
            "ragas_version": _ragas_version(),
            "mode": mode,
            "metrics": list(METRIC_KEYS),
            "context_precision_mode": args.context_precision,
            "answer_relevancy_strictness": strictness,
            "judge_provider": "ollama",
            "judge_model": judge_model,
            "judge_base_url": judge_base_url,
            "embedding_model": settings.embedding_model,
            "generation_provider": settings.llm_provider,
            "retrieval_top_k": settings.retrieval_top_k,
            "min_relevance_score": settings.min_relevance_score,
            "min_bm25_score": settings.min_bm25_score,
            "max_context_chars": settings.max_context_chars,
            "dataset": str(DATASET),
            "dataset_version": dataset.get("version"),
            "reference_source": "derived from answer_must_mention_any (dataset unmodified)",
            "pipeline_fingerprint": pipe_fp,
            "judging_fingerprint": judge_fp,
            "cache_namespace": cache_namespace if not args.no_cache else None,
            "concurrency": args.concurrency,
            "question_count": len(records),
            "estimated_judge_calls": _estimate_judge_calls(records, strictness),
            "judge_calls": counter["calls"],
            "run_notes": args.run_notes,
            "filters": {
                "fast": args.fast, "fast_size": args.fast_size, "limit": args.limit,
                "ids": args.ids, "course": args.course, "category": args.category,
                "difficulty": args.difficulty, "answerable_only": args.answerable_only,
                "unanswerable_only": args.unanswerable_only,
            },
        }
        if args.fast:
            run_meta["mode_warning"] = (
                "FAST/DEVELOPMENT run: representative subset with reduced AnswerRelevancy "
                "strictness. NOT equivalent to the full evaluation; do not report these "
                "numbers as full-evaluation results."
            )
        payload = {
            "run": run_meta,
            "overall": {}, "by_course": {}, "by_category": {}, "by_difficulty": {},
            "abstention": {}, "failures": [], "per_question": records,
        }
        if partial:
            payload["partial"] = True
            payload["run"]["scored"] = scored
        return payload

    def checkpoint(done):
        # Partial files carry the aggregates too, so an interrupted run leaves a
        # complete, usable artifact (only coverage counts differ).
        payload = base_payload(partial=True, scored=done)
        payload.update(aggregate(records))
        payload["failures"] = collect_failures(records)
        payload["run"]["scoring_seconds"] = None
        _atomic_write(out_path, payload)
        print(f"  checkpoint: {done}/{len(records)} scored -> {out_path}")

    # Resume: skip questions already scored under the same judging config.
    pending = [r for r in records if not (not args.no_resume and r.get("id") in existing_scored)]
    if len(pending) != len(records):
        print(f"Resuming: {len(records) - len(pending)} question(s) already scored, "
              f"{len(pending)} remaining.")
    # Seed reuse: copy scores from previous results into the current records.
    for record in records:
        prior = existing_scored.get(record["id"])
        if prior is not None and not args.no_resume:
            record["scores"] = prior.get("scores", {k: None for k in METRIC_KEYS})
            record["metric_errors"] = prior.get("metric_errors", {})
            if not record["answerable"]:
                record["abstention_correct"] = prior.get("abstention_correct")
            record["scored"] = True

    print(f"Scoring {len(pending)} question(s) with RAGAS ({', '.join(METRIC_KEYS)})...")
    started = time.perf_counter()
    asyncio.run(_score_all(pending, metrics, args.context_precision, args.concurrency, checkpoint))
    elapsed = time.perf_counter() - started

    payload = base_payload()
    payload.update(aggregate(records))
    payload["failures"] = collect_failures(records)
    payload["run"]["scoring_seconds"] = round(elapsed, 1)
    payload["run"]["judge_calls"] = counter["calls"]
    _atomic_write(out_path, payload)

    agg = aggregate(records)
    label = "FAST/DEVELOPMENT" if args.fast else "FULL"
    print(f"\n=== RAGAS RESULTS [{label}] (question-level means; answerable questions) ===")
    for key in METRIC_KEYS:
        print(f"  {key:<18} {agg['overall'][key]}  (n={agg['overall'][key + '_n']})")
    abst = agg["abstention"]
    print(f"\n  abstention: {abst['correct_abstentions']}/{abst['unanswerable_questions']} "
          f"unanswerable questions correctly declined; "
          f"{abst['hallucinated_answers']} hallucinated")
    print(f"  judge calls: {counter['calls']} | scoring {elapsed:.0f}s | results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
