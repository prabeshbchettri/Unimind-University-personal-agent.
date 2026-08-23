"""Evaluation infrastructure (Phase 13).

Hermetic, reproducible evaluation of the complete system:

- :mod:`app.evaluation.corpus` — a labeled evaluation corpus spanning the
  seven document categories (syllabus, notices, rules/regulations, academic
  calendar, past questions, library books, external/current information).
- :mod:`app.evaluation.metrics` — retrieval metrics (Recall@K, Precision@K,
  MRR).
- :mod:`app.evaluation.environment` — the deterministic offline environment
  (in-memory Qdrant, deterministic embedder, stub LLM, in-memory graph,
  BM25) used for every evaluation run.
- :mod:`app.evaluation.retrieval` — Normal/Hybrid/Graph retrieval
  evaluation.
- :mod:`app.evaluation.answers` — answer evaluation (faithfulness,
  relevance, source/citation correctness, hallucination rate).
- :mod:`app.evaluation.recommendations` — library recommendation scenario
  evaluation.
- :mod:`app.evaluation.ocr` — digital vs scanned PDF extraction evaluation.

Run everything with ``python scripts/evaluate.py`` (backend directory).
"""

from app.evaluation.corpus import (
    build_corpus,
    retrieval_queries,
    DOCUMENTS,
    RETRIEVAL_QUERIES,
)
from app.evaluation.environment import EvaluationEnvironment

__all__ = [
    "EvaluationEnvironment",
    "build_corpus",
    "retrieval_queries",
    "DOCUMENTS",
    "RETRIEVAL_QUERIES",
]