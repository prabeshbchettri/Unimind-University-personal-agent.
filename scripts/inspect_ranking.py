"""Show the retrieved source ranking for a query, before and after intent ranking.

Prints the raw vector/BM25/RRF order and the order after the document-type
intent adjustment (:mod:`app.document_intent`), with each chunk's available
scores and any intent boost. No LLM is called.

Usage (from the repository root):

    python scripts/inspect_ranking.py "syllabus of the software engineering"
    python scripts/inspect_ranking.py "syllabus of the software engineering" "software engineering chapter 1"

The Qdrant embedded engine takes an exclusive file lock, so the API server must
not be running.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.bm25 import build_bm25_index  # noqa: E402
from app.config import Settings  # noqa: E402
from app.document_intent import apply_intent_ranking, detect_syllabus_intent  # noqa: E402
from app.embeddings import build_embedding_client  # noqa: E402
from app.retriever import HybridRetriever, VectorRetriever  # noqa: E402
from app.routing import STRATEGY_NORMAL, QueryRouter  # noqa: E402
from app.vectorstore import QdrantVectorStore  # noqa: E402

DEFAULT_QUERIES = [
    "syllabus of the software engineering",
    "software engineering syllabus",
    "software engineering chapter 1",
    "explain software testing in software engineering",
    "syllabus of computer networks",
]


def _fmt(chunk, rank: int, *, boost: float = 0.0) -> str:
    semantic = "-" if chunk.semantic_score is None else f"{chunk.semantic_score:.3f}"
    bm25 = "-" if chunk.bm25_score is None else f"{chunk.bm25_score:.3f}"
    base = 1.0 / (rank + 1)
    return (
        f"  {rank + 1}. {chunk.document} · p{chunk.page}"
        f"   sem={semantic} bm25={bm25} intent_boost={boost:+.1f} final={base + boost:.3f}"
    )


def main() -> int:
    queries = sys.argv[1:] or DEFAULT_QUERIES
    settings = Settings(_env_file=str(_ROOT / "backend" / ".env"))
    embedding = build_embedding_client(settings)
    store = QdrantVectorStore(
        url=settings.qdrant_url, collection=settings.qdrant_collection, dim=embedding.dim
    )
    top_k = settings.retrieval_top_k
    vector = VectorRetriever(store=store, embedding=embedding, top_k=top_k)
    hybrid = HybridRetriever(vector=vector, bm25=build_bm25_index(store), top_k=top_k)
    router = QueryRouter()

    for query in queries:
        decision = router.route(query)
        retriever = vector if decision.strategy == STRATEGY_NORMAL else hybrid
        before = retriever.retrieve(query)
        after = apply_intent_ranking(query, before)

        print("=" * 78)
        print(f"Query:     {query}")
        print(f"Strategy:  {decision.strategy}   top_k={top_k}   "
              f"syllabus_intent={detect_syllabus_intent(query)}")
        print("BEFORE:")
        for rank, chunk in enumerate(before):
            print(_fmt(chunk, rank))
        print("AFTER:")
        for rank, chunk in enumerate(after):
            print(_fmt(chunk, rank, boost=chunk.intent_boost))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
