"""Manual verification: similarity search over the ingested demo corpus.

Run from backend/ after `python -m app.ingest`:

    python ../scripts/verify_ingestion.py

Not part of the test suite; this is the phase's end-to-end check that vectors
persisted by a previous process are retrievable with relevant metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This script is launched by file path (``python ../scripts/verify_ingestion.py``),
# so Python puts scripts/ on sys.path instead of backend/. Add backend/ so the
# ``app`` package is importable.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.embeddings import build_embedding_client  # noqa: E402
from app.vectorstore import QdrantVectorStore  # noqa: E402

QUESTIONS = [
    "What is the minimum attendance requirement?",
    "What are the rules for examination eligibility?",
    "What does the syllabus say about CS201?",
    "How many books can I borrow from the library?",
    "Who won the football world cup?",  # out-of-corpus query
]


def main() -> None:
    settings = get_settings()
    embedding = build_embedding_client(settings)
    store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=embedding.dim,
    )

    print(f"provider={embedding.name} collection={settings.qdrant_collection}\n")
    for question in QUESTIONS:
        hits = store.search(query_vector=embedding.embed_query(question), top_k=2)
        print(f"Q: {question}")
        if not hits:
            print("  (no results)\n")
            continue
        for hit in hits:
            snippet = hit.text[:70] + ("…" if len(hit.text) > 70 else "")
            print(
                f"  [{hit.score:.3f}] {hit.document_name} p.{hit.page} "
                f"chunk {hit.chunk_index} :: {snippet}"
            )
        print()


if __name__ == "__main__":
    main()
