"""End-to-end ingestion pipeline and its CLI.

Pipeline position:  PDFs -> chunks -> embeddings -> Qdrant.

Run from ``backend/``::

    python -m app.ingest                 # ingest ../data/documents
    python -m app.ingest --recreate      # drop and rebuild the collection
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.embeddings import EmbeddingClient, EmbeddingError, build_embedding_client
from app.ingestion import Chunk, DocumentIngestionError, chunk_pdf, discover_pdfs
from app.vectorstore import QdrantVectorStore, VectorStoreError


@dataclass(frozen=True)
class IngestionReport:
    """Summary of one ingestion run."""

    documents: list[str]
    chunk_count: int
    vector_count: int
    collection: str
    collection_created: bool
    embedding_provider: str
    embedding_dim: int


def ingest_directory(
    documents_dir: Path,
    settings: Settings,
    embedding: EmbeddingClient | None = None,
    store: QdrantVectorStore | None = None,
    recreate: bool = False,
) -> IngestionReport:
    """Ingest every PDF in ``documents_dir`` into the configured Qdrant collection.

    ``embedding`` and ``store`` can be injected for testing; by default they are
    built from ``settings``.
    """
    pdf_paths = discover_pdfs(documents_dir)

    chunks: list[Chunk] = []
    for pdf_path in pdf_paths:
        chunks.extend(chunk_pdf(pdf_path, settings.chunk_size, settings.chunk_overlap))

    embedding = embedding or build_embedding_client(settings)
    if embedding.dim != settings.embedding_dim:
        raise EmbeddingError(
            f"EMBEDDING_DIM is {settings.embedding_dim} but the embedding client "
            f"'{embedding.name}' produces {embedding.dim}-dimensional vectors. "
            "Update EMBEDDING_DIM in the environment."
        )
    vectors = embedding.embed_texts([chunk.text for chunk in chunks])

    store = store or QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=embedding.dim,
    )
    if recreate:
        store.reset()
    collection_created = store.ensure_collection()
    vector_count = store.upsert_chunks(chunks, vectors)

    return IngestionReport(
        documents=[path.name for path in pdf_paths],
        chunk_count=len(chunks),
        vector_count=vector_count,
        collection=settings.qdrant_collection,
        collection_created=collection_created,
        embedding_provider=embedding.name,
        embedding_dim=embedding.dim,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Ingest university PDFs into Qdrant")
    parser.add_argument(
        "--documents",
        default=None,
        help="Documents directory (default: DOCUMENTS_DIR from the environment)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the existing collection before ingesting",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    documents_dir = Path(args.documents) if args.documents else Path(settings.documents_dir)

    try:
        report = ingest_directory(
            documents_dir=documents_dir, settings=settings, recreate=args.recreate
        )
    except (DocumentIngestionError, EmbeddingError, VectorStoreError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Embedded by      : {report.embedding_provider} (dim={report.embedding_dim})")
    print(f"Documents read   : {len(report.documents)}")
    for name in report.documents:
        print(f"  - {name}")
    print(f"Chunks created   : {report.chunk_count}")
    print(f"Vectors stored   : {report.vector_count}")
    print(
        f"Collection       : {report.collection} "
        f"({'created' if report.collection_created else 'already existed'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
