"""CLI entry point for the PDF ingestion / structure / indexing pipeline.

Run from the ``backend`` directory so the ``app`` package is importable:

    python scripts/ingest_pdf.py path/to/document.pdf [--summary] [--analyze] [--index]

Without ``--analyze``, prints the normalized document as JSON (``--summary``
shows a compact view). With ``--analyze``, runs document classification and
metadata/structure extraction and prints the structured document. With
``--index`` (implies ``--analyze``), the document is chunked, embedded and
indexed into Qdrant and a sample vector search is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure the backend directory (parent of scripts/) is importable regardless of
# how the script is invoked (python scripts/ingest_pdf.py or python -m ...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.embedding import build_embedder
from app.chunking import SemanticChunker
from app.ingestion import IngestionPipeline
from app.ingestion.errors import IngestionError
from app.repositories.qdrant import QdrantRepository
from app.structure import StructurePipeline
from app.services.vector_indexing import VectorIndexingService


def _build_indexing_service() -> VectorIndexingService:
    embedder = build_embedder(settings)
    repository = QdrantRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_names=settings.collection_names,
        default_dimensions=settings.embedding_dimensions,
    )
    return VectorIndexingService(
        chunker=SemanticChunker(
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
        embedder=embedder,
        repository=repository,
        collection_names=settings.collection_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF and print the normalized and/or structured document."
    )
    parser.add_argument("path", help="Path to the PDF file")
    parser.add_argument("--summary", action="store_true", help="Print a compact summary instead of full text")
    parser.add_argument("--analyze", action="store_true", help="Also classify and extract metadata/structure")
    parser.add_argument("--index", action="store_true", help="Also chunk, embed and index into Qdrant")
    args = parser.parse_args()

    ingestion = IngestionPipeline()
    structure = StructurePipeline()
    indexing: VectorIndexingService | None = None
    try:
        document = ingestion.ingest_file(args.path)
        structured = structure.process(document) if (args.analyze or args.index) else None
        if args.index:
            indexing = _build_indexing_service()
            index_result = asyncio.run(indexing.index(structured))
            sample = asyncio.run(indexing.search(structured.metadata.subject or document.filename, top_k=3))
    except IngestionError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    if args.index:
        print("=== INDEX RESULT ===")
        print(json.dumps(index_result, indent=2, ensure_ascii=False))
        print("\n=== SAMPLE SEARCH ===")
        print(json.dumps(
            [{"score": r.score, "text": r.text, "metadata": r.metadata} for r in sample],
            indent=2, ensure_ascii=False,
        ))
        return

    if args.analyze:
        print(json.dumps(structured.to_dict(), indent=2, ensure_ascii=False))
        return

    if args.summary:
        output = {
            "document_id": document.document_id,
            "filename": document.filename,
            "page_count": document.page_count,
            "extraction_method": document.extraction_method,
            "total_chars": document.total_chars(),
            "pages": [
                {"page_number": p.page_number, "chars": len(p.text)}
                for p in document.pages
            ],
            "metadata": document.metadata,
        }
    else:
        output = document.to_dict()

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()