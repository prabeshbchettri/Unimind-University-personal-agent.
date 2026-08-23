"""Sample Phase 4 demo: build sample documents, index them and run searches.

Produces the "sample indexed data" and "sample search result" artefacts for the
Phase 4 acceptance criteria. Uses the real classifier/extractor pipelines and
the deterministic embedder (no external services required), printing:

    1. the chunks produced per document (with metadata),
    2. an index summary per collection,
    3. sample search results for a few queries.

Run from the ``backend`` directory:

    python scripts/sample_search.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.embedding import build_embedder
from app.chunking import SemanticChunker
from app.ingestion.models import NormalizedDocument, PageContent
from app.repositories.qdrant import QdrantRepository
from app.structure import StructurePipeline
from app.services.vector_indexing import VectorIndexingService

SAMPLE_SYLLABUS = [
    "TRIBHUVAN UNIVERSITY\nB.Sc. CSIT\nSemester V\nDatabase Management System Syllabus\n"
    "Unit 1: Introduction to Databases\nDatabases are collections of related data. "
    "A database management system (DBMS) is software that stores, retrieves and "
    "manages data efficiently.\n"
    "Unit 2: Normalization\nNormalization reduces data redundancy and improves data "
    "integrity by decomposing tables into well-structured relations.\n"
    "Unit 3: Transaction Processing\nA transaction is a logical unit of work. "
    "ACID properties guarantee reliable processing of database transactions.",
]

SAMPLE_PAST_QUESTION = [
    "2080\nDatabase Management System\nTime: 3 Hrs.\nFull Marks: 60\n"
    "Q1. Define a database management system. 5 Marks\nDBMS is a software system "
    "that enables users to define, create, maintain and control access to the database.\n"
    "Q5. Explain normalization with an example. 10 Marks\nNormalization is the process "
    "of organizing data to minimize redundancy. The normal forms 1NF, 2NF and 3NF "
    "guide the decomposition of relations.",
]

SAMPLE_BOOK = [
    "Database Systems: Design, Implementation and Management\nby Carlos Coronel\n"
    "Contents\nPreface\n"
    "Chapter 1: Introduction to Databases\n1.1 Database Concepts\n1.2 Data Modeling\n"
    "Data modeling is the first step in database design and describes the structure "
    "of the data.\n"
    "Chapter 2: The Relational Model\n2.1 Tables\n2.2 Keys\n"
    "ISBN 978-1-337-62790-0",
]

QUERIES = [
    "What is normalization in database design?",
    "Define a database management system",
    "transaction processing ACID",
    "data modeling steps",
]


def _normalized(filename: str, pages: list[str], document_id: str) -> NormalizedDocument:
    return NormalizedDocument(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        extraction_method="pymupdf",
        pages=[PageContent(page_number=i + 1, text=text) for i, text in enumerate(pages)],
    )


async def main() -> None:
    pipeline = StructurePipeline()
    structured_docs = [
        pipeline.process(_normalized("dbms_syllabus.pdf", SAMPLE_SYLLABUS, "DOC-SYLLABUS-001")),
        pipeline.process(_normalized("dbms_2080.pdf", SAMPLE_PAST_QUESTION, "DOC-PQ-2080-001")),
        pipeline.process(_normalized("database_systems_book.pdf", SAMPLE_BOOK, "DOC-BOOK-001")),
    ]

    repository = QdrantRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_names=settings.collection_names,
        default_dimensions=settings.embedding_dimensions,
    )
    service = VectorIndexingService(
        # A smaller chunk size than the default is used so the demo shows
        # multiple context-preserving chunks per document.
        chunker=SemanticChunker(max_chars=350, overlap_chars=80),
        embedder=build_embedder(settings),
        repository=repository,
        collection_names=settings.collection_names,
    )

    print("=== SAMPLE INDEXED DATA (chunks) ===")
    all_indexed: list[dict] = []
    for structured in structured_docs:
        result = await service.index(structured)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        for collection in result["collections"]:
            count = await repository.count(collection)
            all_indexed.append({"collection": collection, "count": count})
    print("\n=== COLLECTION COUNTS ===")
    print(json.dumps(all_indexed, indent=2))

    print("\n=== SAMPLE SEARCH RESULTS ===")
    for query in QUERIES:
        results = await service.search(query, top_k=3)
        print(f"\nQuery: {query!r}")
        print(json.dumps(
            [
                {
                    "score": round(r.score, 4),
                    "collection": r.collection,
                    "text": r.text[:160],
                    "metadata": {
                        "document_id": r.metadata.get("document_id"),
                        "document_type": r.metadata.get("document_type"),
                        "page": r.metadata.get("page"),
                        "topic": r.metadata.get("topic"),
                        "subject": r.metadata.get("subject"),
                    },
                }
                for r in results
            ],
            indent=2,
            ensure_ascii=False,
        ))

    await repository.close()


if __name__ == "__main__":
    asyncio.run(main())