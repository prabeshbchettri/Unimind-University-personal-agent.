"""Sample Phase 6 demo: dense vs hybrid retrieval comparison.

Builds a small corpus that includes exact-match cases (subject code, regulation
number, date, academic year, past-question number) and prints, for each
representative query, the dense-only top results and the hybrid (dense + BM25 +
RRF) top results — including the retrieval method and score of each result.

Offline-safe: uses the deterministic embedder, in-memory Qdrant and the
in-memory BM25 index. Run from the ``backend`` directory:

    python scripts/sample_hybrid.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import SemanticChunker
from app.embedding import DeterministicEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.normal_retriever import NormalRetriever
from app.retrieval.sparse import BM25Index
from app.services.vector_indexing import VectorIndexingService
from app.structure.models import DocumentMetadata, DocumentStructure, StructuredDocument
from app.ingestion.models import NormalizedDocument, PageContent

COLLECTIONS = ["university_docs", "past_questions", "library_books"]


def _structured(text: str, document_type: str, title: str, document_id: str) -> StructuredDocument:
    pages = [PageContent(page_number=1, text=text)]
    return StructuredDocument(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        extraction_method="pymupdf",
        page_count=1,
        pages=pages,
        metadata=DocumentMetadata(document_type=document_type, title=title, subject="General"),
        structure=DocumentStructure(topics=[]),
    )


CORPUS = [
    _structured(
        "Unit 1: Introduction\nDatabases store data.\nUnit 2: Normalization\nNormalization reduces "
        "data redundancy by decomposing tables into well-structured relations.",
        "syllabus",
        "DBMS Syllabus",
        "doc-syllabus",
    ),
    _structured(
        "Rules and Regulations\nRegulation 12\nStudents must attend at least 75 percent of classes "
        "to sit the final exam.",
        "rules_regulations",
        "Rules and Regulations",
        "doc-reg",
    ),
    _structured(
        "Course Code CSIT 325\nThis course introduces advanced databases.",
        "syllabus",
        "CSIT 325 Syllabus",
        "doc-csit",
    ),
    _structured(
        "Notice dated 2024-05-01\nClasses are cancelled on 2024-05-01.",
        "notice",
        "May Notice",
        "doc-notice",
    ),
    _structured(
        "2080\nQ5. Explain normalization with an example. 10 Marks\nNormalization is the process of "
        "organizing data to minimize redundancy.",
        "past_question",
        "DBMS 2080",
        "doc-past",
    ),
]

QUERIES = [
    ("Semantic question", "Explain normalization."),
    ("Exact subject code", "What is CSIT 325 about?"),
    ("Regulation number", "What does regulation 12 say about attendance?"),
    ("Date", "What happened on 2024-05-01?"),
    ("Academic year", "Which year past paper is 2080?"),
    ("Past question number", "What is question 5 about?"),
    ("Exact phrase", "normal forms 1NF 2NF 3NF"),
]


def _row(result, rank):
    return {
        "rank": rank,
        "score": round(result.score, 4),
        "method": result.method,
        "collection": result.collection,
        "title": result.metadata.get("title"),
        "topic": result.metadata.get("topic"),
        "document_id": result.metadata.get("document_id"),
        "text": result.text[:140],
    }


async def main() -> None:
    repository = __import__("app.repositories.qdrant", fromlist=["QdrantRepository"]).QdrantRepository(
        url=":memory:", collection_names=COLLECTIONS, default_dimensions=384
    )
    embedder = DeterministicEmbedder(384)
    service = VectorIndexingService(
        chunker=SemanticChunker(max_chars=80, overlap_chars=0),
        embedder=embedder,
        repository=repository,
        collection_names=COLLECTIONS,
        sparse_index=BM25Index(),
    )
    for structured in CORPUS:
        result = await service.index(structured)
        print("indexed:", json.dumps(result, ensure_ascii=False))

    dense = NormalRetriever(embedder=embedder, repository=repository, collection_names=COLLECTIONS, top_k=3)
    hybrid = HybridRetriever(dense_retriever=dense, sparse_index=service.sparse_index, top_k=3)

    print("\n=== DENSE vs HYBRID (correct source must appear in top-3) ===")
    expected = {
        "Explain normalization.": "DBMS Syllabus",
        "What is CSIT 325 about?": "CSIT 325 Syllabus",
        "What does regulation 12 say about attendance?": "Rules and Regulations",
        "What happened on 2024-05-01?": "May Notice",
        "Which year past paper is 2080?": "DBMS 2080",
        "What is question 5 about?": "DBMS 2080",
        "normal forms 1NF 2NF 3NF": "DBMS 2080",
    }
    for label, query in QUERIES:
        dense_results = await dense.retrieve(query, top_k=3)
        hybrid_results = await hybrid.retrieve(query, top_k=3)
        want = expected[query]
        dense_hit = any(r.metadata.get("title") == want for r in dense_results)
        hybrid_hit = any(r.metadata.get("title") == want for r in hybrid_results)
        dense_rank = next((i + 1 for i, r in enumerate(dense_results) if r.metadata.get("title") == want), None)
        hybrid_rank = next((i + 1 for i, r in enumerate(hybrid_results) if r.metadata.get("title") == want), None)
        print(f"\n[{label}] {query!r}  (expected: {want})")
        print("  dense : hit_in_top3=%s rank=%s %s" % (dense_hit, dense_rank, json.dumps([_row(r, i + 1) for i, r in enumerate(dense_results)], ensure_ascii=False)))
        print("  hybrid: hit_in_top3=%s rank=%s %s" % (hybrid_hit, hybrid_rank, json.dumps([_row(r, i + 1) for i, r in enumerate(hybrid_results)], ensure_ascii=False)))

    await repository.close()


if __name__ == "__main__":
    asyncio.run(main())