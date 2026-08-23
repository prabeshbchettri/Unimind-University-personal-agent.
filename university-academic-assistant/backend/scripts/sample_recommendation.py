"""Phase 9 demo: syllabus-aware library book recommendation.

Builds a syllabus + three library books (strong / partial / poor coverage),
indexes them, runs the recommendation service for the acceptance queries and
prints the ranked books with evidence (matched/missing topics).

Run:  python scripts/sample_recommendation.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Self-contained demo: in-memory storage, deterministic embedder, stub LLM.
os.environ["QDRANT_URL"] = ":memory:"
os.environ["EMBEDDER_BACKEND"] = "deterministic"
os.environ["LLM_BACKEND"] = "stub"
os.environ["GRAPH_BACKEND"] = "memory"

from app.chunking import SemanticChunker  # noqa: E402
from app.config.settings import Settings  # noqa: E402
from app.embedding import build_embedder  # noqa: E402
from app.graph.extractor import GraphExtractor  # noqa: E402
from app.ingestion.cleaning import TextCleaner  # noqa: E402
from app.ingestion.models import NormalizedDocument, PageContent  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.repositories.graph import build_graph_repository  # noqa: E402
from app.repositories.qdrant import QdrantRepository  # noqa: E402
from app.retrieval.sparse import BM25Index  # noqa: E402
from app.services.graph import KnowledgeGraphService  # noqa: E402
from app.services.recommendation import RecommendationService  # noqa: E402
from app.services.vector_indexing import VectorIndexingService  # noqa: E402
from app.structure.extractors.deterministic import DeterministicMetadataExtractor  # noqa: E402
from app.structure.models import Chapter, DocumentMetadata, DocumentStructure, StructuredDocument  # noqa: E402

SYLLABUS_TEXT = (
    "Database Management System Syllabus\n"
    "Unit 1: Introduction\nDatabases store data.\n"
    "Unit 2: Normalization\nNormalization removes redundancy in tables.\n"
    "Unit 3: Transactions\nACID properties keep transactions safe.\n"
    "Unit 4: Indexing\nIndexes speed up queries.\n"
)

BOOKS = [
    {
        "title": "Database System Concepts",
        "chapters": [
            ("Introduction", ["Data Models"]),
            ("Normalization", ["Normal Forms", "First Normal Form"]),
            ("Transactions", ["ACID Properties"]),
            ("Indexing", ["B-Trees"]),
        ],
    },
    {
        "title": "Database Fundamentals",
        "chapters": [
            ("Introduction", []),
            ("Normalization", ["Normal Forms"]),
        ],
    },
    {
        "title": "Operating System Principles",
        "chapters": [
            ("Processes", ["Scheduling"]),
            ("Memory", ["Paging"]),
        ],
    },
]

QUERIES = [
    "Which book is best for normalization?",
    "Recommend a book for DBMS.",
    "Which library book covers most of my syllabus?",
]


def make_library_book(document_id: str, title: str, chapters: list[tuple[str, list[str]]]) -> StructuredDocument:
    lines: list[str] = []
    for index, (chapter_title, subtopics) in enumerate(chapters, start=1):
        lines.append(f"Chapter {index}: {chapter_title}")
        for sub_index, subtopic in enumerate(subtopics, start=1):
            lines.append(f"{index}.{sub_index} {subtopic}")
    return StructuredDocument(
        document_id=document_id,
        filename=f"{title}.pdf",
        extraction_method="pymupdf",
        page_count=1,
        pages=[PageContent(page_number=1, text="\n".join(lines))],
        metadata=DocumentMetadata(document_type="library_book", title=title),
        structure=DocumentStructure(
            topics=[chapter_title for chapter_title, _ in chapters],
            chapters=[
                Chapter(chapter_number=index, title=chapter_title, topics=[chapter_title], subtopics=subtopics)
                for index, (chapter_title, subtopics) in enumerate(chapters, start=1)
            ],
        ),
    )


def main() -> None:
    settings = Settings()
    qdrant = QdrantRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_names=settings.collection_names,
        default_dimensions=settings.embedding_dimensions,
    )
    graph_repository = build_graph_repository(settings)
    graph_service = KnowledgeGraphService(repository=graph_repository, extractor=GraphExtractor())
    embedder = build_embedder(settings)
    indexing = VectorIndexingService(
        embedder=embedder,
        repository=qdrant,
        collection_names=settings.collection_names,
        sparse_index=BM25Index(),
        chunker=SemanticChunker(max_chars=80, overlap_chars=0),
        graph_service=graph_service,
    )

    # Syllabus (structure extracted deterministically).
    normalized = NormalizedDocument(
        document_id="doc-syllabus",
        filename="dbms-syllabus.pdf",
        page_count=1,
        extraction_method="pymupdf",
        pages=[PageContent(page_number=1, text=TextCleaner().clean(SYLLABUS_TEXT))],
    )
    extracted = DeterministicMetadataExtractor().extract(normalized, document_type="syllabus")
    syllabus = StructuredDocument(
        document_id="doc-syllabus",
        filename="dbms-syllabus.pdf",
        page_count=1,
        extraction_method="pymupdf",
        pages=[PageContent(page_number=1, text=TextCleaner().clean(SYLLABUS_TEXT))],
        metadata=extracted.metadata,
        structure=extracted.structure,
    )
    asyncio.run(indexing.index(syllabus))

    # Three library books with different coverage.
    for index, book in enumerate(BOOKS, start=1):
        asyncio.run(indexing.index(make_library_book(f"doc-book-{index}", book["title"], book["chapters"])))

    recommender = RecommendationService(
        repository=qdrant,
        embedder=embedder,
        graph_service=graph_service,
        llm=build_llm(settings),
    )

    print("=" * 72)
    print("Phase 9 - Syllabus-aware library book recommendation")
    print("=" * 72)
    for query in QUERIES:
        result = asyncio.run(recommender.recommend(query))
        print(f"\nquery: {query}")
        if result.no_syllabus_evidence:
            print("  no syllabus evidence -> no recommendations")
            continue
        print(f"  subject: {result.subject}")
        print(f"  required topics: {', '.join(result.required_topics)}")
        for position, coverage in enumerate(result.recommendations, start=1):
            print(f"  {position}. {coverage.book_title} - {coverage.score:.0%}")
            print(f"     matched: {', '.join(coverage.matched_topics) or '-'}")
            print(f"     missing: {', '.join(coverage.missing_topics) or '-'}")
        print(f"  explanation: {result.explanation}")


if __name__ == "__main__":
    main()