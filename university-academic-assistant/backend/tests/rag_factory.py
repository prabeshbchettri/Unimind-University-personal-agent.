"""Shared factory helpers for Phase 5 (RAG) tests.

Builds a fully deterministic, offline pipeline: in-memory Qdrant, the
deterministic embedder and the stub LLM client.
"""

from __future__ import annotations

from tests.structure_factory import make_structured

from app.chunking import SemanticChunker
from app.chunking.models import DocumentChunk
from app.embedding import DeterministicEmbedder
from app.graph.extractor import GraphExtractor
from app.llm import StubLLMClient
from app.rag.context import ContextBuilder
from app.repositories.chat_history import ChatHistoryRepository
from app.repositories.graph import InMemoryGraphRepository
from app.repositories.qdrant import QdrantRepository
from app.retrieval import SearchResult
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.normal_retriever import NormalRetriever
from app.retrieval.sparse import BM25Index
from app.router.analyzers import RuleBasedQueryAnalyzer
from app.router.router import AdaptiveRouter
from app.services.graph import KnowledgeGraphService
from app.services.vector_indexing import VectorIndexingService
from app.structure.models import DocumentMetadata, DocumentStructure, Chapter, Question
from app.web import StubWebSearchProvider

COLLECTIONS = ["university_docs", "past_questions", "library_books"]

DEFAULT_ANSWER = "I could not find sufficient information in the available sources."
# Default relevance floor for the deterministic test pipelines: in-database
# queries score well above this; zero-overlap queries score exactly 0 and are
# treated as "no evidence" (exercising the insufficiency path).
DEFAULT_MIN_SCORE = 0.01


def make_embedder(dimensions: int = 384) -> DeterministicEmbedder:
    return DeterministicEmbedder(dimensions)


def make_repository(dimensions: int = 384) -> QdrantRepository:
    return QdrantRepository(url=":memory:", collection_names=COLLECTIONS, default_dimensions=dimensions)


def make_indexing_service(
    dimensions: int = 384,
    sparse_index: BM25Index | None = None,
    graph_service: KnowledgeGraphService | None = None,
    retrieval_cache=None,
) -> VectorIndexingService:
    return VectorIndexingService(
        embedder=make_embedder(dimensions),
        repository=make_repository(dimensions),
        collection_names=COLLECTIONS,
        chunker=SemanticChunker(max_chars=80, overlap_chars=0),
        sparse_index=sparse_index,
        graph_service=graph_service,
        retrieval_cache=retrieval_cache,
    )


def make_sparse_index() -> BM25Index:
    return BM25Index()


def make_retriever(embedder, repository, top_k: int = 5, min_score: float = DEFAULT_MIN_SCORE) -> NormalRetriever:
    return NormalRetriever(
        embedder=embedder,
        repository=repository,
        collection_names=COLLECTIONS,
        top_k=top_k,
        min_score=min_score,
    )


def make_hybrid_retriever(
    dense_retriever: NormalRetriever,
    sparse_index: BM25Index,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
) -> HybridRetriever:
    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_index=sparse_index,
        top_k=top_k,
    )


def make_context_builder(max_chars: int = 2000, max_sources: int = 8) -> ContextBuilder:
    return ContextBuilder(max_chars=max_chars, max_sources=max_sources)


def make_llm(responder=None) -> StubLLMClient:
    return StubLLMClient(default_answer=DEFAULT_ANSWER, responder=responder)


def make_chat_history(backend: str = "memory", **kwargs) -> ChatHistoryRepository:
    return ChatHistoryRepository(backend=backend, **kwargs)


def index(service: VectorIndexingService, structured) -> None:
    """Index a structured document (blocking wrapper for tests)."""
    import asyncio

    asyncio.run(service.index(structured))


def sample_structured(
    text: str,
    document_type: str = "syllabus",
    title: str = "DBMS Syllabus",
    subject: str = "Database Management System",
    semester: int = 5,
    topics: list[str] | None = None,
    document_id: str = "doc-1",
):
    return make_structured(
        [text],
        document_id=document_id,
        document_type=document_type,
        metadata=DocumentMetadata(
            document_type=document_type,
            title=title,
            semester=semester,
            subject=subject,
        ),
        structure=DocumentStructure(topics=topics or ["Introduction", "Normalization", "Transactions"]),
    )


def normalization_chunk() -> SearchResult:
    """A representative retrieved chunk for 'Explain normalization'."""
    return SearchResult(
        text="Normalization removes redundancy in tables.",
        score=0.9,
        metadata={
            "document_id": "doc-1",
            "document_type": "syllabus",
            "title": "DBMS Syllabus",
            "page": 3,
            "subject": "Database Management System",
            "topic": "Normalization",
        },
        collection="university_docs",
    )


def make_chunk(
    text: str,
    document_id: str = "doc-1",
    document_type: str = "syllabus",
    title: str = "DBMS Syllabus",
    subject: str = "Database Management System",
    page: int = 1,
    topic: str | None = None,
    chunk_id: str | None = None,
) -> DocumentChunk:
    """A minimal chunk for direct BM25/sparse unit tests (no Qdrant needed)."""
    return DocumentChunk(
        chunk_id=chunk_id or f"{document_id}-{topic or 'chunk'}",
        document_id=document_id,
        document_type=document_type,
        title=title,
        page=page,
        topic=topic,
        subject=subject,
        semester=5,
        text=text,
    )


# ---------------------------------------------------------------------------
# Phase 7 knowledge-graph helpers
# ---------------------------------------------------------------------------

def make_graph_repository() -> InMemoryGraphRepository:
    return InMemoryGraphRepository()


def make_graph_service(repository: InMemoryGraphRepository | None = None) -> KnowledgeGraphService:
    return KnowledgeGraphService(repository=repository or make_graph_repository(), extractor=GraphExtractor())


def make_graph_syllabus(
    document_id: str = "doc-syllabus",
    title: str = "DBMS Syllabus",
    university: str = "Tribhuvan University",
    program: str = "BSc CSIT",
    semester: int = 5,
    subject: str = "Database Management System",
    subject_code: str = "CSIT 325",
    topics: list[str] | None = None,
    subtopics: list[str] | None = None,
) -> "StructuredDocument":
    """A syllabus structured document with full academic metadata."""
    return make_structured(
        ["Database Management System Syllabus\nUnit 1: Introduction\nUnit 2: Normalization\n"],
        document_id=document_id,
        document_type="syllabus",
        metadata=DocumentMetadata(
            document_type="syllabus",
            title=title,
            university=university,
            program=program,
            semester=semester,
            subject=subject,
            subject_code=subject_code,
            topic=topics[0] if topics else None,
            subtopic=subtopics[0] if subtopics else None,
        ),
        structure=DocumentStructure(
            topics=topics or ["Introduction", "Normalization", "Transactions"],
            subtopics=subtopics or [],
        ),
    )


def make_graph_past_paper(
    document_id: str = "doc-past",
    title: str = "DBMS 2080",
    university: str = "Tribhuvan University",
    program: str = "BSc CSIT",
    semester: int = 5,
    subject: str = "Database Management System",
    academic_year: str = "2080",
    year: int = 2080,
    topic: str = "Normalization",
    questions: list[tuple[int, str, int]] | None = None,
) -> "StructuredDocument":
    """A past question paper structured document with extracted questions."""
    questions = questions or [(5, "Explain normalization with an example.", 10)]
    return make_structured(
        ["DBMS 2080\nQ5. Explain normalization with an example. 10 Marks\n"],
        document_id=document_id,
        document_type="past_question",
        metadata=DocumentMetadata(
            document_type="past_question",
            title=title,
            university=university,
            program=program,
            semester=semester,
            subject=subject,
            academic_year=academic_year,
            year=year,
            topic=topic,
            question_number=questions[0][0],
            marks=questions[0][2],
        ),
        structure=DocumentStructure(
            topics=["Normalization", "SQL"],
            questions=[Question(question_number=q[0], text=q[1], marks=q[2]) for q in questions],
        ),
    )


# ---------------------------------------------------------------------------
# Phase 8 adaptive router helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 9 recommendation helpers
# ---------------------------------------------------------------------------

def make_library_book(
    document_id: str = "doc-book-1",
    title: str = "Database System Concepts",
    author: str | None = None,
    chapters: list[tuple[str, list[str]]] | None = None,
) -> "StructuredDocument":
    """A library book structured document with chapter/subtopic content.

    ``chapters`` is a list of ``(chapter_title, [subtopics])``. The text uses
    ``Chapter N:`` headings and numbered subsections (``N.M``) so the chunker
    produces chunks carrying topic (chapter) and subtopic (subsection) payloads.
    """
    chapters = chapters or [
        ("Introduction", ["Data Models"]),
        ("Normalization", ["Normal Forms", "First Normal Form"]),
    ]
    lines: list[str] = []
    for index, (chapter_title, subtopics) in enumerate(chapters, start=1):
        lines.append(f"Chapter {index}: {chapter_title}")
        for sub_index, subtopic in enumerate(subtopics, start=1):
            lines.append(f"{index}.{sub_index} {subtopic}")
    structure = DocumentStructure(
        topics=[chapter_title for chapter_title, _ in chapters],
        chapters=[
            Chapter(
                chapter_number=index,
                title=chapter_title,
                topics=[chapter_title],
                subtopics=subtopics,
            )
            for index, (chapter_title, subtopics) in enumerate(chapters, start=1)
        ],
    )
    return make_structured(
        ["\n".join(lines)],
        document_id=document_id,
        document_type="library_book",
        metadata=DocumentMetadata(document_type="library_book", title=title, author=author),
        structure=structure,
    )


def make_recommendation_service(
    repository,
    embedder,
    graph_service=None,
    llm=None,
    **kwargs,
):
    from app.services.recommendation import RecommendationService

    return RecommendationService(
        repository=repository,
        embedder=embedder,
        graph_service=graph_service,
        llm=llm,
        **kwargs,
    )


def make_rule_analyzer() -> RuleBasedQueryAnalyzer:
    return RuleBasedQueryAnalyzer()


def make_web_provider() -> StubWebSearchProvider:
    return StubWebSearchProvider()


def make_web_retriever(provider=None, top_k: int = 4, timeout: float = 1.0) -> "WebRetriever":
    from app.retrieval.web_retriever import WebRetriever

    return WebRetriever(provider=provider or make_web_provider(), top_k=top_k, timeout=timeout)


def make_graph_retriever(graph_service: KnowledgeGraphService, dense_retriever=None, top_k: int = 5) -> GraphRetriever:
    return GraphRetriever(
        graph_service=graph_service,
        analyzer=make_rule_analyzer(),
        top_k=top_k,
        dense_retriever=dense_retriever,
    )


def make_adaptive_router(
    normal_retriever: NormalRetriever,
    hybrid_retriever: HybridRetriever,
    graph_retriever: GraphRetriever | None = None,
    top_k: int = 5,
    **kwargs,
) -> AdaptiveRouter:
    return AdaptiveRouter(
        normal_retriever=normal_retriever,
        hybrid_retriever=hybrid_retriever,
        graph_retriever=graph_retriever,
        analyzer=make_rule_analyzer(),
        top_k=top_k,
        **kwargs,
    )