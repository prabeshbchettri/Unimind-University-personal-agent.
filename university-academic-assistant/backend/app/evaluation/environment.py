"""Deterministic offline evaluation environment (Phase 13).

Builds the full pipeline with hermetic backends — in-memory Qdrant, the
deterministic embedder, the stub LLM, the in-memory knowledge graph and the
BM25 sparse index — indexes :mod:`app.evaluation.corpus` and exposes every
component an evaluation needs (retrievers, adaptive router, chat service,
recommendation service, ingestion service).
"""

from __future__ import annotations

from dataclasses import dataclass

import asyncio

from app.caching import RetrievalCache
from app.chunking import SemanticChunker
from app.embedding import DeterministicEmbedder
from app.graph.extractor import GraphExtractor
from app.llm.stub import StubLLMClient
from app.rag.context import ContextBuilder
from app.repositories.graph import InMemoryGraphRepository
from app.repositories.qdrant import QdrantRepository
from app.retrieval.cached import CachedRetriever
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.normal_retriever import NormalRetriever
from app.retrieval.sparse import BM25Index
from app.router.analyzers import RuleBasedQueryAnalyzer
from app.router.router import AdaptiveRouter
from app.services.chat import ChatService
from app.services.graph import KnowledgeGraphService
from app.services.ingestion import DocumentIngestionService
from app.services.recommendation import RecommendationService
from app.services.vector_indexing import VectorIndexingService
from app.structure.models import StructuredDocument
from app.web import StubWebSearchProvider

COLLECTIONS = ["university_docs", "past_questions", "library_books"]

DEFAULT_MIN_SCORE = 0.01


@dataclass
class EvaluationEnvironment:
    """Everything an evaluation runner needs, fully wired."""

    repository: QdrantRepository
    embedder: DeterministicEmbedder
    sparse_index: BM25Index
    graph_repository: InMemoryGraphRepository
    graph_service: KnowledgeGraphService
    indexing_service: VectorIndexingService
    normal_retriever: NormalRetriever
    hybrid_retriever: HybridRetriever
    graph_retriever: GraphRetriever
    router: AdaptiveRouter
    chat_service: ChatService
    recommendation_service: RecommendationService
    ingestion_service: DocumentIngestionService
    llm: StubLLMClient
    context_builder: ContextBuilder


def _run(coro):
    return asyncio.run(coro)


def build_environment(rounder_chunker: bool = False) -> EvaluationEnvironment:
    """Index the evaluation corpus and return the wired environment."""
    embedder = DeterministicEmbedder(dimensions=384)
    repository = QdrantRepository(url=":memory:", collection_names=COLLECTIONS, default_dimensions=384)
    sparse_index = BM25Index()
    graph_repository = InMemoryGraphRepository()
    graph_service = KnowledgeGraphService(repository=graph_repository, extractor=GraphExtractor())
    retrieval_cache = RetrievalCache()

    indexing_service = VectorIndexingService(
        embedder=embedder,
        repository=repository,
        collection_names=COLLECTIONS,
        chunker=SemanticChunker(max_chars=1200, overlap_chars=150),
        sparse_index=sparse_index,
        graph_service=graph_service,
        retrieval_cache=retrieval_cache,
    )

    from app.evaluation.corpus import DOCUMENTS

    for document in DOCUMENTS:
        _run(indexing_service.index(document))

    normal = NormalRetriever(
        embedder=embedder,
        repository=repository,
        collection_names=COLLECTIONS,
        top_k=5,
        min_score=DEFAULT_MIN_SCORE,
    )
    hybrid = HybridRetriever(dense_retriever=normal, sparse_index=sparse_index, top_k=5)
    graph_retriever = GraphRetriever(graph_service=graph_service, analyzer=RuleBasedQueryAnalyzer(), top_k=5, dense_retriever=normal)

    router = AdaptiveRouter(
        normal_retriever=normal,
        hybrid_retriever=hybrid,
        graph_retriever=graph_retriever,
        analyzer=RuleBasedQueryAnalyzer(),
        top_k=5,
        classifier_mode="rules",
    )

    llm = StubLLMClient(default_answer="I could not find sufficient information in the available sources.")
    context_builder = ContextBuilder(max_chars=4000, max_sources=8)
    chat_service = ChatService(
        retriever=CachedRetriever(router, retrieval_cache),
        context_builder=context_builder,
        llm=llm,
        top_k=5,
    )
    recommendation_service = RecommendationService(
        repository=repository,
        embedder=embedder,
        graph_service=graph_service,
        llm=llm,
    )

    return EvaluationEnvironment(
        repository=repository,
        embedder=embedder,
        sparse_index=sparse_index,
        graph_repository=graph_repository,
        graph_service=graph_service,
        indexing_service=indexing_service,
        normal_retriever=normal,
        hybrid_retriever=hybrid,
        graph_retriever=graph_retriever,
        router=router,
        chat_service=chat_service,
        recommendation_service=recommendation_service,
        ingestion_service=DocumentIngestionService(),
        llm=llm,
        context_builder=context_builder,
    )


def document_id_of(result) -> str:
    """The document id of a retrieved result (dense or graph evidence)."""
    metadata = result.metadata or {}
    doc_id = metadata.get("document_id")
    if doc_id:
        return doc_id
    title = metadata.get("title") or ""
    from app.evaluation.corpus import DOCUMENTS

    for document in DOCUMENTS:
        if document.metadata.title == title:
            return document.document_id
    return title or result.collection