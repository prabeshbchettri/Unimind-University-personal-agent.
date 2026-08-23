"""Phase 8 + Phase 11 demo: query analyzer + adaptive router evaluation.

Routes the 36-query eval set (NORMAL / HYBRID / GRAPH / WEB) through the
adaptive router, prints the routing metrics (accuracy / false routing /
fallback) and shows the structured plan for one query of each strategy.

Run:  python scripts/evaluate_router.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import Settings  # noqa: E402
from app.embedding import build_embedder  # noqa: E402
from app.graph.extractor import GraphExtractor  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.retrieval.graph_retriever import GraphRetriever  # noqa: E402
from app.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from app.retrieval.normal_retriever import NormalRetriever  # noqa: E402
from app.retrieval.sparse import BM25Index  # noqa: E402
from app.repositories.graph import build_graph_repository  # noqa: E402
from app.repositories.qdrant import QdrantRepository  # noqa: E402
from app.router.analyzers import LLMQueryAnalyzer, RuleBasedQueryAnalyzer  # noqa: E402
from app.router.eval_set import ROUTING_EVAL_SET, evaluate_router  # noqa: E402
from app.router.router import AdaptiveRouter  # noqa: E402
from app.services.graph import KnowledgeGraphService  # noqa: E402


def build_router() -> AdaptiveRouter:
    settings = Settings()
    qdrant = QdrantRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_names=settings.collection_names,
        default_dimensions=settings.embedding_dimensions,
    )
    graph_repository = build_graph_repository(settings)
    graph_service = KnowledgeGraphService(repository=graph_repository, extractor=GraphExtractor())
    return AdaptiveRouter(
        normal_retriever=NormalRetriever(
            embedder=build_embedder(settings),
            repository=qdrant,
            collection_names=settings.collection_names,
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
        ),
        hybrid_retriever=HybridRetriever(
            dense_retriever=NormalRetriever(
                embedder=build_embedder(settings),
                repository=qdrant,
                collection_names=settings.collection_names,
                top_k=settings.rag_top_k,
                min_score=settings.rag_min_score,
            ),
            sparse_index=BM25Index(),
            top_k=settings.rag_top_k,
            rrf_k=settings.hybrid_rrf_k,
            rrf_weight=settings.hybrid_rerank_rrf_weight,
            dense_weight=settings.hybrid_rerank_dense_weight,
            pool_factor=settings.hybrid_pool_factor,
        ),
        graph_retriever=GraphRetriever(
            graph_service=graph_service,
            analyzer=RuleBasedQueryAnalyzer(),
            top_k=settings.rag_top_k,
        ),
        analyzer=RuleBasedQueryAnalyzer(),
        llm_analyzer=LLMQueryAnalyzer(build_llm(settings)) if settings.router_classifier in {"auto", "llm"} else None,
        top_k=settings.rag_top_k,
        fixed_strategy=None if settings.rag_retrieval_strategy.lower() == "auto" else settings.rag_retrieval_strategy,
        fallback_strategy=settings.router_fallback_strategy,
        classifier_mode=settings.router_classifier,
    )


def main() -> None:
    router = build_router()
    metrics = evaluate_router(router)

    print("=" * 72)
    print("Phase 8 + Phase 11 - Query Analyzer and Adaptive Router evaluation")
    print(f"eval set: {metrics['total']} queries "
          f"({sum(1 for i in ROUTING_EVAL_SET if i['expected'] == 'NORMAL')} NORMAL, "
          f"{sum(1 for i in ROUTING_EVAL_SET if i['expected'] == 'HYBRID')} HYBRID, "
          f"{sum(1 for i in ROUTING_EVAL_SET if i['expected'] == 'GRAPH')} GRAPH, "
          f"{sum(1 for i in ROUTING_EVAL_SET if i['expected'] == 'WEB')} WEB)")
    print("-" * 72)
    print(f"routing accuracy  : {metrics['routing_accuracy']:.2%}")
    print(f"false routing rate: {metrics['false_routing_rate']:.2%}")
    print(f"fallback rate     : {metrics['fallback_rate']:.2%}")
    print("=" * 72)

    for query in [
        "Explain normalization.",
        "What does regulation 12 say about attendance?",
        "Which subjects are in semester 5?",
        "What is the latest Python version?",
    ]:
        plan = router.analyze(query)
        print(f"\nquery : {query}")
        print(f"plan  : strategy={plan.strategy.value}")
        print(f"        reason  = {plan.reason}")
        if plan.filters:
            print(f"        filters = {plan.filters}")
        if plan.graph_parameters:
            print(f"        graph   = {plan.graph_parameters}")
        if plan.web_parameters:
            print(f"        web     = {plan.web_parameters}")


if __name__ == "__main__":
    main()