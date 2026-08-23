"""Retrieval package.

Phase 4: semantic (vector) retrieval over Qdrant with collection routing.
Phase 5: NormalRetriever (dense). Phase 6: BM25 sparse index + HybridRetriever
(dense + sparse fused with RRF). Phase 8: GraphRetriever (Neo4j/in-memory
knowledge graph), consumed by the adaptive router. Phase 11: WebRetriever
(external/current information behind the same ``retrieve`` interface).
"""

from .collections import DEFAULT_COLLECTIONS, all_collections, collection_for
from .graph_retriever import GRAPH_COLLECTION, GraphRetriever
from .hybrid_retriever import FusedResult, HybridRetriever
from .models import SearchHit, SearchResult
from .normal_retriever import NormalRetriever
from .sparse import BM25Index, SparseHit, tokenize
from .web_retriever import WebRetrievalError, WebRetriever

__all__ = [
    "DEFAULT_COLLECTIONS",
    "all_collections",
    "collection_for",
    "SearchHit",
    "SearchResult",
    "NormalRetriever",
    "BM25Index",
    "SparseHit",
    "tokenize",
    "FusedResult",
    "HybridRetriever",
    "GRAPH_COLLECTION",
    "GraphRetriever",
    "WebRetriever",
    "WebRetrievalError",
]
