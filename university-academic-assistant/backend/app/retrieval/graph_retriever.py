"""Graph retrieval (Phase 8).

The first Graph RAG flow:

    Query -> Graph retrieval (Neo4j/in-memory) -> entities/relationships
          -> optional vector retrieval when the graph has nothing
          -> SearchResults -> ContextBuilder -> Llama

Graph queries do not always need vector retrieval: when the graph answers
directly (e.g. "Which subjects are in semester 5?"), the entities are returned
as evidence without consulting Qdrant. Only when the graph has no answer for
the intent is the dense retriever consulted as a fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.retrieval.models import SearchResult
from app.router.analyzers import QueryAnalyzer, RuleBasedQueryAnalyzer
from app.router.plan import (
    GRAPH_QUESTIONS_ABOUT_TOPIC,
    GRAPH_SUBJECTS_CONTAINING_TOPIC,
    GRAPH_SUBJECTS_IN_SEMESTER,
    GRAPH_SUBTOPICS_OF_TOPIC,
    GRAPH_TOPICS_OF_SUBJECT,
    RetrievalPlan,
    RetrievalStrategy,
)

if TYPE_CHECKING:
    from app.services.graph import KnowledgeGraphService

GRAPH_COLLECTION = "knowledge_graph"


def _describe(intent: str, row: dict) -> str:
    """Human-readable evidence text for one graph entity."""
    title = row.get("title")
    page = row.get("page")
    source = f" [source: {title}, page {page}]" if title else ""
    if intent == GRAPH_SUBJECTS_IN_SEMESTER:
        code = f" ({row.get('code')})" if row.get("code") else ""
        semester = f", semester {row.get('semester')}" if row.get("semester") else ""
        return f"Subject: {row.get('name')}{code}{semester}{source}"
    if intent == GRAPH_SUBJECTS_CONTAINING_TOPIC:
        return f"Subject: {row.get('name')}{source}"
    if intent == GRAPH_TOPICS_OF_SUBJECT:
        return f"Topic: {row.get('name')}{source}"
    if intent == GRAPH_SUBTOPICS_OF_TOPIC:
        return f"Subtopic: {row.get('name')}{source}"
    if intent == GRAPH_QUESTIONS_ABOUT_TOPIC:
        number = row.get("question_number")
        marks = f", {row.get('marks')} marks" if row.get("marks") else ""
        year = f" ({row.get('year')})" if row.get("year") else ""
        text = row.get("text") or "past question"
        return f"Q{number}{year} ({marks.lstrip(', ')}): {text}{source}".replace(" (): ", ": ")
    return f"{row.get('name', '')}{source}"


class GraphRetriever:
    """Retrieves evidence from the academic knowledge graph."""

    name = "GraphRetriever"

    def __init__(
        self,
        graph_service: "KnowledgeGraphService",
        analyzer: QueryAnalyzer | None = None,
        top_k: int = 5,
        dense_retriever=None,
    ) -> None:
        self.graph_service = graph_service
        self.analyzer = analyzer or RuleBasedQueryAnalyzer()
        self.top_k = top_k
        self.dense_retriever = dense_retriever

    def _graph_method(self, intent: str) -> str:
        return intent  # intents map 1:1 to KnowledgeGraphService methods

    async def _query_graph(self, parameters: dict) -> list[dict]:
        intent = parameters.get("intent")
        method = getattr(self.graph_service, self._graph_method(intent), None)
        if method is None:
            return []
        args = {
            GRAPH_SUBJECTS_IN_SEMESTER: {"semester": parameters.get("semester")},
            GRAPH_TOPICS_OF_SUBJECT: {"subject": parameters.get("subject")},
            GRAPH_SUBTOPICS_OF_TOPIC: {"topic": parameters.get("topic")},
            GRAPH_QUESTIONS_ABOUT_TOPIC: {"topic": parameters.get("topic")},
            GRAPH_SUBJECTS_CONTAINING_TOPIC: {"topic": parameters.get("topic")},
        }.get(intent, {})
        if not args or any(value is None for value in args.values()):
            return []
        return await method(**args)

    async def retrieve_for_plan(self, plan: RetrievalPlan, top_k: int | None = None) -> list[SearchResult]:
        """Graph retrieval for a router plan (parameters come from the plan)."""
        limit = top_k or plan.top_k or self.top_k
        rows = await self._query_graph(plan.graph_parameters)
        if rows:
            return self._to_results(rows, plan.graph_parameters.get("intent", ""), limit)
        return await self._dense_fallback(plan.query, limit)

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Standalone graph retrieval (parses the query itself)."""
        limit = top_k or self.top_k
        profile = self.analyzer.analyze(query)
        if profile.strategy != RetrievalStrategy.GRAPH:
            return []
        plan = RetrievalPlan(
            strategy=RetrievalStrategy.GRAPH,
            query=query,
            reason=profile.reason,
            graph_parameters=profile.graph_parameters,
            top_k=limit,
        )
        return await self.retrieve_for_plan(plan, limit)

    async def _dense_fallback(self, query: str, top_k: int) -> list[SearchResult]:
        """Optional vector retrieval when the graph has no answer."""
        if self.dense_retriever is None:
            return []
        return await self.dense_retriever.retrieve(query, top_k=top_k)

    @staticmethod
    def _to_results(rows: list[dict], intent: str, top_k: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for index, row in enumerate(rows[:top_k]):
            metadata = dict(row)
            metadata["intent"] = intent
            results.append(
                SearchResult(
                    text=_describe(intent, row),
                    score=round(max(1.0 - index * 0.01, 0.9), 4),
                    metadata=metadata,
                    collection=GRAPH_COLLECTION,
                    method="graph",
                )
            )
        return results
