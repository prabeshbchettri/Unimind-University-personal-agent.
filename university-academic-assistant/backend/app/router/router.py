"""Adaptive router (Phase 8 + Phase 11).

Combines the query analyzers with the retrieval strategies
(NORMAL / HYBRID / GRAPH / WEB) and produces a structured, explainable
:class:`RetrievalPlan` for every query. The router:

1. asks the deterministic rule-based analyzer first;
2. consults the LLM analyzer only when the rules are inconclusive;
3. falls back to a safe strategy (hybrid) when classification is uncertain;
4. executes WEB retrieval (Phase 11) with graceful degradation: a failing or
   timing-out web search falls back to internal retrieval and never fails
   the answer.

It implements the same ``retrieve(query, top_k)`` interface as the retrievers,
so ``ChatService`` is unchanged: every chat query is routed automatically.
"""

from __future__ import annotations

import logging

from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import SearchResult
from app.retrieval.normal_retriever import NormalRetriever
from app.retrieval.web_retriever import WebRetrievalError, WebRetriever
from app.router.analyzers import LLMQueryAnalyzer, QueryAnalyzer, RuleBasedQueryAnalyzer
from app.router.plan import RetrievalPlan, RetrievalStrategy
from app.services.base import Service

logger = logging.getLogger("app.router.router")


class AdaptiveRouter(Service):
    """Selects and executes the retrieval strategy per query."""

    def __init__(
        self,
        normal_retriever: NormalRetriever,
        hybrid_retriever: HybridRetriever,
        graph_retriever: GraphRetriever | None = None,
        analyzer: QueryAnalyzer | None = None,
        llm_analyzer: LLMQueryAnalyzer | None = None,
        top_k: int = 5,
        fixed_strategy: str | None = None,
        fallback_strategy: str = "hybrid",
        classifier_mode: str = "rules",
        web_retriever: WebRetriever | None = None,
    ) -> None:
        super().__init__()
        self.normal_retriever = normal_retriever
        self.hybrid_retriever = hybrid_retriever
        self.graph_retriever = graph_retriever
        self.analyzer = analyzer or RuleBasedQueryAnalyzer()
        self.llm_analyzer = llm_analyzer
        self.top_k = top_k
        self.fixed_strategy = fixed_strategy
        self.fallback_strategy = fallback_strategy
        self.classifier_mode = classifier_mode
        self.web_retriever = web_retriever
        self.last_plan: RetrievalPlan | None = None

        self._retrievers: dict[RetrievalStrategy, object] = {
            RetrievalStrategy.NORMAL: normal_retriever,
            RetrievalStrategy.HYBRID: hybrid_retriever,
        }
        if graph_retriever is not None:
            self._retrievers[RetrievalStrategy.GRAPH] = graph_retriever
        if web_retriever is not None:
            self._retrievers[RetrievalStrategy.WEB] = web_retriever

    async def health(self) -> dict:
        return {
            "name": self.name,
            "status": "ok",
            "classifier": self.classifier_mode,
            "fixed_strategy": self.fixed_strategy,
            "strategies": [strategy.value for strategy in self._retrievers],
            "fallback_strategy": self.fallback_strategy,
            "web_search": type(self.web_retriever.provider).__name__ if self.web_retriever else None,
        }

    def analyze(self, query: str) -> RetrievalPlan:
        """Produce the structured retrieval plan for ``query`` (no retrieval)."""
        if self.fixed_strategy is not None:
            return RetrievalPlan(
                strategy=RetrievalStrategy(self.fixed_strategy.upper()),
                query=query,
                reason=f"Fixed retrieval strategy configured ({self.fixed_strategy}).",
                top_k=self.top_k,
            )

        profile = self.analyzer.analyze(query)
        reason = profile.reason
        fallback = False

        if profile.strategy is None:
            if self.llm_analyzer and self.classifier_mode in {"auto", "llm"}:
                profile = self.llm_analyzer.analyze(query)
                if profile.strategy is not None:
                    reason = f"Rules inconclusive; {reason}"
            if profile.strategy is None:
                fallback = True
                return RetrievalPlan(
                    strategy=RetrievalStrategy(self.fallback_strategy.upper()),
                    query=query,
                    reason="Could not classify the query; falling back to "
                    f"{self.fallback_strategy.upper()} retrieval.",
                    top_k=self.top_k,
                    fallback=True,
                )
        elif self.classifier_mode == "llm" and self.llm_analyzer and profile.confidence < 0.8:
            # Low-confidence rule verdicts can be double-checked by the LLM.
            llm_profile = self.llm_analyzer.analyze(query)
            if llm_profile.strategy is not None:
                profile = llm_profile
                reason = f"Rules uncertain; {reason}"

        return RetrievalPlan(
            strategy=profile.strategy,
            query=query,
            reason=reason,
            filters=profile.filters,
            top_k=self.top_k,
            graph_parameters=profile.graph_parameters,
            web_parameters=profile.web_parameters,
            reranking_required=profile.strategy == RetrievalStrategy.HYBRID or bool(profile.filters),
            fallback=fallback,
        )

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Route ``query`` and return the chosen strategy's top results."""
        plan = self.analyze(query)
        self.last_plan = plan
        limit = top_k or plan.top_k or self.top_k

        if plan.strategy == RetrievalStrategy.WEB:
            return await self._retrieve_web(plan, limit)

        retriever = self._retrievers[plan.strategy]
        if plan.strategy == RetrievalStrategy.GRAPH and isinstance(retriever, GraphRetriever):
            results = await retriever.retrieve_for_plan(plan, limit)
        else:
            results = await retriever.retrieve(query, top_k=limit)

        return self._apply_filters(plan, results)

    async def _retrieve_web(self, plan: RetrievalPlan, limit: int) -> list[SearchResult]:
        """Execute the WEB strategy with graceful degradation.

        Mixed questions (university context + web signal) also retrieve the
        university documents, which stay the authority for university facts.
        A failing or timing-out web search falls back to internal retrieval
        (``plan.fallback = True``) instead of failing the answer.
        """
        internal: list[SearchResult] = []
        if plan.web_parameters.get("mixed"):
            internal = await self.hybrid_retriever.retrieve(plan.query, top_k=limit)

        try:
            web = await self._safe_web_retrieve(plan.query, limit)
        except WebRetrievalError as exc:
            logger.warning("Web search failed; falling back to internal retrieval: %s", exc)
            plan.fallback = True
            if not internal:
                internal = await self.hybrid_retriever.retrieve(plan.query, top_k=limit)
            return internal

        return [*internal, *web]

    async def _safe_web_retrieve(self, query: str, limit: int) -> list[SearchResult]:
        """Web retrieval; raises :class:`WebRetrievalError` on failure/timeout."""
        if self.web_retriever is None:
            raise WebRetrievalError("Web search is not configured.")
        return await self.web_retriever.retrieve(query, top_k=limit)

    @staticmethod
    def _apply_filters(plan: RetrievalPlan, results: list[SearchResult]) -> list[SearchResult]:
        """Apply the plan's filters when they are satisfied by any result."""
        document_type = plan.filters.get("document_type")
        if not document_type:
            return results
        filtered = [result for result in results if result.metadata.get("document_type") == document_type]
        # Over-filtering is worse than no filtering: keep unfiltered results
        # when the filter would empty the list.
        return filtered if filtered else results
