"""Adaptive router endpoints (Phase 8)."""

from fastapi import APIRouter, Request

from app.schemas.routing import QueryPlanRequest, RetrievalPlanSchema

router = APIRouter(prefix="/router", tags=["router"])


@router.post("/plan", response_model=RetrievalPlanSchema, summary="Classify a query into a retrieval plan")
async def retrieval_plan(request: Request, body: QueryPlanRequest) -> RetrievalPlanSchema:
    """Return the structured retrieval plan the adaptive router would execute.

    The plan contains the strategy (NORMAL/HYBRID/GRAPH/WEB), the explainable
    reason, optional filters, graph parameters, web parameters and reranking
    requirements — without running any retrieval.
    """
    plan = request.app.state.adaptive_router.analyze(body.query)
    return RetrievalPlanSchema(
        strategy=plan.strategy.value,
        query=plan.query,
        reason=plan.reason,
        filters=plan.filters,
        top_k=plan.top_k,
        graph_parameters=plan.graph_parameters,
        web_parameters=plan.web_parameters,
        reranking_required=plan.reranking_required,
        fallback=plan.fallback,
    )