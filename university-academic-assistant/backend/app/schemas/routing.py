"""Adaptive router endpoint schemas (Phase 8)."""

from typing import Any

from pydantic import BaseModel, Field


class QueryPlanRequest(BaseModel):
    """Request body for ``POST /router/plan``."""

    query: str = Field(..., min_length=1, description="User query to classify.")


class RetrievalPlanSchema(BaseModel):
    """The structured routing decision for a query."""

    strategy: str = Field(..., description="NORMAL | HYBRID | GRAPH | WEB")
    query: str = Field(..., description="The original query.")
    reason: str = Field("", description="Explainable reason for the decision.")
    filters: dict[str, str] = Field(default_factory=dict, description="Metadata filters to apply.")
    top_k: int | None = Field(None, description="Number of results to retrieve.")
    graph_parameters: dict[str, Any] = Field(default_factory=dict, description="Graph query parameters (GRAPH strategy).")
    web_parameters: dict[str, Any] = Field(default_factory=dict, description="Web parameters (WEB strategy; mixed=True means web supplements internal retrieval).")
    reranking_required: bool = Field(False, description="Whether reranking is required.")
    fallback: bool = Field(False, description="Whether this is a fallback decision.")