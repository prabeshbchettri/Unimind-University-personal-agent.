"""Retrieval plans and query profiles (Phase 8).

The adaptive router turns a user query into a structured :class:`RetrievalPlan`
describing *which* retrieval strategy to use and *how* (filters, graph
parameters, reranking). Everything is explainable: every plan carries the
reason it was chosen.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Supported graph query intents (mapped 1:1 to KnowledgeGraphService methods).
GRAPH_SUBJECTS_IN_SEMESTER = "subjects_in_semester"
GRAPH_TOPICS_OF_SUBJECT = "topics_of_subject"
GRAPH_SUBTOPICS_OF_TOPIC = "subtopics_of_topic"
GRAPH_QUESTIONS_ABOUT_TOPIC = "questions_about_topic"
GRAPH_SUBJECTS_CONTAINING_TOPIC = "subjects_containing_topic"

GRAPH_INTENTS: tuple[str, ...] = (
    GRAPH_SUBJECTS_IN_SEMESTER,
    GRAPH_TOPICS_OF_SUBJECT,
    GRAPH_SUBTOPICS_OF_TOPIC,
    GRAPH_QUESTIONS_ABOUT_TOPIC,
    GRAPH_SUBJECTS_CONTAINING_TOPIC,
)


class RetrievalStrategy(str, Enum):
    """The retrieval strategies the router can select."""

    NORMAL = "NORMAL"
    HYBRID = "HYBRID"
    GRAPH = "GRAPH"
    WEB = "WEB"


class QueryProfile(BaseModel):
    """The analyzer's opinion about a query (strategy is None when unknown)."""

    strategy: RetrievalStrategy | None = None
    confidence: float = 0.0
    reason: str = ""
    filters: dict[str, str] = Field(default_factory=dict)
    graph_parameters: dict[str, Any] = Field(default_factory=dict)
    web_parameters: dict[str, Any] = Field(default_factory=dict)


class RetrievalPlan(BaseModel):
    """Structured routing decision for one query.

    Only relevant fields are populated; ``top_k`` and ``reranking_required``
    are always present so downstream consumers have a complete plan.
    """

    strategy: RetrievalStrategy
    query: str
    reason: str = ""
    filters: dict[str, str] = Field(default_factory=dict)
    top_k: int | None = None
    graph_parameters: dict[str, Any] = Field(default_factory=dict)
    web_parameters: dict[str, Any] = Field(default_factory=dict)
    reranking_required: bool = False
    fallback: bool = False
