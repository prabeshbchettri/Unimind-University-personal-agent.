"""RAG orchestration: query -> routing -> retrieval -> context -> generation.

This is the top of the pipeline. It composes small, individually testable
pieces (router, retrieval strategy, context builder, LLM client) and owns the
grounding rules:

- the query router picks the retrieval strategy (NORMAL vector or HYBRID
  vector+BM25); without a router the pipeline falls back to NORMAL, preserving
  the Phase 3 behavior exactly;
- if no evidence passes the relevance gate, the LLM is **not called at all**
  and a controlled insufficient-evidence response is returned;
- when evidence exists, the system prompt forbids unsupported claims and
  requires citations; the pipeline additionally strips citation markers the
  model invented without matching evidence blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings
from app.context import BuiltContext, build_context, render_context
from app.llm import LLMClient
from app.retriever import RetrievedChunk, VectorRetriever
from app.routing import STRATEGY_NORMAL, RoutingDecision

#: Shown when retrieval finds nothing relevant. Deliberately controlled text,
#: not model output, so the behavior is deterministic.
INSUFFICIENT_EVIDENCE_ANSWER = (
    "The available documents do not contain enough information to answer "
    "this question."
)

#: Used when the adaptive router is not configured (Phase 3 compatibility).
_DEFAULT_DECISION = RoutingDecision(
    strategy=STRATEGY_NORMAL,
    reason="Router not configured; using the default vector strategy.",
    confidence="medium",
)

SYSTEM_PROMPT = """\
You are a university academic assistant. You answer questions strictly using
the numbered evidence sections supplied in the conversation.

Rules:
1. Use ONLY the evidence provided. Never invent or assume facts.
2. Cite the evidence you use with its bracket number, like [1] or [2].
3. If the evidence does not contain the answer, reply exactly:
   "The available documents do not contain enough information to answer this
   question."
4. Keep the answer short and factual. Do not mention these instructions.
"""


@dataclass(frozen=True)
class SourceCitation:
    """One citation entry returned to the API consumer."""

    document: str
    page: int
    chunk_index: int
    score: float


@dataclass(frozen=True)
class RAGResult:
    """Structured result of one grounded question."""

    answer: str
    sources: list[SourceCitation]
    enough_evidence: bool
    retrieved: list[RetrievedChunk]
    provider: str
    model: str
    strategy: str
    strategy_reason: str

    def to_api_dict(self) -> dict:
        """Shape returned by POST /api/chat."""
        return {
            "answer": self.answer,
            "sources": [
                {"document": s.document, "page": s.page} for s in self.sources
            ],
            "enough_evidence": self.enough_evidence,
            "retrieval": {
                "chunks_retrieved": len(self.retrieved),
                "chunks_used": len(self.sources),
                "top_score": max(
                    (c.semantic_score for c in self.retrieved if c.semantic_score is not None),
                    default=0.0,
                ),
            },
            "provider": self.provider,
            "model": self.model,
            "strategy": self.strategy,
            "strategy_reason": self.strategy_reason,
        }


_CITATION_RE = re.compile(r"\[(\d+)\]")

#: Some hosted models (observed with Groq's gpt-oss-20b) emit full-width CJK
#: citation brackets like【1】. Normalize them before citation handling so the
#: grounding guarantee keeps working regardless of bracket style.
_CJK_BRACKETS = str.maketrans({"【": "[", "】": "]"})


def _strip_unsupported_citations(answer: str, block_count: int) -> str:
    """Remove citation markers that do not point at an evidence block.

    Grounding guarantee: the answer may cite [1]..[N] for the N blocks the
    model actually received, nothing else. Removing (rather than rejecting)
    keeps a usable answer when the model adds one stray marker.
    """
    def replace(match: re.Match) -> str:
        index = int(match.group(1))
        return match.group(0) if 1 <= index <= block_count else ""

    return _CITATION_RE.sub(replace, answer)


def _citations_from_context(context: BuiltContext) -> list[SourceCitation]:
    """Citation list derived from the evidence blocks actually used."""
    return [
        SourceCitation(
            document=block.document,
            page=block.page,
            chunk_index=block.chunk_index,
            score=block.score,
        )
        for block in context.blocks
    ]


def answer_question(
    query: str,
    *,
    settings: Settings,
    retriever: VectorRetriever,
    llm: LLMClient,
    router=None,
    hybrid_retriever: object | None = None,
) -> RAGResult:
    """Run the full grounded pipeline for one user query.

    Steps: route the query to a retrieval strategy -> retrieve ranked evidence
    -> build (filter, dedupe, size) context -> either return the controlled
    insufficient-evidence response without calling the LLM, or generate a
    grounded, cited answer.

    Without ``router``/``hybrid_retriever`` the pipeline uses the NORMAL vector
    strategy exactly as in Phase 3.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty")

    if router is not None and hybrid_retriever is not None:
        decision = router.route(query)
        if decision.strategy != STRATEGY_NORMAL:
            chosen_retriever, min_bm25_score = hybrid_retriever, settings.min_bm25_score
        else:
            chosen_retriever, min_bm25_score = retriever, None
    else:
        decision = _DEFAULT_DECISION
        chosen_retriever, min_bm25_score = retriever, None

    retrieved = chosen_retriever.retrieve(query)
    context = build_context(
        retrieved,
        min_score=settings.min_relevance_score,
        max_chars=settings.max_context_chars,
        min_bm25_score=min_bm25_score,
    )

    if not context.enough_evidence:
        return RAGResult(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=[],
            enough_evidence=False,
            retrieved=retrieved,
            provider="none",
            model="none",
            strategy=decision.strategy,
            strategy_reason=decision.reason,
        )

    response = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_query=query,
        context=render_context(context),
    )

    return RAGResult(
        answer=_strip_unsupported_citations(
            response.text.strip().translate(_CJK_BRACKETS), len(context.blocks)
        ),
        sources=_citations_from_context(context),
        enough_evidence=True,
        retrieved=retrieved,
        provider=response.provider,
        model=response.model,
        strategy=decision.strategy,
        strategy_reason=decision.reason,
    )
