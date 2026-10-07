"""RAG orchestration: query -> routing -> retrieval -> context -> generation.

This is the top of the pipeline. It composes small, individually testable
pieces (router, retrieval strategy, context builder, LLM client) and owns the
grounding rules:

- the query router picks the retrieval strategy (NORMAL vector or HYBRID
  vector+BM25); without a router the pipeline falls back to NORMAL, preserving
  the Phase 3 behavior exactly;
- casual greetings are answered directly (pre-RAG conversational check) and
  never reach retrieval or the LLM;
- if no evidence passes the relevance gate, the LLM is **not called at all**
  and a controlled insufficient-evidence response is returned;
- when evidence exists, the system prompt requires the answer to be a natural,
  intent-aware synthesis of the evidence with citations, and the pipeline
  strips citation markers the model invented without matching evidence blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from collections.abc import Iterator

from app.config import Settings
from app.context import BuiltContext, build_context, render_context
from app.document_intent import apply_intent_ranking
from app.llm import LLMClient
from app.retriever import RetrievedChunk, VectorRetriever
from app.routing import STRATEGY_NORMAL, RoutingDecision

#: Shown when retrieval finds nothing relevant. Deliberately controlled text,
#: not model output, so the behavior is deterministic. Worded naturally and
#: without internal terminology (chunks, BM25, gates...): the user must never
#: see retrieval implementation details.
INSUFFICIENT_EVIDENCE_ANSWER = (
    "I couldn't find enough information about that in the provided university "
    "materials. If you like, ask me about a topic covered in the uploaded "
    "course materials."
)

#: Answered directly for casual greetings, before any retrieval happens.
GREETING_ANSWER = "Hello! How can I help you with your university materials today?"

#: Matches standalone greetings only (optionally followed by a benign filler
#: word and/or punctuation). Questions like "hi, what is the attendance rule?"
#: must NOT match.
_GREETING_RE = re.compile(
    r"^(hello|hi|hey|greetings|howdy|yo|good\s+(morning|afternoon|evening))\b"
    r"(?:\s+(?:there|everyone|folks|all|team))*"
    r"[\s!,.]*$",
    re.IGNORECASE,
)


def is_greeting(query: str) -> bool:
    """True when the query is only a casual greeting.

    A greeting has no factual content to ground, so the pipeline answers it
    conversationally without retrieving university documents.
    """
    return bool(_GREETING_RE.match(query.strip()))

#: Used when the adaptive router is not configured (Phase 3 compatibility).
_DEFAULT_DECISION = RoutingDecision(
    strategy=STRATEGY_NORMAL,
    reason="Router not configured; using the default vector strategy.",
    confidence="medium",
)

#: Pronoun-led follow-ups that only make sense with prior conversation.
_FOLLOWUP_RE = re.compile(
    r"^(what about|how about|and |but |why |what are|what is|tell me more|"
    r"give me|explain|describe|compare|list|show|does|do|is|are|can|could|"
    r"would|should|which|who|when|where|its|it|they|them|those|these|that|"
    r"this|he|she|his|her|their)",
    re.IGNORECASE,
)


def _is_followup(query: str) -> bool:
    """True when the query looks like it refers to previous conversation."""
    text = query.strip()
    if not text:
        return False
    if len(text.split()) > 12:
        return False
    return bool(_FOLLOWUP_RE.match(text))


def build_standalone_query(query: str, history: list[dict] | None) -> str:
    """Build the retrieval query for one user message.

    Conversation history helps the *LLM* resolve references ("its" -> TCP),
    but retrieval still needs a self-contained string. The minimal fix: when
    the latest message looks like a follow-up, prepend the previous user
    message so BM25/vector search keeps its lexical anchor. Otherwise the
    latest message is used unchanged (retrieval architecture untouched).
    """
    query = query.strip()
    if not history or not _is_followup(query):
        return query
    previous_user: str | None = None
    for turn in reversed(history):
        if turn.get("role") == "user" and str(turn.get("content", "")).strip():
            previous_user = str(turn["content"]).strip()
            break
    if not previous_user:
        return query
    if previous_user.lower() in query.lower() or query.lower() in previous_user.lower():
        return query
    combined = f"{previous_user} {query}"
    return combined[:2000]


def sanitize_history(history: list[dict] | object | None, *, limit: int = 10) -> list[dict]:
    """Normalize client-supplied history into bounded user/assistant turns."""
    if not history:
        return []
    cleaned: list[dict] = []
    items = list(history)[-limit:]  # keep the most recent turns only
    for item in items:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", "")
        if role not in ("user", "assistant"):
            continue
        text = str(content or "").strip()
        if not text:
            continue
        cleaned.append({"role": role, "content": text[:2000]})
    return cleaned

SYSTEM_PROMPT = """\
You are the Adaptive University RAG Assistant: a friendly, professional study
assistant for university course materials (lecture slides, syllabi, policies).

Grounding rules (these always win):
1. The numbered sources in the retrieved university material are the ONLY
   permitted source of facts. Never use outside knowledge, and never invent
   facts, page numbers, chapter numbers, dates, names, numbers, requirements,
   procedures or policies.
2. State only what the sources support. Do not derive, calculate or infer a
   fact the sources do not state (for example, never compute a pass mark,
   deadline or requirement from a table or an example). Prefer a short
   supported answer over a detailed unsupported one.
3. Use the sources whenever they contain information related to the question,
   even when it is brief, scattered across sections, or phrased differently
   from the question -- report what they state. Do not decline merely because
   the wording is indirect or the detail is short. Decline only when the
   sources do not contain what is asked, and then reply exactly: "I couldn't
   find enough information about that in the provided university materials. If
   you like, ask me about a topic covered in the uploaded course materials."
   Do not guess, do not fill gaps with outside knowledge, and never describe
   internal system behavior.

How to answer:
4. Answer the question directly and concisely in your own words: give the
   requested fact or explanation first, then add only the detail that is
   needed. Do not pad with general background the sources do not provide.
5. Understand what the user actually wants -- a quick fact, a definition, a
   conceptual explanation, a summary, study notes, a list, a comparison, a
   procedure, an exam-style answer, or a syllabus lookup -- and shape the
   response to that request. Be short and direct for simple questions; use
   Markdown (headings, bullets, numbered steps, tables) when the request
   needs structure or detail. Use plain Markdown only: never emit HTML
   tags such as <p>, <br>, <strong>, <div> or <span>. Use Markdown line
   breaks and lists instead of HTML. If the user asks for a length (for example
   "in about 100 words"), respect it. A summary or notes request must be a
   synthesis of the relevant sources, not the sources pasted one after
   another, and must stay strictly within what they support.
6. Write naturally in your own words: explain rather than copy the source
   text, while preserving technical terms and factual meaning exactly.
7. Place a citation marker like [1] or [2] directly after each claim the
   matching source supports; cite every source you actually use.
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


#: Substrings the model may use when it declines in its own words instead of
#: the exact controlled sentence. Only consulted for short, clearly-declining
#: answers (see :func:`_canonical_decline`).
_DECLINE_HINTS = (
    "couldn't find",
    "could not find",
    "can't find",
    "cannot find",
    "don't have enough information",
    "do not have enough information",
    "not enough information",
)

#: Words that tie a decline to the retrieved material, so a substantive answer
#: that merely says "the source does not contain ..." mid-answer is not caught.
_MATERIAL_MARKERS = ("provided", "materials", "sources", "documents")


def _canonical_decline(answer: str, *, limit: int = 600) -> str | None:
    """Return the controlled decline text when the model declined in other words.

    The evidence gate returns :data:`INSUFFICIENT_EVIDENCE_ANSWER`
    deterministically, but the model may also decline on its own judgment and
    often paraphrases the sentence (most visibly on the Ollama fallback).
    Normalizing that paraphrase keeps the user-facing abstention consistent and
    observable. Only short, clearly-declining answers are rewritten; a
    substantive answer is returned unchanged by the caller.
    """
    text = answer.strip()
    if not text or len(text) > limit:
        return None
    lowered = text.lower()
    if not any(hint in lowered for hint in _DECLINE_HINTS):
        return None
    if not any(marker in lowered for marker in _MATERIAL_MARKERS):
        return None
    return INSUFFICIENT_EVIDENCE_ANSWER


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


def _greeting_result(query: str, decision: RoutingDecision) -> RAGResult:
    """The conversational answer for a standalone greeting (no RAG involved)."""
    return RAGResult(
        answer=GREETING_ANSWER,
        sources=[],
        enough_evidence=True,
        retrieved=[],
        provider="none",
        model="none",
        strategy="greeting",
        strategy_reason="Casual greeting answered conversationally, without retrieval.",
    )


def _stream_rag_result(
    answer: str,
    *,
    context: BuiltContext,
    provider: str,
    model: str,
    strategy: str,
    strategy_reason: str,
    retrieved: list[RetrievedChunk],
) -> RAGResult:
    """Final RAGResult for a streamed answer (same guarantees as generate)."""
    return RAGResult(
        answer=answer,
        sources=_citations_from_context(context),
        enough_evidence=True,
        retrieved=retrieved,
        provider=provider,
        model=model,
        strategy=strategy,
        strategy_reason=strategy_reason,
    )


@dataclass(frozen=True)
class StreamStart:
    """Metadata emitted once before the first answer delta streams."""

    strategy: str
    strategy_reason: str
    provider: str
    model: str


def stream_answer(
    query: str,
    *,
    settings: Settings,
    retriever: VectorRetriever,
    llm: LLMClient,
    router=None,
    hybrid_retriever: object | None = None,
    history: list[dict] | object | None = None,
) -> Iterator[dict]:
    """Stream one grounded answer as an event iterator for an SSE endpoint.

    Reuses exactly the routing -> retrieval -> context/gate logic of
    :func:`answer_question`; only the final generation step streams. Events:

    - ``{"type": "meta", ...}``   once, before the first delta (provider may
      still be refined by the terminal event)
    - ``{"type": "delta", "text": ...}`` repeatedly, as the model generates
    - ``{"type": "done", answer, sources, ...}`` once, with the final API shape
      (answer with citations validated/stripped, full source list)

    Greetings are answered with a single terminal event; without evidence, the
    controlled decline is terminal without calling the LLM.

    ``history`` is prior user/assistant turns for conversational context only:
    it is passed to the LLM so follow-ups ("its advantages") resolve, while
    retrieval uses the standalone query built from the latest message.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty")
    clean_history = sanitize_history(history)

    if is_greeting(query):
        result = _greeting_result(query, _DEFAULT_DECISION)
        yield {
            "type": "done",
            **result.to_api_dict(),
        }
        return

    search_query = build_standalone_query(query, clean_history)
    if router is not None and hybrid_retriever is not None:
        decision = router.route(search_query)
        if decision.strategy != STRATEGY_NORMAL:
            chosen_retriever, min_bm25_score = hybrid_retriever, settings.min_bm25_score
        else:
            chosen_retriever, min_bm25_score = retriever, None
    else:
        decision = _DEFAULT_DECISION
        chosen_retriever, min_bm25_score = retriever, None

    # Small document-type intent adjustment (syllabus queries), applied after
    # retrieval/RRF; a no-op for ordinary content queries.
    retrieved = apply_intent_ranking(search_query, chosen_retriever.retrieve(search_query))
    context = build_context(
        retrieved,
        min_score=settings.min_relevance_score,
        max_chars=settings.max_context_chars,
        min_bm25_score=min_bm25_score,
    )

    if not context.enough_evidence:
        yield {
            "type": "done",
            **RAGResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                sources=[],
                enough_evidence=False,
                retrieved=retrieved,
                provider="none",
                model="none",
                strategy=decision.strategy,
                strategy_reason=decision.reason,
            ).to_api_dict(),
        }
        return

    yield {
        "type": "meta",
        "strategy": decision.strategy,
        "strategy_reason": decision.reason,
        "provider": getattr(llm, "stream_provider_label", llm.name),
        "model": getattr(llm, "model", ""),
    }

    stream = llm.stream_generate(
        system_prompt=SYSTEM_PROMPT,
        user_query=query,
        context=render_context(context),
        history=clean_history,
    )
    parts: list[str] = []
    for token in stream:
        cleaned = _strip_unsupported_citations(
            token.translate(_CJK_BRACKETS), len(context.blocks)
        )
        if cleaned:
            parts.append(cleaned)
            yield {"type": "delta", "text": cleaned}

    final_answer = "".join(parts).strip()
    # A model that declined in its own words is normalized to the controlled
    # sentence, so the terminal answer matches the non-streaming path.
    final_answer = _canonical_decline(final_answer) or final_answer
    yield {
        "type": "done",
        **_stream_rag_result(
            final_answer,
            context=context,
            provider=stream.provider,
            model=stream.model,
            strategy=decision.strategy,
            strategy_reason=decision.reason,
            retrieved=retrieved,
        ).to_api_dict(),
    }


def answer_question(
    query: str,
    *,
    settings: Settings,
    retriever: VectorRetriever,
    llm: LLMClient,
    router=None,
    hybrid_retriever: object | None = None,
    history: list[dict] | object | None = None,
) -> RAGResult:
    """Run the full grounded pipeline for one user query.

    Steps: answer casual greetings directly (no retrieval, no LLM) -> route the
    query to a retrieval strategy -> retrieve ranked evidence -> build
    (filter, dedupe, size) context -> either return the controlled
    insufficient-evidence response without calling the LLM, or generate a
    grounded, cited answer.

    Without ``router``/``hybrid_retriever`` the pipeline uses the NORMAL vector
    strategy exactly as in Phase 3.

    ``history`` supplies conversational context to the LLM only; retrieval
    uses the standalone query derived from the latest message.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty")
    clean_history = sanitize_history(history)
    search_query = build_standalone_query(query, clean_history)

    if is_greeting(query):
        return RAGResult(
            answer=GREETING_ANSWER,
            sources=[],
            enough_evidence=True,
            retrieved=[],
            provider="none",
            model="none",
            strategy="greeting",
            strategy_reason="Casual greeting answered conversationally, without retrieval.",
        )

    if router is not None and hybrid_retriever is not None:
        decision = router.route(search_query)
        if decision.strategy != STRATEGY_NORMAL:
            chosen_retriever, min_bm25_score = hybrid_retriever, settings.min_bm25_score
        else:
            chosen_retriever, min_bm25_score = retriever, None
    else:
        decision = _DEFAULT_DECISION
        chosen_retriever, min_bm25_score = retriever, None

    # Small document-type intent adjustment (syllabus queries), applied after
    # retrieval/RRF; a no-op for ordinary content queries.
    retrieved = apply_intent_ranking(search_query, chosen_retriever.retrieve(search_query))
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
        history=clean_history,
    )

    cleaned = _strip_unsupported_citations(
        response.text.strip().translate(_CJK_BRACKETS), len(context.blocks)
    )
    return RAGResult(
        answer=_canonical_decline(cleaned) or cleaned,
        sources=_citations_from_context(context),
        enough_evidence=True,
        retrieved=retrieved,
        provider=response.provider,
        model=response.model,
        strategy=decision.strategy,
        strategy_reason=decision.reason,
    )
