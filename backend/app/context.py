"""Context construction: select, deduplicate and size retrieved evidence.

The API layer never assembles LLM context itself; this module owns the rules
for what the model is allowed to see:

- drop chunks below the relevance threshold (insufficient-evidence guard) --
  a chunk with a semantic (cosine) score is gated by ``min_score``; a chunk
  found only lexically by the HYBRID strategy is gated by ``min_bm25_score``
- drop near-duplicate text (same content retrieved twice via overlap)
- respect a maximum context budget
- keep the citation metadata for every chunk that is kept

Chunks arrive ranked best-first (vector score for NORMAL, fused RRF score for
HYBRID); relevance gating is per-chunk because for HYBRID the fused order need
not be monotonic in the gate score.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retriever import RetrievedChunk


@dataclass(frozen=True)
class ContextBlock:
    """One evidence block handed to the LLM, with its citation metadata."""

    text: str
    document: str
    page: int
    chunk_index: int
    score: float


@dataclass(frozen=True)
class BuiltContext:
    """The assembled context plus the evidence status used by the RAG layer."""

    blocks: list[ContextBlock]
    enough_evidence: bool
    total_chars: int


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _is_near_duplicate(candidate: str, kept: list[str]) -> bool:
    """True when ``candidate`` is contained in (or contains) a kept block.

    Overlapping chunks legitimately retrieve the same passage twice; the
    duplicated copy adds no information and would waste context budget.
    """
    normalized = _normalized(candidate)
    return any(
        normalized in _normalized(existing) or _normalized(existing) in normalized
        for existing in kept
    )


def _passes_relevance_gate(
    chunk: RetrievedChunk,
    *,
    min_semantic_score: float,
    min_bm25_score: float | None,
) -> bool:
    """Decide whether one retrieved chunk may enter the LLM context.

    NORMAL strategy chunks always carry ``semantic_score`` and are gated by
    cosine similarity. HYBRID lexical-only chunks (no semantic score) are gated
    by the BM25 threshold; a chunk with neither score is rejected.
    """
    if chunk.semantic_score is not None:
        return chunk.semantic_score >= min_semantic_score
    if chunk.bm25_score is not None and min_bm25_score is not None:
        return chunk.bm25_score >= min_bm25_score
    return False


def build_context(
    chunks: list[RetrievedChunk],
    *,
    min_score: float,
    max_chars: int,
    min_bm25_score: float | None = None,
) -> BuiltContext:
    """Turn ranked retrieved chunks into a sized, deduplicated context.

    - chunks that fail the relevance gate are dropped (see
      :func:`_passes_relevance_gate`)
    - near-duplicates of already-selected text are skipped
    - ``max_chars`` caps the total evidence text
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    blocks: list[ContextBlock] = []
    kept_texts: list[str] = []
    total = 0

    for chunk in chunks:
        if not _passes_relevance_gate(
            chunk, min_semantic_score=min_score, min_bm25_score=min_bm25_score
        ):
            continue  # not relevant enough (for this strategy)
        if _is_near_duplicate(chunk.text, kept_texts):
            continue
        if total + len(chunk.text) > max_chars:
            continue  # does not fit; try the next (shorter) chunk
        blocks.append(
            ContextBlock(
                text=chunk.text,
                document=chunk.document,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
            )
        )
        kept_texts.append(chunk.text)
        total += len(chunk.text)
        if total >= max_chars:
            break

    return BuiltContext(blocks=blocks, enough_evidence=bool(blocks), total_chars=total)


def render_context(context: BuiltContext) -> str:
    """Render the context blocks as numbered, citable evidence sections."""
    if not context.blocks:
        return ""
    sections = [
        f"[{i}] source: {block.document}, page {block.page}\n{block.text}"
        for i, block in enumerate(context.blocks, start=1)
    ]
    return "\n\n".join(sections)


def truncate_for_prompt(text: str, max_chars: int) -> str:
    """Hard-cap any text at ``max_chars`` with a visible ellipsis marker."""
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"
