"""Syllabus topic identification (Phase 9).

The recommendation flow starts by mapping the user query to the university
syllabus. This is done with retrieval evidence, never hard-coded mappings:

1. Expand common course abbreviations (e.g. ``DBMS`` -> ``Database
   Management System``) so shorthand queries still match the indexed text.
2. Embed the query and search the ``university_docs`` collection.
3. A syllabus chunk is **evidence** when its cosine similarity is at or above
   ``min_score`` OR it shares at least one meaningful token with the query
   (the BM25-style signal — this keeps short syllabi evidence even when the
   crude token-hash embedder scores them near zero).
4. The required topics are the union of the evidence chunks' ``topic``
   payloads and — when a subject is identified and the knowledge graph has
   it — the graph's topic list for that subject (the graph is authoritative
   for syllabus topics).

When no chunk is evidence, ``SyllabusContext.no_syllabus_evidence`` is True
and no recommendation can be made.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.repositories.qdrant import QdrantRepository

# Common course abbreviations (general academic shorthand, not university
# metadata). Expandable as new abbreviations appear in the indexed corpus.
SUBJECT_ALIASES: dict[str, str] = {
    "dbms": "Database Management System",
    "dsa": "Data Structures and Algorithms",
}

# Words that carry no topic signal for the shared-token evidence rule.
_STOPWORDS = {
    "which", "what", "where", "when", "how", "who",
    "book", "books", "is", "are", "was", "were", "am",
    "best", "good", "for", "of", "the", "a", "an",
    "recommend", "recommends", "recommended", "covers", "cover",
    "most", "my", "our", "i", "do", "does", "use", "should",
    "me", "to", "in", "on", "about", "with", "and", "or",
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {match.group() for match in _TOKEN_RE.finditer(text.lower())} - _STOPWORDS


def expand_aliases(query: str) -> str:
    """Replace known course abbreviations in ``query`` with full names."""
    lowered = query.lower()
    for alias, canonical in SUBJECT_ALIASES.items():
        if re.search(rf"\b{alias}\b", lowered):
            query = re.sub(rf"\b{alias}\b", canonical, query, flags=re.IGNORECASE)
    return query


@dataclass
class SyllabusContext:
    """The required topics identified from the syllabus for a query."""

    query: str
    subject: str | None = None
    required_topics: list[str] = field(default_factory=list)
    required_subtopics: list[str] = field(default_factory=list)
    no_syllabus_evidence: bool = False
    reason: str = ""


class TopicResolver:
    """Identifies required syllabus topics for a query using retrieval."""

    def __init__(
        self,
        embedder,
        repository: QdrantRepository,
        graph_service=None,
        syllabus_collection: str = "university_docs",
        top_k: int = 10,
        min_score: float = 0.01,
    ) -> None:
        self.embedder = embedder
        self.repository = repository
        self.graph_service = graph_service
        self.syllabus_collection = syllabus_collection
        self.top_k = top_k
        self.min_score = min_score

    async def identify(self, query: str) -> SyllabusContext:
        query = expand_aliases(query.strip())
        query_tokens = _tokens(query)
        vector = self.embedder.embed_one(query)
        try:
            hits = await self.repository.search(self.syllabus_collection, vector, top_k=self.top_k)
        except ValueError:
            # Nothing indexed yet (collection does not exist): no evidence.
            return SyllabusContext(
                query=query,
                no_syllabus_evidence=True,
                reason="No syllabus evidence matched the query.",
            )
        syllabus_hits = [
            hit
            for hit in hits
            if hit.payload.get("document_type") == "syllabus" and self._is_evidence(hit, query_tokens)
        ]

        if not syllabus_hits:
            return SyllabusContext(
                query=query,
                no_syllabus_evidence=True,
                reason="No syllabus evidence matched the query.",
            )

        subjects = Counter(
            hit.payload["subject"] for hit in syllabus_hits if hit.payload.get("subject")
        )
        subject = subjects.most_common(1)[0][0] if subjects else None

        # Subject expansion: once a subject is identified, every syllabus
        # chunk of that subject is evidence (they belong to the same indexed
        # syllabus), so the required topics cover the whole subject, not just
        # the token overlap with the query.
        expanded_hits = [
            hit for hit in hits
            if hit.payload.get("document_type") == "syllabus"
            and (subject is None or hit.payload.get("subject") == subject)
        ]
        if subject:
            syllabus_hits = expanded_hits

        topics = _distinct(hit.payload.get("topic") for hit in syllabus_hits)
        subtopics = _distinct(hit.payload.get("subtopic") for hit in syllabus_hits)

        graph_topics: list[str] = []
        if subject and self.graph_service is not None:
            try:
                graph_topics = [row["name"] for row in await self.graph_service.topics_of_subject(subject)]
            except Exception:
                graph_topics = []
        topics = _distinct([*topics, *graph_topics])

        # The graph is authoritative for syllabus subtopics too: syllabi
        # rarely carry subtopic chunk payloads, but the graph extracts
        # subtopics from single-topic structures. Without this, the subtopic
        # term of the coverage formula can never fire.
        if subject and self.graph_service is not None and graph_topics:
            try:
                for topic in graph_topics:
                    rows = await self.graph_service.subtopics_of_topic(topic)
                    subtopics.extend(row.get("name") for row in rows)
            except Exception:
                pass
        subtopics = _distinct(subtopics)

        if not topics and subject:
            # Last-resort fallback: compare against the subject name itself.
            topics = [subject]

        reason = (
            f"Identified subject {subject!r} with {len(topics)} required topic(s) "
            f"from {len(syllabus_hits)} syllabus chunk(s)."
            if subject
            else f"Identified {len(topics)} required topic(s) from {len(syllabus_hits)} syllabus chunk(s)."
        )
        return SyllabusContext(
            query=query,
            subject=subject,
            required_topics=topics,
            required_subtopics=subtopics,
            reason=reason,
        )

    def _is_evidence(self, hit, query_tokens: set[str]) -> bool:
        """A syllabus chunk is evidence by cosine floor or shared token."""
        if hit.score >= self.min_score:
            return True
        chunk_tokens = _tokens(hit.payload.get("text") or "")
        return bool(query_tokens & chunk_tokens)


def _distinct(values) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen