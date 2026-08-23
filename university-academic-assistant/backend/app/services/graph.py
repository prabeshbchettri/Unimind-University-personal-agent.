"""Knowledge-graph service (Phase 7).

Orchestrates the pipeline

    StructuredDocument -> Entity Extraction -> Relationship Extraction
    -> Validation -> Graph (Neo4j / in-memory)

and exposes the basic graph queries. The graph is a focused academic model:
only important entities (university, program, semester, subject, topics,
subtopics, past questions) are extracted — never chunks or sentences. Qdrant
(vector similarity) and this graph (explicit relationships) stay separate;
a later phase combines them behind a single retrieval interface.
"""

from __future__ import annotations

from app.graph.extractor import GraphExtractor
from app.graph.schema import (
    deduplicate_nodes,
    deduplicate_relationships,
    validate_relationships,
)
from app.repositories.graph import GraphRepository
from app.services.base import Service
from app.structure.models import StructuredDocument


class KnowledgeGraphService(Service):
    """Extracts, validates and queries the academic knowledge graph."""

    def __init__(self, repository: GraphRepository, extractor: GraphExtractor | None = None) -> None:
        super().__init__()
        self.repository = repository
        self.extractor = extractor or GraphExtractor()

    async def health(self) -> dict:
        try:
            reachable = await self.repository.ping()
        except Exception as exc:  # pragma: no cover - defensive
            return {"name": self.name, "status": "error", "error": str(exc)}
        return {
            "name": self.name,
            "status": "ok" if reachable else "unavailable",
            "backend": type(self.repository).__name__,
        }

    async def index(self, structured: StructuredDocument) -> dict:
        """Extract, validate and merge a document's graph entities."""
        extraction = self.extractor.extract(structured)

        nodes = deduplicate_nodes(extraction.nodes)
        valid, invalid = validate_relationships(extraction.relationships)
        if invalid:
            self.logger.warning(
                "Discarded unsupported graph relationships",
                extra={"extra_fields": {"count": len(invalid), "document_id": structured.document_id}},
            )
        relationships = deduplicate_relationships(valid)

        await self.repository.merge_nodes(nodes)
        await self.repository.merge_relationships(relationships)

        return {
            "document_id": structured.document_id,
            "nodes": len(nodes),
            "relationships": len(relationships),
            "status": "ok",
        }

    async def summary(self) -> dict:
        """Aggregate graph status and entity counts."""
        health = await self.health()
        counts = await self.repository.graph_summary()
        return {
            "status": health["status"],
            "backend": health["backend"],
            **counts,
        }

    async def clear(self) -> None:
        await self.repository.clear()

    # --- Query delegation ---------------------------------------------------

    async def subjects_in_semester(self, semester: int) -> list[dict]:
        return await self.repository.subjects_in_semester(semester)

    async def topics_of_subject(self, subject: str) -> list[dict]:
        return await self.repository.topics_of_subject(subject)

    async def subtopics_of_topic(self, topic: str) -> list[dict]:
        return await self.repository.subtopics_of_topic(topic)

    async def questions_about_topic(self, topic: str) -> list[dict]:
        return await self.repository.questions_about_topic(topic)

    async def subjects_containing_topic(self, topic: str) -> list[dict]:
        return await self.repository.subjects_containing_topic(topic)
