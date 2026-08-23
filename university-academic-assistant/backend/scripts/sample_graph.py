"""Sample Phase 7 demo: Neo4j academic knowledge graph.

Builds a small academic corpus (a syllabus and a past question paper), runs the
full Phase 7 pipeline (extraction -> validation -> graph) using the in-memory
graph backend, and prints the entity counts plus the results of the basic graph
queries.

Offline-safe: uses the in-memory graph, in-memory Qdrant and the deterministic
embedder. Run from the ``backend`` directory:

    python scripts/sample_graph.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.graph.extractor import GraphExtractor
from app.repositories.graph import build_graph_repository
from app.services.graph import KnowledgeGraphService
from app.structure.models import DocumentMetadata, DocumentStructure, Question
from tests.structure_factory import make_structured


def syllabus(document_id: str) -> "StructuredDocument":
    return make_structured(
        ["Database Management System Syllabus\nUnit 1: Introduction\nUnit 2: Normalization\n"],
        document_id=document_id,
        document_type="syllabus",
        metadata=DocumentMetadata(
            document_type="syllabus",
            title="DBMS Syllabus",
            university="Tribhuvan University",
            program="BSc CSIT",
            semester=5,
            subject="Database Management System",
            subject_code="CSIT 325",
            topic="Normalization",
        ),
        structure=DocumentStructure(
            topics=["Introduction", "Normalization", "Transactions"],
            subtopics=[],
        ),
    )


def past_paper(document_id: str) -> "StructuredDocument":
    return make_structured(
        ["DBMS 2080\nQ5. Explain normalization with an example. 10 Marks\n"],
        document_id=document_id,
        document_type="past_question",
        metadata=DocumentMetadata(
            document_type="past_question",
            title="DBMS 2080",
            university="Tribhuvan University",
            program="BSc CSIT",
            semester=5,
            subject="Database Management System",
            academic_year="2080",
            year=2080,
            topic="Normalization",
        ),
        structure=DocumentStructure(
            topics=["Normalization", "SQL"],
            questions=[Question(question_number=5, text="Explain normalization with an example.", marks=10)],
        ),
    )


async def main() -> None:
    repository = build_graph_repository(Settings(graph_backend="memory"))
    graph = KnowledgeGraphService(repository=repository, extractor=GraphExtractor())

    for document in [syllabus("doc-syllabus"), past_paper("doc-past")]:
        result = await graph.index(document)
        print("indexed:", json.dumps(result, ensure_ascii=False))

    print("\n=== Graph summary ===")
    print(json.dumps(await graph.summary(), indent=2, ensure_ascii=False))

    print("\n=== Queries ===")
    for label, rows in [
        ("Subjects in semester 5", await graph.subjects_in_semester(5)),
        ("Topics of Database Management System", await graph.topics_of_subject("Database Management System")),
        ("Subtopics of Normalization", await graph.subtopics_of_topic("Normalization")),
        ("Past questions about Normalization", await graph.questions_about_topic("Normalization")),
        ("Subjects containing topic Normalization", await graph.subjects_containing_topic("Normalization")),
    ]:
        print(f"\n[{label}]")
        for row in rows:
            print("  ", json.dumps(row, ensure_ascii=False))

    await repository.close()


if __name__ == "__main__":
    asyncio.run(main())