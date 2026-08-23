"""Tests for Phase 7: Neo4j academic knowledge graph.

Covers deterministic extraction, schema validation, duplicate handling, source
traceability, the in-memory and Neo4j repositories, the semantic graph queries,
and the vector + graph integration (Qdrant remains unchanged).
"""

import asyncio

import pytest

from tests import rag_factory as factory
from tests.structure_factory import make_structured

from app.config import Settings
from app.graph.extractor import GraphExtractor
from app.graph.models import GraphNode, GraphRelationship
from app.graph.schema import (
    ALLOWED_RELATIONSHIPS,
    is_supported_relationship,
    node_key,
    validate_relationships,
)
from app.repositories.graph import (
    GraphUnavailableError,
    InMemoryGraphRepository,
    Neo4jGraphRepository,
    build_graph_repository,
)
from app.services.graph import KnowledgeGraphService
from app.structure.models import DocumentMetadata, DocumentStructure


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_syllabus_extracts_core_entities() -> None:
    extraction = GraphExtractor().extract(factory.make_graph_syllabus())
    labels = {node.label for node in extraction.nodes}
    assert labels == {"University", "Program", "Semester", "Subject", "Topic"}

    rel_types = {(rel.type, rel.start_label, rel.end_label) for rel in extraction.relationships}
    assert ("HAS_PROGRAM", "University", "Program") in rel_types
    assert ("HAS_SEMESTER", "Program", "Semester") in rel_types
    assert ("HAS_SUBJECT", "Semester", "Subject") in rel_types
    assert ("HAS_TOPIC", "Subject", "Topic") in rel_types
    assert all((start, rtype, end) in ALLOWED_RELATIONSHIPS for rtype, start, end in rel_types)


def test_past_paper_extracts_questions_with_relationships() -> None:
    extraction = GraphExtractor().extract(factory.make_graph_past_paper())
    questions = [node for node in extraction.nodes if node.label == "PastQuestion"]
    assert len(questions) == 1
    question = questions[0]
    assert question.key == "doc-past:5"
    assert question.props["question_number"] == 5
    assert question.props["marks"] == 10
    assert question.props["year"] == 2080
    assert "normalization" in question.props["text"].lower()

    rel_types = {(rel.type, rel.start_label, rel.end_label) for rel in extraction.relationships}
    assert ("HAS_QUESTION", "Subject", "PastQuestion") in rel_types
    assert ("ABOUT", "PastQuestion", "Topic") in rel_types


def test_source_traceability_preserved() -> None:
    extraction = GraphExtractor().extract(factory.make_graph_syllabus())
    for node in extraction.nodes:
        assert node.props["document_id"] == "doc-syllabus"
        assert node.props["source_type"] == "syllabus"
        assert node.props["page"] >= 1
    for rel in extraction.relationships:
        assert rel.props["document_id"] == "doc-syllabus"
        assert rel.props["source_type"] == "syllabus"


def test_page_traceability_uses_content_page() -> None:
    structured = make_structured(
        ["Cover page only.",
         "Unit 2: Normalization\nNormalization removes redundancy in tables."],
        document_id="doc-x",
        document_type="syllabus",
        metadata=DocumentMetadata(
            document_type="syllabus",
            title="DBMS Syllabus",
            subject="Database Management System",
        ),
        structure=DocumentStructure(topics=["Normalization"]),
    )
    extraction = GraphExtractor().extract(structured)
    topic = next(node for node in extraction.nodes if node.label == "Topic")
    assert topic.props["page"] == 2


def test_subtopic_from_metadata_pair() -> None:
    structured = factory.make_graph_syllabus(topics=["Normalization"], subtopics=["Normal Forms"])
    extraction = GraphExtractor().extract(structured)
    subtopic = [node for node in extraction.nodes if node.label == "Subtopic"]
    assert len(subtopic) == 1
    assert subtopic[0].props["name"] == "Normal Forms"
    rels = [rel for rel in extraction.relationships if rel.type == "HAS_SUBTOPIC"]
    assert len(rels) == 1
    assert rels[0].start_key == node_key("Topic", "Normalization")
    assert rels[0].end_key == node_key("Subtopic", "Normal Forms")


def test_single_topic_absorbs_flat_subtopics() -> None:
    structured = make_structured(
        ["Normalization\n1NF\n2NF\n3NF"],
        document_id="doc-y",
        document_type="syllabus",
        metadata=DocumentMetadata(
            document_type="syllabus",
            title="DBMS Syllabus",
            subject="Database Management System",
        ),
        structure=DocumentStructure(topics=["Normalization"], subtopics=["1NF", "2NF", "3NF"]),
    )
    extraction = GraphExtractor().extract(structured)
    subtopics = [node for node in extraction.nodes if node.label == "Subtopic"]
    assert {node.props["name"] for node in subtopics} == {"1NF", "2NF", "3NF"}
    rels = [rel for rel in extraction.relationships if rel.type == "HAS_SUBTOPIC"]
    assert len(rels) == 3
    assert all(rel.start_key == node_key("Topic", "Normalization") for rel in rels)


def test_duplicate_topics_collapsed_within_document() -> None:
    extraction = GraphExtractor().extract(
        factory.make_graph_syllabus(topics=["Normalization", "Normalization", "SQL"])
    )
    topics = [node for node in extraction.nodes if node.label == "Topic"]
    assert len(topics) == 2
    rels = [rel for rel in extraction.relationships if rel.type == "HAS_TOPIC"]
    assert len(rels) == 2


def test_no_graph_entities_without_academic_metadata() -> None:
    structured = make_structured(
        ["Notice: classes cancelled on 2024-05-01."],
        document_id="doc-n",
        document_type="notice",
        metadata=DocumentMetadata(document_type="notice", title="May Notice"),
        structure=DocumentStructure(),
    )
    extraction = GraphExtractor().extract(structured)
    assert extraction.nodes == []
    assert extraction.relationships == []


def test_questions_fallback_from_metadata() -> None:
    structured = make_structured(
        ["DBMS 2080\nQ5. Normalization"],
        document_id="doc-q",
        document_type="past_question",
        metadata=DocumentMetadata(
            document_type="past_question",
            title="DBMS 2080",
            subject="Database Management System",
            question_number=5,
            marks=10,
        ),
        structure=DocumentStructure(),
    )
    extraction = GraphExtractor().extract(structured)
    questions = [node for node in extraction.nodes if node.label == "PastQuestion"]
    assert len(questions) == 1
    assert questions[0].props["question_number"] == 5


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_unsupported_relationships_rejected() -> None:
    valid, invalid = validate_relationships(
        [
            GraphRelationship(type="HAS_TOPIC", start_label="Subject", start_key="s", end_label="Topic", end_key="t"),
            GraphRelationship(type="WRITES", start_label="Student", start_key="s", end_label="Exam", end_key="e"),
            GraphRelationship(type="HAS_SUBTOPIC", start_label="Subject", start_key="s", end_label="Subtopic", end_key="st"),
        ]
    )
    assert len(valid) == 1
    assert len(invalid) == 2
    assert is_supported_relationship(valid[0])
    assert not is_supported_relationship(invalid[0])
    # Endpoints are checked via the allowed triples, never silently accepted.
    assert invalid[1].type == "HAS_SUBTOPIC"


def test_node_keys_are_stable() -> None:
    assert node_key("Subject", "Database Management System") == node_key("Subject", "database management system")
    assert node_key("Semester", 5) == "sem:5"
    assert node_key("PastQuestion", ("doc-1", 5)) == "doc-1:5"
    with pytest.raises(ValueError):
        node_key("Student", "Alice")


# ---------------------------------------------------------------------------
# Repository (in-memory)
# ---------------------------------------------------------------------------

def _indexed_graph(*structured):
    service = factory.make_graph_service()
    for document in structured:
        _run(service.index(document))
    return service


def test_repository_merges_deduplicates_nodes_and_relationships() -> None:
    service = _indexed_graph(
        factory.make_graph_syllabus(document_id="doc-a"),
        factory.make_graph_syllabus(document_id="doc-b", title="DBMS Syllabus 2"),
    )
    repo = service.repository
    # Same subject/program/university/semester in both documents -> one node each.
    assert _run(repo.node_count("Subject")) == 1
    assert _run(repo.node_count("University")) == 1
    assert _run(repo.node_count("Program")) == 1
    assert _run(repo.node_count("Semester")) == 1
    assert _run(repo.node_count("Topic")) == 3
    # Relationships deduplicated by identity too.
    assert _run(repo.relationship_count("HAS_SUBJECT")) == 1
    assert _run(repo.relationship_count("HAS_TOPIC")) == 3


def test_repository_skips_dangling_relationships() -> None:
    repo = factory.make_graph_repository()
    _run(repo.merge_nodes([GraphNode(label="Topic", key="t", props={"name": "t"})]))
    _run(
        repo.merge_relationships(
            [
                GraphRelationship(type="HAS_SUBTOPIC", start_label="Topic", start_key="t", end_label="Subtopic", end_key="st"),
            ]
        )
    )
    assert _run(repo.relationship_count()) == 0


def test_repository_clear() -> None:
    service = _indexed_graph(factory.make_graph_syllabus())
    _run(service.repository.clear())
    assert _run(service.repository.node_count()) == 0
    assert _run(service.repository.relationship_count()) == 0


def test_graph_summary_counts() -> None:
    service = _indexed_graph(
        factory.make_graph_syllabus(),
        factory.make_graph_past_paper(),
    )
    summary = _run(service.repository.graph_summary())
    assert summary["nodes"]["Subject"] == 1
    assert summary["nodes"]["Topic"] == 4  # 3 syllabus topics + 1 ABOUT target
    assert summary["nodes"]["PastQuestion"] == 1
    assert summary["relationships"]["HAS_QUESTION"] == 1
    assert summary["relationships"]["ABOUT"] == 1


# ---------------------------------------------------------------------------
# Semantic queries
# ---------------------------------------------------------------------------

def _query_graph():
    return _indexed_graph(
        factory.make_graph_syllabus(topics=["Introduction", "Normalization", "Transactions"]),
        factory.make_graph_past_paper(questions=[(5, "Explain normalization with an example.", 10)]),
    )


def test_subjects_in_semester() -> None:
    results = _run(_query_graph().subjects_in_semester(5))
    assert [result["name"] for result in results] == ["Database Management System"]
    assert results[0]["code"] == "CSIT 325"
    assert results[0]["document_id"] == "doc-syllabus"


def test_topics_of_subject() -> None:
    topics = _run(_query_graph().topics_of_subject("Database Management System"))
    assert {topic["name"] for topic in topics} == {"Introduction", "Normalization", "Transactions"}
    assert all(topic["source_type"] == "syllabus" for topic in topics)


def test_subtopics_of_topic() -> None:
    structured = make_structured(
        ["Normalization\n1NF\n2NF\n3NF"],
        document_id="doc-y",
        document_type="syllabus",
        metadata=DocumentMetadata(
            document_type="syllabus",
            title="DBMS Syllabus",
            subject="Database Management System",
        ),
        structure=DocumentStructure(topics=["Normalization"], subtopics=["1NF", "2NF", "3NF"]),
    )
    service = factory.make_graph_service()
    _run(service.index(structured))
    subtopics = _run(service.subtopics_of_topic("Normalization"))
    assert {subtopic["name"] for subtopic in subtopics} == {"1NF", "2NF", "3NF"}
    assert _run(service.subtopics_of_topic("SQL")) == []


def test_questions_about_topic() -> None:
    questions = _run(_query_graph().questions_about_topic("Normalization"))
    assert len(questions) == 1
    assert questions[0]["question_number"] == 5
    assert questions[0]["year"] == 2080
    assert questions[0]["marks"] == 10
    assert questions[0]["document_id"] == "doc-past"
    assert questions[0]["source_type"] == "past_question"
    assert _run(_query_graph().questions_about_topic("Physics")) == []


def test_subjects_containing_topic() -> None:
    subjects = _run(_query_graph().subjects_containing_topic("Normalization"))
    assert [subject["name"] for subject in subjects] == ["Database Management System"]
    assert _run(_query_graph().subjects_containing_topic("Quantum Physics")) == []


def test_service_summary_and_health() -> None:
    service = _indexed_graph(factory.make_graph_syllabus())
    summary = _run(service.summary())
    assert summary["status"] == "ok"
    assert summary["backend"] == "InMemoryGraphRepository"
    assert summary["nodes"]["Subject"] == 1
    health = _run(service.health())
    assert health["status"] == "ok"


# ---------------------------------------------------------------------------
# Vector + graph integration (Qdrant unchanged)
# ---------------------------------------------------------------------------

def test_vector_indexing_also_populates_graph() -> None:
    graph_service = factory.make_graph_service()
    service = factory.make_indexing_service(sparse_index=factory.make_sparse_index(), graph_service=graph_service)
    result = _run(service.index(factory.make_graph_syllabus()))
    assert result["graph"]["status"] == "ok"
    assert result["graph"]["nodes"] >= 7
    assert result["graph"]["relationships"] >= 4
    # Qdrant still works exactly as before.
    assert result["collections"]["university_docs"] >= 1
    hits = _run(service.search("Database Management System", top_k=3))
    assert hits
    assert hits[0].metadata["document_type"] == "syllabus"
    # And the graph is queryable alongside the vector store.
    subjects = _run(graph_service.subjects_in_semester(5))
    assert [subject["name"] for subject in subjects] == ["Database Management System"]


def test_graph_failure_does_not_break_vector_indexing(monkeypatch) -> None:
    graph_service = factory.make_graph_service()
    service = factory.make_indexing_service(graph_service=graph_service)

    def _boom(*args, **kwargs):
        raise GraphUnavailableError("Neo4j unreachable")

    monkeypatch.setattr(graph_service, "index", _boom)
    result = _run(service.index(factory.make_graph_syllabus()))
    assert result["graph"]["status"] == "unavailable"
    assert result["chunk_count"] >= 1
    assert result["collections"]["university_docs"] >= 1


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_index_endpoint_populates_graph_summary(client, make_text_pdf, tmp_path) -> None:
    path = make_text_pdf(
        tmp_path / "graph_syllabus.pdf",
        ["Database Management System Syllabus\nUnit 1: Introduction\nUnit 2: Normalization\n"],
    )
    with open(path, "rb") as handle:
        response = client.post(
            "/documents/index",
            files={"file": ("graph_syllabus.pdf", handle, "application/pdf")},
        )
    assert response.status_code == 200
    assert response.json()["graph"]["status"] == "ok"

    summary = client.get("/graph/summary").json()
    assert summary["status"] == "ok"
    assert summary["backend"] == "InMemoryGraphRepository"
    assert summary["nodes"]["Subject"] == 1
    assert summary["relationships"]["HAS_TOPIC"] >= 2


# ---------------------------------------------------------------------------
# Neo4j repository (no server required)
# ---------------------------------------------------------------------------

def test_neo4j_repository_ping_returns_false_when_unreachable() -> None:
    repo = Neo4jGraphRepository(uri="bolt://localhost:1", username="neo4j", password="x", timeout=0.5)
    assert _run(repo.ping()) is False
    _run(repo.close())


def test_neo4j_repository_raises_unavailable_on_write() -> None:
    repo = Neo4jGraphRepository(uri="bolt://localhost:1", username="neo4j", password="x", timeout=0.5)
    with pytest.raises(GraphUnavailableError):
        _run(repo.merge_nodes([GraphNode(label="Topic", key="t", props={"name": "t"})]))
    _run(repo.close())


def test_build_graph_repository_backend_selection() -> None:
    memory = build_graph_repository(Settings(graph_backend="memory"))
    assert isinstance(memory, InMemoryGraphRepository)
    auto_memory = build_graph_repository(Settings(graph_backend="auto", neo4j_uri=":memory:"))
    assert isinstance(auto_memory, InMemoryGraphRepository)
    neo4j = build_graph_repository(Settings(graph_backend="neo4j"))
    assert isinstance(neo4j, Neo4jGraphRepository)
    _run(memory.close())
    _run(neo4j.close())