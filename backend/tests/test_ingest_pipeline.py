"""End-to-end ingestion pipeline tests (fakes for embeddings and Qdrant).

The pipeline wiring is real: PDF -> chunks -> vectors -> store. Only the
embedding model and the Qdrant server are faked.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.embeddings import EmbeddingError
from app.ingest import ingest_directory
from tests.conftest import FakeEmbedding, FakeQdrant, write_pdf

DOC_A = (
    "The minimum attendance requirement is seventy-five percent for every "
    "registered course. Students below the threshold are ineligible for the "
    "final examination unless condonation is approved by the department."
)
DOC_B = (
    "CS201 Data Structures covers arrays, linked lists, stacks, queues, trees, "
    "hash tables and elementary graph algorithms over four credits in the "
    "third semester."
)


@pytest.fixture
def pipeline_settings(tmp_path) -> Settings:
    return Settings(
        documents_dir=str(tmp_path),
        chunk_size=120,
        chunk_overlap=20,
        qdrant_url="http://fake",
        qdrant_collection="test_docs",
        embedding_dim=64,
        embedding_provider="local",
        _env_file=None,
    )


def seed_documents(tmp_path, contents: dict[str, str]) -> None:
    for name, text in contents.items():
        write_pdf(tmp_path / name, name, [text])


def test_pipeline_ingests_documents_and_returns_report(tmp_path, pipeline_settings) -> None:
    seed_documents(tmp_path, {"attendance.pdf": DOC_A, "syllabus.pdf": DOC_B})

    client = FakeQdrant(dim=64)
    store = _make_store(client, pipeline_settings)
    report = ingest_directory(
        documents_dir=tmp_path,
        settings=pipeline_settings,
        embedding=FakeEmbedding(dim=64),
        store=store,
    )

    assert report.documents == ["attendance.pdf", "syllabus.pdf"]
    assert report.chunk_count >= 2
    assert report.vector_count == report.chunk_count
    assert report.collection_created is True
    assert report.embedding_provider == "fake"
    assert len(client.points) == report.vector_count


def test_pipeline_chunks_carry_full_metadata_in_payload(tmp_path, pipeline_settings) -> None:
    from tests.conftest import assert_payload_keys

    seed_documents(tmp_path, {"attendance.pdf": DOC_A})
    client = FakeQdrant(dim=64)
    store = _make_store(client, pipeline_settings)
    ingest_directory(
        documents_dir=tmp_path,
        settings=pipeline_settings,
        embedding=FakeEmbedding(dim=64),
        store=store,
    )

    for point in client.points.values():
        assert_payload_keys(point["payload"])
        assert point["payload"]["document_name"] == "attendance.pdf"
        assert point["payload"]["page"] == 1
        assert point["payload"]["text"]


def test_pipeline_second_run_overwrites_without_duplicates(tmp_path, pipeline_settings) -> None:
    seed_documents(tmp_path, {"attendance.pdf": DOC_A})
    embedding = FakeEmbedding(dim=64)
    client = FakeQdrant(dim=64)
    store = _make_store(client, pipeline_settings)

    first = ingest_directory(
        documents_dir=tmp_path, settings=pipeline_settings, embedding=embedding, store=store
    )
    second = ingest_directory(
        documents_dir=tmp_path, settings=pipeline_settings, embedding=embedding, store=store
    )

    assert first.vector_count == second.vector_count
    assert len(client.points) == first.vector_count


def test_pipeline_errors_are_typed_not_swallowed(tmp_path, pipeline_settings) -> None:
    empty_dir = tmp_path / "nothing"
    empty_dir.mkdir()
    with pytest.raises(Exception, match="No PDF files found"):
        ingest_directory(
            documents_dir=empty_dir,
            settings=pipeline_settings,
            embedding=FakeEmbedding(dim=64),
            store=_make_store(FakeQdrant(dim=64), pipeline_settings),
        )


def test_pipeline_detects_dimension_mismatch_between_client_and_settings(
    tmp_path, pipeline_settings
) -> None:
    seed_documents(tmp_path, {"attendance.pdf": DOC_A})
    with pytest.raises(EmbeddingError, match="EMBEDDING_DIM"):
        ingest_directory(
            documents_dir=tmp_path,
            settings=pipeline_settings,
            embedding=FakeEmbedding(dim=32),
            store=_make_store(FakeQdrant(dim=32), pipeline_settings),
        )


def _make_store(client: FakeQdrant, settings: Settings):
    from app.vectorstore import QdrantVectorStore

    return QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
        client=client,
    )
