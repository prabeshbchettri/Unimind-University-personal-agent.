"""Tests for the document upload endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient


def test_upload_text_pdf_returns_normalized_document(client: TestClient, text_pdf: Path) -> None:
    with open(text_pdf, "rb") as handle:
        response = client.post(
            "/documents/upload",
            files={"file": ("syllabus_sample.pdf", handle, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["extraction_method"] == "pymupdf"
    assert data["page_count"] == 2
    assert len(data["pages"]) == 2
    assert data["pages"][0]["page_number"] == 1
    assert data["document_id"]


def test_upload_non_pdf_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notice.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_invalid_pdf_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")},
    )
    assert response.status_code == 400


def test_documents_list_still_pending(client: TestClient) -> None:
    assert client.get("/documents").status_code == 501


def test_analyze_returns_structured_document(client: TestClient, text_pdf: Path) -> None:
    with open(text_pdf, "rb") as handle:
        response = client.post(
            "/documents/analyze",
            files={"file": ("syllabus_sample.pdf", handle, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"]
    assert data["metadata"]["document_type"] == "syllabus"
    assert data["metadata"]["subject"] == "Database Management Systems"
    assert data["page_count"] == 2
    assert len(data["pages"]) == 2


def test_analyze_non_pdf_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents/analyze",
        files={"file": ("notice.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415


def test_index_returns_chunks_and_collection(client: TestClient, make_text_pdf, tmp_path: Path) -> None:
    path = make_text_pdf(
        tmp_path / "syllabus.pdf",
        [
            "Database Management System Syllabus\n"
            "Unit 1: Introduction\nDatabases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.",
        ],
    )
    with open(path, "rb") as handle:
        response = client.post(
            "/documents/index",
            files={"file": ("syllabus.pdf", handle, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"]
    assert data["document_type"] == "syllabus"
    assert data["chunk_count"] >= 1
    assert "university_docs" in data["collections"]


def test_index_then_search(client: TestClient, make_text_pdf, tmp_path: Path) -> None:
    path = make_text_pdf(
        tmp_path / "syllabus.pdf",
        [
            "Database Management System Syllabus\n"
            "Unit 1: Introduction\nDatabases store data.\n"
            "Unit 2: Normalization\nNormalization removes redundancy in tables.",
        ],
    )
    with open(path, "rb") as handle:
        index = client.post(
            "/documents/index",
            files={"file": ("syllabus.pdf", handle, "application/pdf")},
        )
    assert index.status_code == 200

    response = client.get("/search", params={"q": "normalization removes redundancy", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "normalization removes redundancy"
    assert payload["results"]
    top = payload["results"][0]
    assert "text" in top and "score" in top and "metadata" in top
    assert top["metadata"]["document_type"] == "syllabus"
    assert "Normalization removes redundancy in tables." in top["text"]
    assert top["score"] >= 0


def test_search_empty_query_rejected(client: TestClient) -> None:
    assert client.get("/search", params={"q": ""}).status_code == 422