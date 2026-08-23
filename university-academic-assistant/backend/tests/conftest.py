"""Shared pytest fixtures.

Tests run in an isolated, dependency-free environment: no external services
are required. Qdrant is used in-memory, the deterministic (offline) embedder
is selected, the stub LLM backend is used, and the hybrid retrieval strategy
is enabled before any app module is imported.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["QDRANT_URL"] = ":memory:"
os.environ["EMBEDDER_BACKEND"] = "deterministic"
os.environ["LLM_BACKEND"] = "stub"
os.environ["RAG_MIN_SCORE"] = "0.01"
os.environ["RAG_RETRIEVAL_STRATEGY"] = "auto"
os.environ["ROUTER_CLASSIFIER"] = "rules"
os.environ["GRAPH_BACKEND"] = "memory"
os.environ["DATABASE_BACKEND"] = "memory"
# Every test client shares one IP address; rate limiting is exercised by a
# dedicated unit test with a patched limit instead.
os.environ["RATE_LIMIT_ENABLED"] = "false"

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Return a fresh settings instance for configuration tests."""
    return Settings()


@pytest.fixture()
def client() -> TestClient:
    """Return a TestClient against a freshly created app."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _write_text_pdf(path: Path, page_texts: list[str]) -> Path:
    """Create a text-readable PDF with one page per string."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def text_pdf(tmp_path: Path) -> Path:
    """A small text-readable PDF fixture."""
    return _write_text_pdf(
        tmp_path / "syllabus_sample.pdf",
        [
            "Database Management Systems - Syllabus\n"
            "Unit 1: Introduction to databases, data models, normalization.",
            "Unit 2: Relational algebra, SQL, transactions and concurrency.",
        ],
    )


@pytest.fixture()
def make_text_pdf():
    """Factory fixture producing text-readable PDFs."""

    def _make(path: str | Path, page_texts: list[str]) -> Path:
        return _write_text_pdf(Path(path), page_texts)

    return _make
