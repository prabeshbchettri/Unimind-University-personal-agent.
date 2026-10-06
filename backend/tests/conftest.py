"""Shared test fixtures.

External services (Qdrant) and embedding models are replaced with small
deterministic fakes here; one separate integration module tests against a real
Qdrant server when it is reachable.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from app.llm import LLMResponse
from app.retriever import RetrievedChunk
from app.vectorstore import PAYLOAD_KEYS


class FakeEmbedding:
    """Deterministic hash-based embedding client.

    Produces stable unit vectors from the text content, so identical texts give
    identical vectors and similar-but-not-identical texts give nearby vectors
    (shared words contribute to the same hash buckets). No network, no model.
    """

    name = "fake"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in text.lower().split():
            digest = hashlib.md5(word.encode("utf-8")).digest()
            index = struct.unpack("H", digest[:2])[0] % self.dim
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class FakeQdrant:
    """Minimal in-memory stand-in for QdrantVectorStore's client.

    Implements just the operations the vector store wrapper uses:
    collection_exists, get_collection, create_collection, upsert, query_points.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.collection: str | None = None
        self.points: dict[str, dict] = {}

    # -- collection management ------------------------------------------------
    def collection_exists(self, collection_name: str) -> bool:
        return self.collection == collection_name

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        if self.collection != collection_name:
            raise RuntimeError("collection does not exist")
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=self.dim)))
        )

    def create_collection(self, collection_name: str, vectors_config: SimpleNamespace) -> None:
        self.collection = collection_name
        self.dim = vectors_config.size

    def delete_collection(self, collection_name: str) -> None:
        if self.collection == collection_name:
            self.collection = None
            self.points.clear()

    # -- data operations -------------------------------------------------------
    def upsert(self, collection_name: str, points: list, wait: bool = True) -> None:
        assert collection_name == self.collection, "upsert before ensure_collection"
        for point in points:
            self.points[point.id] = {"vector": point.vector, "payload": point.payload}

    def query_points(self, collection_name: str, query: list[float], limit: int, with_payload: bool = True):
        assert collection_name == self.collection
        scores: list[SimpleNamespace] = []
        for point in self.points.values():
            vector = point["vector"]
            score = sum(a * b for a, b in zip(query, vector))
            scores.append(SimpleNamespace(score=score, payload=point["payload"]))
        scores.sort(key=lambda item: item.score, reverse=True)
        return SimpleNamespace(points=scores[:limit])

    def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        assert collection_name == self.collection
        points = [
            SimpleNamespace(id=point_id, payload=data["payload"])
            for point_id, data in self.points.items()
        ]
        return points[:limit], None


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding()


class FakeLLM:
    """Records prompts and returns a canned completion. No network."""

    name = "fake"

    def __init__(self, text: str = "The policy says 75 percent attendance [1].") -> None:
        self.text = text
        self.calls: list[dict] = []

    def generate(self, *, system_prompt: str, user_query: str, context: str) -> LLMResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_query": user_query,
                "context": context,
            }
        )
        return LLMResponse(text=self.text, provider=self.name, model="fake-model")


class FakeRetriever:
    """Returns a fixed ranked list regardless of the query."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self._chunks


def make_retrieved(
    text: str,
    document: str = "doc.pdf",
    page: int = 1,
    score: float = 0.8,
    index: int = 0,
    chunk_id: str | None = None,
    bm25_score: float | None = None,
) -> RetrievedChunk:
    """Build one retrieved chunk with full metadata.

    ``semantic_score`` defaults to ``score`` so normal-style chunks are gated
    by the semantic threshold, matching the real VectorRetriever behavior.
    """
    return RetrievedChunk(
        chunk_id=chunk_id or f"{document}::p{page}::c{index}",
        text=text,
        document=document,
        page=page,
        score=score,
        chunk_index=index,
        semantic_score=score,
        bm25_score=bm25_score,
    )


@pytest.fixture
def fake_store() -> SimpleNamespace:
    """A QdrantVectorStore wired to the in-memory fake client."""
    from app.vectorstore import QdrantVectorStore

    client = FakeQdrant(dim=64)
    store = QdrantVectorStore(url="http://fake", collection="test_docs", dim=64, client=client)
    store.ensure_collection()
    return SimpleNamespace(store=store, client=client)


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> Path:
    """Create a small real PDF (via PyMuPDF) for extraction tests."""
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    y = 60
    page.insert_text((60, y), title, fontname="helv", fontsize=14)
    y += 30
    for paragraph in paragraphs:
        page.insert_text((60, y), paragraph, fontname="helv", fontsize=11)
        y += 24
    document.save(path)
    document.close()
    return path


def make_payload_chunk(text: str, page: int = 1, index: int = 0) -> dict:
    """Build a payload dict with every required metadata key."""
    from app.ingestion import Chunk

    chunk = Chunk(
        id=f"doc::p{page}::c{index}",
        document_name="doc.pdf",
        source_path="/data/doc.pdf",
        page=page,
        chunk_index=index,
        text=text,
    )
    return {
        "chunk_id": chunk.id,
        "document_name": chunk.document_name,
        "source_path": chunk.source_path,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
    }


def assert_payload_keys(payload: dict) -> None:
    missing = [key for key in PAYLOAD_KEYS if key not in payload]
    assert not missing, f"payload is missing metadata keys: {missing}"
