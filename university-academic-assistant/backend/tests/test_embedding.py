"""Tests for embedding backends."""

import math

import pytest

from app.embedding import DeterministicEmbedder, EmbeddingUnavailableError, OllamaEmbedder


def test_deterministic_dimensions() -> None:
    assert DeterministicEmbedder(16).dimensions == 16
    assert DeterministicEmbedder().dimensions == 384


def test_deterministic_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        DeterministicEmbedder(0)


def test_deterministic_embeds_and_normalizes() -> None:
    embedder = DeterministicEmbedder(16)
    vector = embedder.embed_one("hello world")
    assert len(vector) == 16
    assert all(isinstance(value, float) for value in vector)
    norm = math.sqrt(sum(value * value for value in vector))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_deterministic_is_stable() -> None:
    embedder = DeterministicEmbedder(16)
    assert embedder.embed_one("normalization removes redundancy") == embedder.embed_one("normalization removes redundancy")


def test_deterministic_distinguishes_text() -> None:
    embedder = DeterministicEmbedder(16)
    assert embedder.embed_one("normalization removes redundancy") != embedder.embed_one("transaction ACID properties")


def test_deterministic_batch() -> None:
    embedder = DeterministicEmbedder(16)
    vectors = embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 16 for v in vectors)


def test_ollama_embedder_unreachable_raises() -> None:
    embedder = OllamaEmbedder(base_url="http://127.0.0.1:9", model="bge-m3")
    with pytest.raises(EmbeddingUnavailableError):
        embedder.embed(["hello"])


def test_ollama_embedder_empty_input() -> None:
    embedder = OllamaEmbedder(base_url="http://127.0.0.1:9", model="bge-m3")
    assert embedder.embed([]) == []