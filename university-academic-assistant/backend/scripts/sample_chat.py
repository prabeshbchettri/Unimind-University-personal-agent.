"""Sample Phase 5 demo: build sample documents, index them and answer queries.

Produces the "sample response" artefact for the Phase 5 acceptance criteria by
running the full vertical slice (retriever -> Qdrant -> context builder ->
LLM -> answer). The demo is offline-safe: it uses the deterministic embedder
and a grounding-aware stub LLM, so no external services are required.

Run from the ``backend`` directory:

    python scripts/sample_chat.py

To use real models, set EMBEDDER_BACKEND=ollama and LLM_BACKEND=ollama (with
Ollama running and ``ollama pull bge-m3`` / the configured chat model pulled).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.embedding import build_embedder
from app.chunking import SemanticChunker
from app.ingestion.models import NormalizedDocument, PageContent
from app.llm import StubLLMClient
from app.rag.context import NO_CONTEXT_NOTE, ContextBuilder
from app.repositories.qdrant import QdrantRepository
from app.retrieval.normal_retriever import NormalRetriever
from app.services.chat import ChatService
from app.services.vector_indexing import VectorIndexingService
from app.structure import StructurePipeline

SAMPLE_SYLLABUS = [
    "TRIBHUVAN UNIVERSITY\nB.Sc. CSIT\nSemester V\nDatabase Management System Syllabus\n"
    "Unit 1: Introduction to Databases\nDatabases are collections of related data. "
    "A database management system (DBMS) is software that stores, retrieves and "
    "manages data efficiently.\n"
    "Unit 2: Normalization\nNormalization reduces data redundancy and improves data "
    "integrity by decomposing tables into well-structured relations.\n"
    "Unit 3: Transaction Processing\nA transaction is a logical unit of work. "
    "ACID properties guarantee reliable processing of database transactions.",
]

SAMPLE_PAST_QUESTION = [
    "2080\nDatabase Management System\nTime: 3 Hrs.\nFull Marks: 60\n"
    "Q1. Define a database management system. 5 Marks\nDBMS is a software system "
    "that enables users to define, create, maintain and control access to the database.\n"
    "Q5. Explain normalization with an example. 10 Marks\nNormalization is the process "
    "of organizing data to minimize redundancy. The normal forms 1NF, 2NF and 3NF "
    "guide the decomposition of relations.",
]

QUERIES = [
    "Explain normalization.",
    "What is DBMS?",
    "Explain this syllabus topic.",
    "What is the capital of France?",
]


def _normalized(filename: str, pages: list[str], document_id: str) -> NormalizedDocument:
    return NormalizedDocument(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        extraction_method="pymupdf",
        pages=[PageContent(page_number=i + 1, text=text) for i, text in enumerate(pages)],
    )


_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "do", "does", "for", "from",
    "in", "is", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "with", "it", "its",
}


def _meaningful_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if w not in _STOPWORDS}


def _grounded_responder(prompt: str) -> str:
    """Offline stand-in for Llama 3.1 8B that follows the grounding rules.

    When no context was retrieved, or the top source shares no meaningful
    vocabulary with the query, it states that the sources are insufficient
    instead of fabricating an answer.
    """
    if NO_CONTEXT_NOTE in prompt:
        return (
            "I could not find sufficient information in the available university "
            "documents to answer this question. Please consult the department office "
            "or official notices."
        )
    query = prompt.split("User Query: ", 1)[1].split("\n", 1)[0]
    context = prompt.split("CONTEXT:\n", 1)[1].split("\n\n--- END OF CONTEXT ---", 1)[0]
    # A source block is "header line" + the chunk text; chunk text itself may
    # contain blank lines, so bound the first block by the next "[Source" marker.
    first_block = context.split("\n\n[Source 2]", 1)[0]
    header, _, body = first_block.partition("\n")

    if not (_meaningful_tokens(query) & _meaningful_tokens(body)):
        return (
            "I could not find sufficient information in the available university "
            "documents to answer this question. The retrieved documents do not "
            "cover this topic."
        )
    snippet = " ".join(body.split())[:280]
    return f"Based on the retrieved context ({header}), {snippet}."


async def main() -> None:
    pipeline = StructurePipeline()
    structured_docs = [
        pipeline.process(_normalized("dbms_syllabus.pdf", SAMPLE_SYLLABUS, "DOC-SYLLABUS-001")),
        pipeline.process(_normalized("dbms_2080.pdf", SAMPLE_PAST_QUESTION, "DOC-PQ-2080-001")),
    ]

    repository = QdrantRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_names=settings.collection_names,
        default_dimensions=settings.embedding_dimensions,
    )
    embedder = build_embedder(settings)
    service = VectorIndexingService(
        chunker=SemanticChunker(max_chars=350, overlap_chars=80),
        embedder=embedder,
        repository=repository,
        collection_names=settings.collection_names,
    )

    print("=== INDEX SUMMARY ===")
    for structured in structured_docs:
        result = await service.index(structured)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    retriever = NormalRetriever(
        embedder=embedder,
        repository=repository,
        collection_names=settings.collection_names,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    chat = ChatService(
        retriever=retriever,
        context_builder=ContextBuilder(
            max_chars=settings.context_max_chars,
            max_sources=settings.context_max_sources,
        ),
        llm=StubLLMClient(responder=_grounded_responder),
        top_k=settings.rag_top_k,
    )

    print("\n=== SAMPLE RAG RESPONSES (POST /chat equivalent) ===")
    for query in QUERIES:
        result = await chat.answer(query)
        print(f"\nMessage: {query!r}")
        print(json.dumps(
            {
                "answer": result.answer,
                "sources": [
                    {
                        "score": round(source.score, 4),
                        "collection": source.collection,
                        "text": source.text[:200],
                        "metadata": {
                            "document_id": source.metadata.get("document_id"),
                            "document_type": source.metadata.get("document_type"),
                            "page": source.metadata.get("page"),
                            "topic": source.metadata.get("topic"),
                            "subject": source.metadata.get("subject"),
                        },
                    }
                    for source in result.sources
                ],
            },
            indent=2,
            ensure_ascii=False,
        ))

    await repository.close()


if __name__ == "__main__":
    asyncio.run(main())