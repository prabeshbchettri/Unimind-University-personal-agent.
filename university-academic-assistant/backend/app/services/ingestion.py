"""Document ingestion service.

Phase 2: delegates to the ingestion pipeline. Handles PDF intake via PyMuPDF
for text-readable documents and OCR (Tesseract) for scanned documents, and
returns a normalized document structure.

Phase 3: also runs the structure pipeline to classify documents and extract
academic metadata/structure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.logging import get_logger
from app.ingestion import IngestionPipeline, NormalizedDocument
from app.services.base import Service
from app.structure import StructurePipeline, StructuredDocument

logger = get_logger("app.services.ingestion")


class DocumentIngestionService(Service):
    """Handles document intake (PDF/OCR) and normalization/structuring."""

    def __init__(
        self,
        pipeline: IngestionPipeline | None = None,
        structure_pipeline: StructurePipeline | None = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline or IngestionPipeline()
        self.structure_pipeline = structure_pipeline or StructurePipeline()

    async def health(self) -> dict:
        return {"name": self.name, "status": "ok"}

    async def ingest(self, document_path: str) -> str:
        """Ingest a document and return its ``document_id``."""
        document = await self.ingest_document(document_path)
        return document.document_id

    async def ingest_document(self, document_path: str) -> NormalizedDocument:
        """Ingest a PDF file and return the normalized document.

        The synchronous pipeline runs in a worker thread so the event loop is
        not blocked during extraction/OCR.
        """
        logger.info(
            "Ingesting document",
            extra={"extra_fields": {"path": str(document_path)}},
        )
        path = Path(document_path)
        return await asyncio.to_thread(self.pipeline.ingest_file, path)

    async def analyze_document(self, document_path: str) -> StructuredDocument:
        """Ingest a PDF and return the schema-validated structured document."""
        document = await self.ingest_document(document_path)
        return await self.structure_document(document)

    async def structure_document(self, document: NormalizedDocument) -> StructuredDocument:
        """Classify and structure an already-ingested normalized document."""
        return await asyncio.to_thread(self.structure_pipeline.process, document)