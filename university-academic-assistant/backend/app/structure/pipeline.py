"""Structure extraction pipeline (Phase 3).

Flow: normalized document -> classification -> metadata/structure extraction
-> schema-validated :class:`StructuredDocument`.
"""

from __future__ import annotations

from app.config import settings
from app.core.logging import get_logger
from app.ingestion.models import NormalizedDocument
from app.structure.classifier import DocumentClassifier
from app.structure.extractors import (
    DeterministicMetadataExtractor,
    LLMMetadataExtractor,
    MetadataExtractor,
)
from app.structure.models import StructuredDocument

logger = get_logger("app.structure.pipeline")


class StructurePipeline:
    """Classify a normalized document and produce a structured document."""

    def __init__(
        self,
        classifier: DocumentClassifier | None = None,
        extractor: MetadataExtractor | None = None,
    ) -> None:
        self.classifier = classifier or DocumentClassifier()
        self.extractor = extractor or self._build_extractor()

    def _build_extractor(self) -> MetadataExtractor:
        mode = settings.structure_extractor
        if mode == "deterministic":
            return DeterministicMetadataExtractor()
        if mode == "llm":
            # The LLM service becomes available from Phase 5; until then a
            # clear error is raised at classification time.
            def generate(prompt: str) -> str:
                raise NotImplementedError(
                    "LLM structured extraction requires an LLM service (Phase 5)."
                )

            return LLMMetadataExtractor(generate=generate)
        raise ValueError(
            f"structure_extractor must be 'deterministic' or 'llm', got {mode!r}"
        )

    def process(self, document: NormalizedDocument) -> StructuredDocument:
        """Classify, extract and validate a normalized document."""
        document_type = self.classifier.classify(document)
        result = self.extractor.extract(document, document_type)

        structured = StructuredDocument(
            document_id=document.document_id,
            filename=document.filename,
            extraction_method=document.extraction_method,
            page_count=document.page_count,
            pages=document.pages,
            metadata=result.metadata,
            structure=result.structure,
        )

        logger.info(
            "Structured document produced",
            extra={
                "extra_fields": {
                    "document_id": structured.document_id,
                    "document_type": structured.metadata.document_type,
                }
            },
        )
        return structured
