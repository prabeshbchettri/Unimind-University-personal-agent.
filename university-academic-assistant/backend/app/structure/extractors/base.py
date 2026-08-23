"""Metadata extractor interface.

The structure pipeline depends only on this interface, so the deterministic
extractor can be replaced by an LLM-based extractor (or a hybrid) later
without changing the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

from app.ingestion.models import NormalizedDocument
from app.structure.models import DocumentMetadata, DocumentStructure, DocumentType


class ExtractionResult(NamedTuple):
    """Output of a metadata extractor."""

    metadata: DocumentMetadata
    structure: DocumentStructure


class MetadataExtractor(ABC):
    """Contract for extracting metadata/structure from a normalized document."""

    name: str = "base"

    @abstractmethod
    def extract(
        self, document: NormalizedDocument, document_type: DocumentType
    ) -> ExtractionResult:
        """Return validated metadata and structure for the document."""
        raise NotImplementedError
