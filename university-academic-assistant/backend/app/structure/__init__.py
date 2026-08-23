"""Document structure extraction package (Phase 3).

Public API:
- :class:`StructurePipeline` — classify + extract + validate.
- :class:`StructuredDocument` / :class:`DocumentMetadata` — schema-validated
  outputs consumed by later phases.
"""

from .classifier import DocumentClassifier
from .extractors import (
    DeterministicMetadataExtractor,
    ExtractionResult,
    LLMMetadataExtractor,
    MetadataExtractor,
)
from .heuristics import (
    find_author,
    find_marks,
    find_program,
    find_semester,
    find_subject_code,
    find_university,
    find_year,
)
from .models import (
    Chapter,
    DocumentMetadata,
    DocumentStructure,
    DocumentType,
    Question,
    StructuredDocument,
)
from .pipeline import StructurePipeline
from .validation import LLMExtractionError, parse_llm_json

__all__ = [
    "StructurePipeline",
    "DocumentClassifier",
    "StructuredDocument",
    "DocumentMetadata",
    "DocumentStructure",
    "DocumentType",
    "Question",
    "Chapter",
    "MetadataExtractor",
    "DeterministicMetadataExtractor",
    "LLMMetadataExtractor",
    "ExtractionResult",
    "LLMExtractionError",
    "parse_llm_json",
    "find_semester",
    "find_year",
    "find_subject_code",
    "find_marks",
    "find_university",
    "find_program",
    "find_author",
]
