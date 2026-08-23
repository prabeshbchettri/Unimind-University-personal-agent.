"""Document type classification.

Deterministic scoring based on content features (keyword/regex patterns), so
classification never relies on the filename. Each document type has weighted
features; the highest scoring type above a threshold wins, otherwise
``unknown``.
"""

from __future__ import annotations

import re

from app.config import settings
from app.ingestion.models import NormalizedDocument
from app.structure.models import DocumentType

# Priority order used to break score ties (more specific types first).
_TYPE_PRIORITY: tuple[DocumentType, ...] = (
    "past_question",
    "syllabus",
    "library_book",
    "rules_regulations",
    "academic_calendar",
    "notice",
)

_FEATURES: dict[DocumentType, list[tuple[str, int]]] = {
    "past_question": [
        (r"\bQ\.?\s?\d+\b", 3),
        (r"\bMarks?\b", 3),
        (r"\bTime\s*:\s*\d", 2),
        (r"\bMax(?:imum)?\s*Marks\b", 2),
        (r"\bFull\s+Marks\b", 2),
        (r"\bAttempt\b", 2),
        (r"\bAnswer\s+(?:any|all)\b", 2),
        (r"\bCandidates?\s+are\s+required\b", 2),
        (r"\bPass\s+Marks?\b", 1),
        (r"\bExaminer\b", 1),
    ],
    "syllabus": [
        (r"\bSyllabus\b", 3),
        (r"\bUnit\s+\d+\b", 2),
        (r"\bModule\s+\d+\b", 2),
        (r"\bCourse\s+Objectives?\b", 2),
        (r"\bCourse\s+Description\b", 1),
        (r"\bCredit(?:s)?\b", 2),
        (r"\bTeaching\s+Hours\b", 2),
        (r"\bRecommended\s+(?:Books?|Texts?|Readings?)\b", 2),
        (r"\bCourse\s+Outcomes?\b", 1),
        (r"\bPrerequisites?\b", 1),
    ],
    "library_book": [
        (r"\bChapter\s+\d+\b", 3),
        (r"\bTable\s+of\s+Contents\b", 3),
        (r"\bContents\b", 1),
        (r"\bPreface\b", 3),
        (r"\bISBN\b", 3),
        (r"\bPublisher\b", 2),
        (r"\bEdition\b", 2),
        (r"\bCopyright\b", 2),
        (r"\bIndex\b", 1),
        (r"\bAcknowledg(e)?ment(s)?\b", 1),
    ],
    "rules_regulations": [
        (r"\bRegulations?\b", 3),
        (r"\bOrdinance\b", 3),
        (r"\bEligibility\b", 2),
        (r"\bAttendance\b", 2),
        (r"\bGrievance\b", 2),
        (r"\bDisciplinary\b", 2),
        (r"\bCode\s+of\s+Conduct\b", 2),
        (r"\bRules?\b", 1),
        (r"\bWithdrawal\b", 1),
        (r"\bRe-admission\b", 1),
    ],
    "academic_calendar": [
        (r"\bAcademic\s+Calendar\b", 4),
        (r"\bSemester\s+Calendar\b", 3),
        (r"\bCommence(?:ment|s)\b", 2),
        (r"\b(?:Mid[- ]Semester|End[- ]Semester)\s+Exam\w*\b", 2),
        (r"\b(?:Winter|Summer|Spring)\s+Break\b", 2),
        (r"\bDeadlines?\b", 2),
        (r"\bAdmission\s+(?:opens|closes|begins)\b", 2),
        (r"\bRegistration\s+(?:opens|closes|begins)\b", 2),
        (r"\bHolidays?\b", 1),
        (r"\bExamination\s+Schedule\b", 2),
    ],
    "notice": [
        (r"\bNotice\b", 3),
        (r"\bThis\s+is\s+to\s+inform\b", 3),
        (r"\bTo\s+all\s+(?:students|staff|faculty)\b", 2),
        (r"\bKindly\b", 1),
        (r"\bConcerned\b", 1),
        (r"\bIssu(?:ed|ance)\b", 1),
        (r"\bDate\s*:", 1),
        (r"\bRe(?:garding)?\b", 1),
    ],
}

_SKIP_LINE_RE = re.compile(
    r"(?i)(^\s*\d+\s*$|page\s*\d+|\.\.\.+|^[\s.:/()\-–—]+$)"
)


def document_text(document: NormalizedDocument) -> str:
    """Concatenate all page text into a single string."""
    return "\n".join(page.text for page in document.pages)


class DocumentClassifier:
    """Classify a normalized document into a :class:`DocumentType`."""

    def __init__(self, min_score: int | None = None) -> None:
        self.min_score = (
            min_score if min_score is not None else settings.document_classifier_min_score
        )

    def classify(self, document: NormalizedDocument) -> DocumentType:
        text = document_text(document).strip()
        if not text:
            return "unknown"

        scores = {
            document_type: self._score(text, features)
            for document_type, features in _FEATURES.items()
        }
        best_score = max(scores.values())
        if best_score < self.min_score:
            return "unknown"

        # On ties, the more specific type (earlier in priority) wins.
        for document_type in _TYPE_PRIORITY:
            if scores[document_type] == best_score:
                return document_type
        return "unknown"

    @staticmethod
    def _score(text: str, features: list[tuple[str, int]]) -> int:
        total = 0
        for pattern, weight in features:
            if re.search(pattern, text, re.IGNORECASE):
                total += weight
        return total

    def scores(self, document: NormalizedDocument) -> dict[DocumentType, int]:
        """Return the raw per-type scores (useful for debugging)."""
        text = document_text(document).strip()
        return {
            document_type: self._score(text, features)
            for document_type, features in _FEATURES.items()
        }
