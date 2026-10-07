"""Document-type intent ranking: a small, deterministic post-retrieval step.

The vector + BM25 + RRF pipeline ranks chunks by *content* similarity. That is
usually the right signal, but a few query words describe the *kind of document*
the user wants rather than the subject matter -- most importantly ``syllabus``.
A course syllabus is mostly administrative text (course objectives, assessment
scheme, module list) and contains far less of the course vocabulary that lecture
slides repeat, so a query like ``syllabus of the software engineering`` can rank
slide chunks above the actual syllabus chunk.

This module adds one small, explainable adjustment **after** retrieval, leaving
vector search, BM25 and RRF untouched:

- detect when the query asks for a syllabus / course-outline document;
- classify each retrieved chunk's document from its filename and folder;
- re-rank so a syllabus document (of the course named in the query, when one is
  named) moves to the top.

When the query shows no document-type intent the ranking is returned unchanged,
so ordinary content queries keep the exact vector/BM25/RRF order.

No extra model call is made: this is a keyword detector plus a metadata lookup,
which keeps it deterministic and easy to demonstrate.
"""

from __future__ import annotations

import re
from dataclasses import replace

from app.retriever import RetrievedChunk

# --------------------------------------------------------------------------- #
# Document types
# --------------------------------------------------------------------------- #
SYLLABUS = "syllabus"
SLIDE = "slide"
CHAPTER = "chapter"
PAST_PAPER = "past_paper"
NOTICE = "notice"
RULES = "rules"
OTHER = "other"

#: Query phrases that indicate the user wants a syllabus / course outline.
_SYLLABUS_INTENT_RE = re.compile(
    r"\b(syllabus|syllabi|course\s+outlines?|course\s+contents?|curriculum|"
    r"course\s+structure|course\s+descriptions?|outline\s+of\s+the\s+course)\b",
    re.IGNORECASE,
)

#: Official course documents are named ``<COURSECODE><COURSENAME>_<stamp>.pdf``
#: (for example ``ENCT352SOFTWAREENGINEERING_2026_04_25_10_01_01.pdf``). Matching
#: a leading course code generalizes across every course in the corpus.
_COURSE_CODE_RE = re.compile(r"^[a-z]{2,5}\s*\d{3}")

#: Folder names that are storage buckets, not courses.
_NON_COURSE_FOLDERS = frozenset(
    {"notices", "notice", "rules", "documents", "books", "past_paper", "calender", "calendar", "syllabus"}
)

#: Filler words dropped when deriving a course name from a folder.
_FILLER = frozenset({"and", "the", "of", "for"})

#: Words that describe the document kind rather than a topic; removed before
#: deciding whether the query names a specific course.
_INTENT_WORDS = frozenset(
    {
        "syllabus", "syllabi", "course", "courses", "outline", "outlines",
        "content", "contents", "curriculum", "structure", "description",
    }
)

#: Conversational filler removed before deciding whether a course is named.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "for", "me", "my", "our", "give", "show",
        "tell", "please", "is", "are", "was", "were", "what", "whats", "i",
        "you", "we", "can", "could", "would", "and", "or", "to", "in", "on",
        "about", "get", "need", "want", "provide", "list", "there",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# --------------------------------------------------------------------------- #
# Ranking constants (documented so the boost is explainable, not magic)
# --------------------------------------------------------------------------- #
#: Base ranking weight of position ``rank`` is ``_RANK_BASE / (rank + 1)``, so
#: position 0 scores 1.0 and position 2 scores ~0.33. The boosts below are on
#: the same scale, which makes their effect easy to reason about.
_RANK_BASE = 1.0

#: Syllabus of the course the query names -> strong boost (dominates the field).
SYLLABUS_MATCH_BOOST = 2.0
#: Any syllabus when the query asks for one but names no course.
SYLLABUS_GENERIC_BOOST = 1.0
#: Other official course documents (past papers / chapters) of the named course.
COURSE_DOC_BOOST = 0.3


# --------------------------------------------------------------------------- #
# Detection / classification helpers
# --------------------------------------------------------------------------- #
def detect_syllabus_intent(query: str) -> bool:
    """True when the query asks for a syllabus / course outline."""
    return bool(_SYLLABUS_INTENT_RE.search(query or ""))


def _folder(source_path: str) -> str:
    """Return the containing folder name of a path (``""`` when unknown)."""
    if not source_path:
        return ""
    parts = [part for part in re.split(r"[\\/]+", source_path.strip()) if part]
    return parts[-2].lower() if len(parts) >= 2 else ""


def classify_document(document_name: str, source_path: str = "") -> str:
    """Classify a document by its filename and folder.

    Deliberately keyword/metadata based (no model): the filename and folder are
    the metadata the pipeline already stores, so this needs no re-ingestion.
    """
    name = (document_name or "").lower()
    folder = _folder(source_path)
    if "slide" in name:
        return SLIDE
    if "chapter" in name:
        return CHAPTER
    if "past_question" in name or "past paper" in name or "past_paper" in name:
        return PAST_PAPER
    if "notice" in name or folder in ("notices", "notice"):
        return NOTICE
    if folder == "rules" or "university_rules" in name:
        return RULES
    if _COURSE_CODE_RE.match(name):
        return SYLLABUS
    return OTHER


def _course_words(document_name: str, source_path: str) -> set[str]:
    """Words that identify the course a document belongs to.

    The folder is the most reliable signal (``software_engineering`` ->
    ``{software, engineering}``); when it is missing we fall back to the
    filename, dropping the course code, digits and generic document words.
    """
    folder = _folder(source_path)
    if folder and folder not in _NON_COURSE_FOLDERS:
        words = {
            word
            for word in re.split(r"[^a-z0-9]+", folder)
            if word and word not in _FILLER
        }
        if words:
            return words

    stem = re.sub(r"\.[a-z0-9]+$", "", (document_name or "").lower())
    stem = re.sub(r"\d+", " ", stem)
    stop = {"pdf", "slide", "slides", "chapter", "past", "question", "notice", "rules"}
    return {
        word
        for word in re.split(r"[^a-z]+", stem)
        if len(word) > 2 and word not in stop and word not in _FILLER
    }


def _course_named_in_query(document_name: str, source_path: str, query_tokens: set[str]) -> bool:
    """True when the query contains every course word of this document."""
    words = _course_words(document_name, source_path)
    return bool(words) and words <= query_tokens


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def apply_intent_ranking(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-rank ``chunks`` for document-type intent; a no-op for content queries.

    Only a syllabus-intent query triggers any change. The original best-first
    order is preserved as the tie-breaker, and chunks keep their strategy scores
    (``semantic_score``/``bm25_score``); each chunk records the boost that was
    applied in ``intent_boost`` for observability.
    """
    if not chunks or not detect_syllabus_intent(query):
        return chunks

    query_tokens = set(_TOKEN_RE.findall((query or "").lower()))
    # Topic words = the query minus the words that describe the document kind
    # and conversational filler. Empty means a generic "give me the syllabus".
    topic_tokens = query_tokens - _INTENT_WORDS - _STOPWORDS

    # Does the query name a course that is actually among the candidates?
    named_course = bool(topic_tokens) and any(
        classify_document(chunk.document, chunk.source_path) == SYLLABUS
        and _course_named_in_query(chunk.document, chunk.source_path, query_tokens)
        for chunk in chunks
    )
    # A query that names a course we cannot match gets no boost: better to keep
    # the content ranking than to surface an unrelated course's syllabus.
    if topic_tokens and not named_course:
        return chunks

    ranked: list[tuple[float, int, RetrievedChunk, float]] = []
    for rank, chunk in enumerate(chunks):
        document_type = classify_document(chunk.document, chunk.source_path)
        boost = 0.0
        if document_type == SYLLABUS:
            if not topic_tokens:
                boost = SYLLABUS_GENERIC_BOOST
            elif _course_named_in_query(chunk.document, chunk.source_path, query_tokens):
                boost = SYLLABUS_MATCH_BOOST
        elif named_course and document_type in (PAST_PAPER, CHAPTER) and _course_named_in_query(
            chunk.document, chunk.source_path, query_tokens
        ):
            boost = COURSE_DOC_BOOST

        final = _RANK_BASE / (rank + 1) + boost
        ranked.append((final, rank, chunk, boost))

    if all(boost == 0.0 for _, _, _, boost in ranked):
        return chunks

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [replace(chunk, intent_boost=boost) for _, _, chunk, boost in ranked]
