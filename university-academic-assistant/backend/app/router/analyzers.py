"""Query analyzers (Phase 8 + Phase 11).

The rule-based analyzer is the default: deterministic, explainable and
dependency-free. It classifies queries as GRAPH (structural relationships),
HYBRID (exact-match tokens that benefit from BM25), NORMAL (semantic/
conceptual) or WEB (current/external information, Phase 11). The optional LLM
analyzer is only consulted when the rules cannot decide — the router never
depends entirely on an LLM.
"""

from __future__ import annotations

import json
import re

from app.llm import LLMClient
from app.router.plan import (
    GRAPH_QUESTIONS_ABOUT_TOPIC,
    GRAPH_SUBJECTS_CONTAINING_TOPIC,
    GRAPH_SUBJECTS_IN_SEMESTER,
    GRAPH_SUBTOPICS_OF_TOPIC,
    GRAPH_TOPICS_OF_SUBJECT,
    QueryProfile,
    RetrievalStrategy,
)

# ---------------------------------------------------------------------------
# Exact-match features (HYBRID)
# ---------------------------------------------------------------------------

_REGULATION = re.compile(r"\b(?:regulation|regulations?)\s+(\d+)\b|\brule[sd]?\s+(\d+)\b", re.IGNORECASE)
_SUBJECT_CODE = re.compile(r"\b[A-Z]{2,6}\s?\d{3}\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_QUESTION_NUMBER = re.compile(r"\b(?:q\d+)\b|\bquestion\s+(\d+)\b", re.IGNORECASE)
_MARKS = re.compile(r"\b(\d+)\s*marks?\b", re.IGNORECASE)
_PERCENT = re.compile(r"\b(\d+)\s*percent\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Graph intents (GRAPH) — checked before the exact-match rules
# ---------------------------------------------------------------------------

_SEMESTER_SUBJECTS = re.compile(r"\b(subjects|courses|modules)\b.*\bsemester\s+(\d+)\b", re.IGNORECASE | re.DOTALL)
_TOPICS_OF = re.compile(r"\btopics?\s+of\s+([a-z0-9&\- .]+?)[?.]?\s*$", re.IGNORECASE)
_TOPICS_INCLUDED_IN = re.compile(r"\btopics?\b.*\bincluded\s+in\s+([a-z0-9&\- .]+?)(?=\s+(?:in|for|during|at|by|semester)|\s*\?|\s*$)", re.IGNORECASE | re.DOTALL)
_SUBTOPICS_OF = re.compile(r"\bsubtopics?\s+of\s+([a-z0-9&\- .]+?)[?.]?\s*$", re.IGNORECASE)
_PAST_QUESTIONS_ABOUT = re.compile(r"\b(?:past\s+questions?|questions?)\b.*\b(?:about|related\s+to|on)\s+([a-z0-9&\- .]+?)[?.]?\s*$", re.IGNORECASE | re.DOTALL)
_SUBJECTS_CONTAINING = re.compile(r"\bsubjects?\b.*\b(?:contain(?:ing)?|cover(?:ing)?|include(?:s)?)\s+(?:the\s+topic\s+|a\s+topic\s+)?([a-z0-9&\- .]+?)[?.]?\s*$", re.IGNORECASE | re.DOTALL)


def _clean_entity(value: str) -> str | None:
    """Normalize a captured entity name; None when it is not a real name."""
    value = value.strip().strip("?.").strip()
    if not value:
        return None
    if value.isdigit() or re.fullmatch(r"[A-Za-z]\d{1,3}", value):  # e.g. "5", "Q5"
        return None
    return value


# ---------------------------------------------------------------------------
# Web intents (Phase 11) — checked last, after every internal strategy
# ---------------------------------------------------------------------------

# Explicit current/external information signals. University questions that
# also carry one of these are treated as MIXED (web supplements the
# university documents) rather than replaced by web.
_WEB_SIGNAL = re.compile(
    r"\b(?:latest|current|today|news|pricing|price|upcoming|recent(?:ly)?|"
    r"updated?|forecast|weather|what\s+happened)\b",
    re.IGNORECASE,
)

# University-domain vocabulary. When present, university documents stay
# preferred: a query with one of these terms is internal (or mixed), never a
# pure web query — web content must not override authoritative university
# documents for university-specific facts.
_UNIVERSITY_CONTEXT = re.compile(
    r"\b(?:attendance|semester|syllabus|academic\s+calendar|exam(?:ination)?|"
    r"marks?|regulation|rule\b|university|college|campus|admission|"
    r"enroll(?:ment|ed)?|department|curriculum|grading|hostel|laboratory|"
    r"library|tuition|scholarship|bsc|csit|bachelor|master's?|program|course)\b",
    re.IGNORECASE,
)


def has_university_context(query: str) -> bool:
    """True when the query references the university domain.

    Used to keep university documents preferred over web results: a query
    with university context never becomes a pure web query.
    """
    return bool(query and _UNIVERSITY_CONTEXT.search(query))


def has_web_signal(query: str) -> bool:
    """True when the query asks for current or external information."""
    return bool(query and _WEB_SIGNAL.search(query))


class QueryAnalyzer:
    """Interface for query classification."""

    name = "query_analyzer"

    def analyze(self, query: str) -> QueryProfile:
        raise NotImplementedError


class RuleBasedQueryAnalyzer(QueryAnalyzer):
    """Deterministic, explainable rule-based classification."""

    name = "rules"

    def analyze(self, query: str) -> QueryProfile:
        if not query or not query.strip():
            return QueryProfile(confidence=0.0, reason="Empty query; cannot classify.")

        # --- GRAPH: structural relationship questions -----------------------
        profile = self._analyze_graph(query)
        if profile:
            return profile

        # --- HYBRID: exact-match tokens that benefit from BM25 --------------
        profile = self._analyze_exact_match(query)
        if profile:
            return profile

        # --- WEB: current / external information (Phase 11) -----------------
        # Checked before the NORMAL catch-all but after every internal rule,
        # so university documents stay preferred for university questions.
        profile = self._analyze_web(query)
        if profile:
            return profile

        # --- NORMAL: everything else is a semantic/conceptual question ------
        return QueryProfile(
            strategy=RetrievalStrategy.NORMAL,
            confidence=0.7,
            reason="Conceptual/semantic question; dense vector retrieval is appropriate.",
        )

    def _analyze_graph(self, query: str) -> QueryProfile | None:
        match = _SEMESTER_SUBJECTS.search(query)
        if match:
            semester = int(match.group(2))
            return QueryProfile(
                strategy=RetrievalStrategy.GRAPH,
                confidence=1.0,
                reason=f"Question asks which subjects are in semester {semester}.",
                graph_parameters={"intent": GRAPH_SUBJECTS_IN_SEMESTER, "semester": semester},
            )

        match = _TOPICS_OF.search(query) or _TOPICS_INCLUDED_IN.search(query)
        if match:
            subject = _clean_entity(match.group(1))
            if subject:
                return QueryProfile(
                    strategy=RetrievalStrategy.GRAPH,
                    confidence=1.0,
                    reason=f"Question asks for the topics of subject {subject!r}.",
                    graph_parameters={"intent": GRAPH_TOPICS_OF_SUBJECT, "subject": subject},
                )

        match = _SUBTOPICS_OF.search(query)
        if match:
            topic = _clean_entity(match.group(1))
            if topic:
                return QueryProfile(
                    strategy=RetrievalStrategy.GRAPH,
                    confidence=1.0,
                    reason=f"Question asks for the subtopics of topic {topic!r}.",
                    graph_parameters={"intent": GRAPH_SUBTOPICS_OF_TOPIC, "topic": topic},
                )

        match = _PAST_QUESTIONS_ABOUT.search(query)
        if match:
            topic = _clean_entity(match.group(1))
            if topic:
                return QueryProfile(
                    strategy=RetrievalStrategy.GRAPH,
                    confidence=1.0,
                    reason=f"Question asks for past questions about topic {topic!r}.",
                    graph_parameters={"intent": GRAPH_QUESTIONS_ABOUT_TOPIC, "topic": topic},
                )

        match = _SUBJECTS_CONTAINING.search(query)
        if match:
            topic = _clean_entity(match.group(1))
            if topic:
                return QueryProfile(
                    strategy=RetrievalStrategy.GRAPH,
                    confidence=1.0,
                    reason=f"Question asks which subjects contain topic {topic!r}.",
                    graph_parameters={"intent": GRAPH_SUBJECTS_CONTAINING_TOPIC, "topic": topic},
                )

        return None

    def _analyze_exact_match(self, query: str) -> QueryProfile | None:
        if _REGULATION.search(query):
            return QueryProfile(
                strategy=RetrievalStrategy.HYBRID,
                confidence=1.0,
                reason="Query contains an exact regulation/rule number.",
                filters={"document_type": "rules_regulations"},
            )
        if _SUBJECT_CODE.search(query):
            return QueryProfile(
                strategy=RetrievalStrategy.HYBRID,
                confidence=1.0,
                reason="Query contains an exact subject code (e.g. CSIT 325).",
            )
        if _DATE.search(query):
            return QueryProfile(
                strategy=RetrievalStrategy.HYBRID,
                confidence=1.0,
                reason="Query contains an exact date.",
            )
        if _YEAR.search(query):
            return QueryProfile(
                strategy=RetrievalStrategy.HYBRID,
                confidence=1.0,
                reason="Query contains an exact academic year.",
            )
        if _QUESTION_NUMBER.search(query):
            return QueryProfile(
                strategy=RetrievalStrategy.HYBRID,
                confidence=1.0,
                reason="Query references a specific question number.",
            )
        if _MARKS.search(query) or _PERCENT.search(query):
            return QueryProfile(
                strategy=RetrievalStrategy.HYBRID,
                confidence=1.0,
                reason="Query contains an exact mark/percent value.",
            )
        return None

    def _analyze_web(self, query: str) -> QueryProfile | None:
        """Classify queries asking for current or external information.

        University priority: when the query also references the university
        domain (attendance, semester, academic calendar, ...), web retrieval
        *supplements* the university documents (``mixed=True``) instead of
        replacing them.
        """
        if not has_web_signal(query):
            return None
        if has_university_context(query):
            return QueryProfile(
                strategy=RetrievalStrategy.WEB,
                confidence=0.9,
                reason=(
                    "Question mixes university context with current/external "
                    "information; web retrieval supplements the university "
                    "documents."
                ),
                web_parameters={"mixed": True},
            )
        return QueryProfile(
            strategy=RetrievalStrategy.WEB,
            confidence=0.9,
            reason="Question asks for current or external information; web retrieval is appropriate.",
            web_parameters={},
        )


_LLM_SYSTEM_PROMPT = (
    "You classify questions asked to a university academic assistant into exactly one of four "
    "retrieval strategies. Rules:\n"
    "- GRAPH: questions about relationships between academic entities, e.g. subjects in a semester, "
    "topics or subtopics of a subject/topic, past questions about a topic, subjects containing a topic.\n"
    "- HYBRID: questions containing exact tokens that benefit from keyword matching: regulation or rule "
    "numbers, subject codes (e.g. CSIT 325), dates (YYYY-MM-DD), academic years, question numbers, marks or percents.\n"
    "- NORMAL: conceptual/semantic questions about the content itself, e.g. 'Explain normalization.'\n"
    "- WEB: questions asking for current or external information, e.g. 'What is the latest Python "
    "version?', 'What happened in today's AI news?'. Prefer WEB only when the question is not about "
    "the university's own documents.\n"
    'Reply with strict JSON only: {"strategy": "NORMAL|HYBRID|GRAPH|WEB", "reason": "short explanation"}'
)


class LLMQueryAnalyzer(QueryAnalyzer):
    """Optional LLM classification, consulted only when rules are inconclusive."""

    name = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def analyze(self, query: str) -> QueryProfile:
        if not query or not query.strip():
            return QueryProfile(confidence=0.0, reason="Empty query; cannot classify.")
        try:
            raw = self.llm.complete(_LLM_SYSTEM_PROMPT, f"Query: {query}\nReply with JSON.")
            payload = json.loads(raw.strip())
            strategy = RetrievalStrategy(str(payload.get("strategy", "")).strip().upper())
        except (ValueError, TypeError, json.JSONDecodeError):
            return QueryProfile(confidence=0.0, reason="LLM classification failed; rules are inconclusive.")
        return QueryProfile(
            strategy=strategy,
            confidence=0.5,
            reason=str(payload.get("reason") or "LLM classified the query."),
        )
