"""Deterministic query router for retrieval strategies.

Decision rules are cheap, readable features of the *query text itself*; no LLM
is called to classify a query. The router signals why it picked a strategy and
how confident it is:

- a course code, acronym, or exact policy term makes lexical matching valuable
  -> HYBRID (high/medium confidence)
- a short conceptual question with no exact identifiers -> NORMAL
- anything uncertain defaults to HYBRID on purpose: lexical matching can only
  add evidence, so it is the safe choice when in doubt.

Pipeline position: request -> router -> strategy -> retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STRATEGY_NORMAL = "normal"
STRATEGY_HYBRID = "hybrid"

#: "CS201", "ENCT 353", "IT-105" -> technical identifiers that need exact match.
_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}[ -]?\d{3,4}\b")

#: Uppercase-initialisms such as GPA, UM, CS, ECTS.
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")

_NUMBER_RE = re.compile(r"\b\d+\b")

#: "and"/"or" hint that a question spans several distinct concepts.
_MULTI_PART_RE = re.compile(r"\b(?:and|or|vs)\b")

#: Words that typically introduce a simple conceptual or factual question.
_CONCEPTUAL_PREFIXES = (
    "what is",
    "what are",
    "what does",
    "what do",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "explain",
    "describe",
    "define",
)

#: Exact policy/regulation vocabulary where word-level matching adds value.
_TERMINOLOGY_TERMS = frozenset(
    {
        "policy",
        "regulation",
        "regulations",
        "ordinance",
        "clause",
        "section",
        "appendix",
        "syllabus",
        "schedule",
        "prerequisite",
        "eligibility",
        "fee",
        "fees",
        "code",
        "credit",
        "credits",
    }
)


@dataclass(frozen=True)
class RoutingDecision:
    """One routing outcome: which strategy to use and why."""

    strategy: str
    reason: str
    confidence: str  # "high" | "medium" | "low"


class QueryRouter:
    """Rule-based router; see module docstring for the decision table."""

    def route(self, query: str) -> RoutingDecision:
        """Return the retrieval strategy for ``query``."""
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")

        hybrid_score = 0
        signals: list[str] = []

        code_match = _COURSE_CODE_RE.search(query)
        if code_match:
            hybrid_score += 2
            signals.append(f"course code '{code_match.group(0)}'")

        acronyms = sorted({match.group(0) for match in _ACRONYM_RE.finditer(query)})
        if acronyms:
            hybrid_score += 1
            signals.append(f"acronym(s) {', '.join(acronyms)}")

        lowered = query.lower()
        terms = [
            term
            for term in _TERMINOLOGY_TERMS
            if re.search(rf"\b{re.escape(term)}\b", lowered)
        ]
        if terms:
            hybrid_score += 1
            signals.append(f"policy term(s) {', '.join(terms)}")

        if len(_NUMBER_RE.findall(query)) >= 2:
            hybrid_score += 1
            signals.append("several exact figures")

        if _MULTI_PART_RE.search(lowered):
            hybrid_score += 1
            signals.append("multi-part question")

        is_conceptual = any(
            lowered.startswith(prefix) for prefix in _CONCEPTUAL_PREFIXES
        ) and len(query.split()) <= 10

        if hybrid_score == 0 and is_conceptual:
            return RoutingDecision(
                strategy=STRATEGY_NORMAL,
                reason="Short conceptual question with no exact identifiers that would need lexical matching.",
                confidence="medium",
            )

        if hybrid_score >= 2:
            return RoutingDecision(
                strategy=STRATEGY_HYBRID,
                reason=f"Lexical matching is valuable: {', '.join(signals)}.",
                confidence="high",
            )

        if hybrid_score == 1:
            return RoutingDecision(
                strategy=STRATEGY_HYBRID,
                reason=f"Exact terminology present: {', '.join(signals)}.",
                confidence="medium",
            )

        return RoutingDecision(
            strategy=STRATEGY_HYBRID,
            reason="No confident signal; defaulting to hybrid so lexical matches are not missed.",
            confidence="low",
        )