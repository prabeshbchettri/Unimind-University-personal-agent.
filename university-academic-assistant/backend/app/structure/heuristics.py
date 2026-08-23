"""Deterministic regex/heuristic helpers for metadata and structure extraction.

All helpers return ``None`` (or empty collections) when no confident match is
found; values are never invented.
"""

from __future__ import annotations

import re

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_WORD_NUMBERS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def _roman_to_int(value: str) -> int | None:
    if not value:
        return None
    total = 0
    prev = 0
    for char in reversed(value.upper()):
        current = _ROMAN.get(char)
        if current is None:
            return None
        if current < prev:
            total -= current
        else:
            total += current
        prev = current
    return total if total > 0 else None


def find_semester(text: str) -> int | None:
    """Find the semester number (1-12) mentioned in ``text``."""
    from collections import Counter

    candidates: list[int] = []
    for match in re.finditer(r"\bSemester\s+([IVXLC]+)\b", text, re.IGNORECASE):
        value = _roman_to_int(match.group(1))
        if value:
            candidates.append(value)
    for match in re.finditer(r"\bSemester\s+([0-9]{1,2})\b", text):
        candidates.append(int(match.group(1)))
    for match in re.finditer(r"\b([0-9]{1,2})\s*(?:st|nd|rd|th)\s+Semester\b", text, re.IGNORECASE):
        candidates.append(int(match.group(1)))
    for match in re.finditer(
        r"\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\s+Semester\b",
        text,
        re.IGNORECASE,
    ):
        candidates.append(_WORD_NUMBERS[match.group(1).lower()])
    if not candidates:
        return None
    return Counter(candidates).most_common(1)[0][0]


def find_year(text: str) -> int | None:
    """Find a 4-digit calendar/examination year (1900-2100)."""
    for match in re.finditer(r"\b(1[89]\d{2}|20\d{2})\b", text):
        value = int(match.group(1))
        if 1900 <= value <= 2100:
            return value
    return None


def find_academic_year(text: str) -> str | None:
    """Find an academic year range such as ``2080/2081`` or ``2024-25``.

    Dates like ``2024-05-01`` are rejected so a day-level date is not mistaken
    for an academic year.
    """
    match = re.search(
        r"\b(20\d{2})\s?[/\-]\s?(\d{2,4})\b(?!\s?-\s?\d{1,2}\b)", text
    )
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def find_subject_code(text: str) -> str | None:
    """Find a subject/course code such as ``BCS-502`` or ``CSC301``."""
    match = re.search(r"\b([A-Z]{2,6})\s?[-–]?\s?(\d{3})\b", text)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def find_marks(text: str) -> int | None:
    """Find a marks value (e.g. ``10 marks``, ``Marks: 5``)."""
    patterns = (
        re.compile(r"\b(?:Marks?|marks)\s*[:=]?\s*(\d{1,3})\b"),
        re.compile(r"\b(\d{1,3})\s*(?:Marks?|marks)\b"),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def find_university(text: str, extra_names: list[str] = ()) -> str | None:
    """Find a university name mentioned in ``text``."""
    if extra_names:
        for name in extra_names:
            if name and name in text:
                return name.strip()
    match = re.search(
        r"([A-Z][A-Za-z .&'’-]{2,40}\s+University)", text, re.IGNORECASE
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return None


def find_program(text: str) -> str | None:
    """Find an academic program mentioned in ``text``."""
    patterns = (
        r"Bachelor of Science in Computer Science and Information Technology",
        r"Bachelor of Computer Application\w*",
        r"Bachelor of Business Administration",
        r"Bachelor of Information Technology",
        r"Bachelor of Computer Science",
        r"Bachelor of [A-Z][A-Za-z ]{2,40}",
        r"B\.?\s?Sc\.?\s?(?:in\s+)?CSIT\b",
        r"B\.?CA\b",
        r"B\.?IM\b",
        r"B\.?IT\b",
        r"B\.?BS\b",
        r"B\.?BA\b",
        r"M\.?\s?Sc\.?",
        r"M\.?CA\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return None


_QUESTION_LINE_RE = re.compile(
    r"^(?:Q\s*\.?\s*(\d+)\s*[):.]\s*|\((\d+)\)\s*|(\d+)[.):]\s*)(.*)$",
    re.IGNORECASE,
)
_MARKS_PHRASE_RE = re.compile(
    r"\s*(?:\(?\s*\d{1,3}\s*marks?\s*\)?|marks?\s*[:=]?\s*\d{1,3})\s*$",
    re.IGNORECASE,
)


def extract_questions(text: str) -> list[dict]:
    """Extract numbered questions with marks from a question paper.

    Returns a list of ``{"question_number", "text", "marks"}`` dicts. Marks are
    parsed from the text and removed from the question body.
    """
    questions: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _QUESTION_LINE_RE.match(stripped)
        if match:
            number = int(match.group(1) or match.group(2) or match.group(3))
            if current:
                questions.append(current)
            current = {
                "question_number": number,
                "text": match.group(4).strip(),
                "marks": None,
            }
        elif current:
            current["text"] = f"{current['text']} {stripped}".strip()
    if current:
        questions.append(current)

    for question in questions:
        question["marks"] = find_marks(question["text"])
        question["text"] = _MARKS_PHRASE_RE.sub("", question["text"]).strip()
    return questions


_QUESTION_VERB_RE = re.compile(
    r"^\s*(?:Explain|Define|Describe|Discuss|State|Write\s+short\s+notes?\s+on\s*"
    r"|What\s+is\s+|What\s+are\s+|What\s+do\s+you\s+mean\s+by\s+|Differentiate\s+between\s+"
    r"|Compare\s+and\s+contrast\s+|Compare\s+|Distinguish\s+between\s+|List\s+|Mention\s+"
    r"|Enumerate\s+|Justify\s+|Prove\s+|Simplify\s+|Evaluate\s+|Draw\s+and\s+explain\s+"
    r"|Define\s+and\s+explain\s+)\s*",
    re.IGNORECASE,
)


def topic_from_question(text: str) -> str | None:
    """Derive a short topic from a question body (e.g. ``normalization``)."""
    if not text:
        return None
    cleaned = _QUESTION_VERB_RE.sub("", text)
    cleaned = re.split(r"[,.:;]|\s+(?:and|with)\s+", cleaned)[0].strip()
    if not cleaned or len(cleaned) > 120:
        return None
    return cleaned


_UNIT_RE = re.compile(r"^Unit\s+([IVXLC\d]+)\s*[:.\-]?\s*(.*)$", re.MULTILINE | re.IGNORECASE)
_MODULE_RE = re.compile(r"^Module\s+([IVXLC\d]+)\s*[:.\-]?\s*(.*)$", re.MULTILINE | re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+)\s*[:.)]\s*(.+)$")


def extract_syllabus_topics(text: str) -> list[str]:
    """Extract topic names from a syllabus.

    Prefers ``Unit``/``Module`` headings; otherwise falls back to short
    heading-like lines (e.g. ``Normalization``, ``Transaction Processing``).
    """
    topics: list[str] = []
    for match in _UNIT_RE.finditer(text):
        title = _clean_title(match.group(2))
        if title and title not in topics:
            topics.append(title)
    if not topics:
        for match in _MODULE_RE.finditer(text):
            title = _clean_title(match.group(2))
            if title and title not in topics:
                topics.append(title)
    if not topics:
        topics = extract_heading_lines(text, max_words=8)
    return topics


def extract_heading_lines(text: str, max_words: int = 8) -> list[str]:
    """Return short heading-like lines (candidate topics)."""
    headings: list[str] = []
    stopwords = re.compile(
        r"(?i)(marks?|credits?|semester|page|university|college|faculty|time\s*:|hrs|"
        r"hours|total|date\s*:|subject\s*:|exam|examination|answer|attempt|section|"
        r"tribhuvan|department|course\s+title|course\s+code|course\s+objectives?|"
        r"course\s+description|course\s+outcomes?|recommended|reference|text\s*books?|"
        r"prerequisites?|evaluation|assessment|grading|teaching\s+hours|lab|lectures?|"
        r"question\s+paper|bachelor|level|syllabus|curriculum)"
    )
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or len(candidate) < 3:
            continue
        if not (candidate[0].isupper() or candidate[0].isdigit()):
            continue
        words = candidate.split()
        if not 1 <= len(words) <= max_words:
            continue
        if re.fullmatch(r"[\d\s.:/\-–—]+", candidate):
            continue
        if candidate.endswith(".") and len(words) > 4:
            continue
        if re.search(r"[A-Za-z]", candidate) is None:
            continue
        if stopwords.search(candidate):
            continue
        if candidate not in headings:
            headings.append(candidate)
    return headings


def extract_chapters(text: str) -> list[dict]:
    """Extract chapters from a library book.

    Detects ``Chapter N`` headings in the body and dotted table-of-contents
    lines. Returns ``{"chapter_number", "title"}`` dicts.
    """
    chapters: list[dict] = []
    seen: set[str] = set()
    body_pattern = re.compile(
        r"^Chapter\s+(\d+|[IVXLC]+)\s*[:.\-]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
    )
    for match in body_pattern.finditer(text):
        title = _clean_title(match.group(2))
        if not title:
            continue
        number = _roman_to_int(match.group(1)) if not match.group(1).isdigit() else int(match.group(1))
        key = f"{number}:{title.lower()}"
        if key not in seen:
            seen.add(key)
            chapters.append({"chapter_number": number, "title": title})

    toc_pattern = re.compile(
        r"^(?:Chapter\s+)?(\d+)\s*[:.]?\s+([A-Z][A-Za-z &'.,:/-]{2,60}?)\s*\.\.+\s*\d+\s*$",
        re.MULTILINE,
    )
    for match in toc_pattern.finditer(text):
        title = _clean_title(match.group(2))
        number = int(match.group(1))
        key = f"{number}:{title.lower()}"
        if title and key not in seen:
            seen.add(key)
            chapters.append({"chapter_number": number, "title": title})
    return chapters


def extract_subtopic_map(text: str) -> dict[str, str]:
    """Map numbered subsection keys (``1.1``, ``2.3.1``) to their titles."""
    pattern = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(.*)$", re.MULTILINE)
    return {
        match.group(1): _clean_title(match.group(2))
        for match in pattern.finditer(text)
        if _clean_title(match.group(2))
    }


def find_author(text: str) -> str | None:
    """Find an author line (``by <name>`` or ``Author: <name>``)."""
    pattern = re.compile(
        r"^(?:by|author)\s*[:]?\s*([A-Za-z][A-Za-z .'&,-]{3,60})$",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return None


def subject_from_syllabus(text: str) -> str | None:
    """Extract the subject name from a syllabus heading.

    ``Database Management System Syllabus`` -> ``Database Management System``.
    """
    for match in re.finditer(r"^(.+?)\s+(?:Syllabus|Curriculum|Course)\b", text, re.IGNORECASE | re.MULTILINE):
        subject = _clean_title(match.group(1))
        if subject and len(subject.split()) >= 2:
            return subject
    return None


_SKIP_SUBJECT_RE = re.compile(
    r"(?i)(university|time\s*:|marks|max(?:imum)?|exam|attempt|full\s+marks|date\s*:"
    r"|semester|faculty|campus|roll\s*no|year\s*:|level|bachelor|group|college|"
    r"department|tribhuvan|code\s*:|credit|candidates?|required|answer|questions?)"
)


def subject_from_past_paper(text: str) -> str | None:
    """Heuristically find the subject name in a past question paper."""
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or len(candidate) < 5 or len(candidate) > 80:
            continue
        if re.fullmatch(r"[\d\s.:/\-–—()]+", candidate):
            continue
        if _SKIP_SUBJECT_RE.search(candidate):
            continue
        if len(re.findall(r"[A-Za-z]", candidate)) < 5:
            continue
        return re.sub(r"\s+", " ", candidate).strip(" .:-")
    return None


def first_substantial_line(first_page_text: str) -> str | None:
    """Return the first non-trivial line of the first page."""
    for line in first_page_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        words = candidate.split()
        if len(words) < 2 or len(candidate) > 200:
            continue
        if re.fullmatch(r"[\d\s.:/\-–—]+", candidate):
            continue
        if re.search(r"[A-Za-z]", candidate) is None:
            continue
        return candidate
    return None


def subject_line(text: str) -> str | None:
    """Extract a notice subject line (``Sub: ...`` / ``Subject: ...``)."""
    match = re.search(r"^(?:Subject|Sub)\s*[:.]?\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if match:
        return _clean_title(match.group(1))
    return None


def _clean_title(value: str) -> str | None:
    """Clean a candidate title/heading string."""
    cleaned = re.sub(r"\s+", " ", value or "").strip(" .:-–—")
    if not cleaned:
        return None
    return cleaned
