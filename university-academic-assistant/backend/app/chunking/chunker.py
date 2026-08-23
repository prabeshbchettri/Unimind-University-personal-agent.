"""Semantic chunker.

Converts a structured document into searchable chunks while preserving the
semantic context that a flat character split would tear apart. Content that
belongs together (a topic heading, its definition and its explanation) is kept
in the same chunk, and every chunk carries its provenance (document, page,
section, topic, subtopic) plus document-level metadata.

Chunking is type-aware:

- ``past_question`` — each question becomes its own block
- ``library_book`` — chapter / subtopic headings open new blocks
- ``syllabus`` — unit/module/topic headings open new blocks
- everything else — paragraph-based blocks

Blocks are grouped into chunks capped at ``max_chars``; a trailing overlap of
``overlap_chars`` is carried into the next chunk so context is not lost at
boundaries. Only an oversized single block is ever split, and only at sentence
boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.chunking.models import DocumentChunk
from app.structure.heuristics import find_marks, topic_from_question
from app.structure.models import StructuredDocument

_QUESTION_RE = re.compile(
    r"^\s*(?:Q\s*\.?\s*(\d+)|\((\d+)\)\s*|(\d+)[).:])\s*(.*)$",
    re.IGNORECASE,
)
_CHAPTER_RE = re.compile(r"^Chapter\s+(\d+|[IVXLC]+)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
_SUBTOPIC_RE = re.compile(r"^(\d+\.\d+)\s*[:.\-]?\s*(.*)$")
_UNIT_RE = re.compile(r"^(Unit|Module)\s+(\d+)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)


@dataclass
class _Block:
    """A semantic unit (heading + its content) within a page."""

    page: int
    text: str
    section: str | None = None
    topic: str | None = None
    subtopic: str | None = None
    question_number: int | None = None
    marks: int | None = None


def split_paragraphs(text: str) -> list[str]:
    """Split ``text`` into non-empty paragraphs (blank-line separated)."""
    return [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]


def _clean_title(value: str | None) -> str | None:
    cleaned = re.sub(r"\s+", " ", value or "").strip(" .:-–—")
    return cleaned or None


class SemanticChunker:
    """Splits :class:`StructuredDocument` into context-preserving chunks."""

    def __init__(self, max_chars: int = 1200, overlap_chars: int = 150) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, structured: StructuredDocument) -> list[DocumentChunk]:
        """Return chunks for ``structured`` (possibly empty)."""
        if not structured.pages:
            return []

        self._topics_set = set(structured.structure.topics or ())
        blocks: list[_Block] = []
        for page in structured.pages:
            blocks.extend(self._page_blocks(structured, page))
        blocks = [block for block in blocks if block.text.strip()]
        blocks = self._split_oversized(blocks)

        return [
            self._build_chunk(structured, group, index)
            for index, group in enumerate(self._group_blocks(blocks))
        ]

    # --- block construction -------------------------------------------------

    def _page_blocks(self, structured: StructuredDocument, page) -> list[_Block]:
        doc_type = structured.metadata.document_type
        if doc_type == "past_question":
            return self._question_blocks(page)
        if doc_type == "library_book":
            return self._book_blocks(page)
        if doc_type == "syllabus":
            return self._syllabus_blocks(page)
        return self._generic_blocks(page)

    @staticmethod
    def _lines(page) -> list[str]:
        return [line.rstrip() for line in page.text.splitlines() if line.strip()]

    def _question_blocks(self, page) -> list[_Block]:
        blocks: list[_Block] = []
        current: _Block | None = None
        for line in self._lines(page):
            match = _QUESTION_RE.match(line)
            if match:
                if current is not None:
                    blocks.append(current)
                number = int(match.group(1) or match.group(2) or match.group(3))
                body = (match.group(4) or "").lstrip(" .:;–-")
                current = _Block(
                    page=page.page_number,
                    text=line,
                    section="question",
                    topic=topic_from_question(body),
                    question_number=number,
                )
            else:
                if current is None:
                    current = _Block(page=page.page_number, text="")
                current.text = f"{current.text}\n{line}" if current.text else line
        if current is not None and current.text.strip():
            blocks.append(current)
        for block in blocks:
            if block.question_number is not None and block.marks is None:
                block.marks = find_marks(block.text)
        return blocks

    def _book_blocks(self, page) -> list[_Block]:
        blocks: list[_Block] = []
        current: _Block | None = None
        for line in self._lines(page):
            chapter = _CHAPTER_RE.match(line)
            if chapter:
                if current is not None:
                    blocks.append(current)
                current = _Block(
                    page=page.page_number,
                    text=line,
                    section="chapter",
                    topic=_clean_title(chapter.group(2)) or None,
                )
                continue
            subtopic = _SUBTOPIC_RE.match(line)
            if subtopic:
                if current is not None:
                    blocks.append(current)
                current = _Block(
                    page=page.page_number,
                    text=line,
                    section="chapter",
                    topic=current.topic if current else None,
                    subtopic=_clean_title(subtopic.group(2)),
                )
                continue
            if current is None:
                current = _Block(page=page.page_number, text="", section="chapter")
            current.text = f"{current.text}\n{line}" if current.text else line
        if current is not None and current.text.strip():
            blocks.append(current)
        return blocks

    def _syllabus_blocks(self, page) -> list[_Block]:
        blocks: list[_Block] = []
        current: _Block | None = None
        topics = getattr(self, "_topics_set", set())
        for line in self._lines(page):
            topic_hit: str | None = None
            unit = _UNIT_RE.match(line)
            if unit:
                topic_hit = _clean_title(unit.group(3)) or f"{unit.group(1).title()} {unit.group(2)}"
            elif line.strip() in topics:
                topic_hit = line.strip()
            if topic_hit:
                if current is not None:
                    blocks.append(current)
                current = _Block(
                    page=page.page_number,
                    text=line,
                    section="topic",
                    topic=topic_hit,
                )
                continue
            if current is None:
                current = _Block(page=page.page_number, text="", section="topic")
            current.text = f"{current.text}\n{line}" if current.text else line
        if current is not None and current.text.strip():
            blocks.append(current)
        return blocks

    def _generic_blocks(self, page) -> list[_Block]:
        return [
            _Block(page=page.page_number, text=paragraph)
            for paragraph in split_paragraphs(page.text)
        ]

    # --- grouping -----------------------------------------------------------

    def _split_oversized(self, blocks: list[_Block]) -> list[_Block]:
        out: list[_Block] = []
        for block in blocks:
            if len(block.text) <= self.max_chars:
                out.append(block)
                continue
            for piece in self._split_text(block.text, self.max_chars):
                out.append(
                    _Block(
                        page=block.page,
                        text=piece,
                        section=block.section,
                        topic=block.topic,
                        subtopic=block.subtopic,
                        question_number=block.question_number,
                        marks=block.marks,
                    )
                )
        return out

    @staticmethod
    def _split_text(text: str, max_chars: int) -> list[str]:
        """Split an oversized block at sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > max_chars:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence
        if current:
            pieces.append(current)
        return pieces

    def _group_blocks(self, blocks: list[_Block]) -> list[list[_Block]]:
        groups: list[list[_Block]] = []
        current: list[_Block] = []
        size = 0
        carry: list[_Block] = []
        for block in blocks:
            if current and self._is_heading(block):
                # A unit/chapter heading always starts a new chunk so the
                # chunk's topic payload reflects its own heading, never a
                # heading glued to the previous chunk's tail.
                groups.append(carry + current)
                carry = self._overlap_tail(current)
                current = []
                size = 0
            block_size = len(block.text) + 2
            if current and size + block_size > self.max_chars:
                groups.append(carry + current)
                carry = self._overlap_tail(current)
                current = []
                size = 0
            current.append(block)
            size += block_size
        if current or carry:
            groups.append(carry + current)
        return groups

    @staticmethod
    def _is_heading(block: _Block) -> bool:
        """Whether ``block`` opens with a topic/chapter heading."""
        if block.section == "topic":
            return block.topic is not None
        if block.section == "chapter":
            return block.subtopic is None
        return False

    def _overlap_tail(self, blocks: list[_Block]) -> list[_Block]:
        """Return the trailing content blocks (whole, never split) used as
        overlap. Heading blocks are excluded: an overlap carries *context*,
        never a section identity — otherwise the next chunk's topic payload
        would be inherited from the previous chapter's heading."""
        if self.overlap_chars <= 0 or not blocks:
            return []
        tail: list[_Block] = []
        used = 0
        for block in reversed(blocks):
            # A pure heading line (e.g. "Chapter 2: X") carries no content of
            # its own, so it must not become the next chunk's identity. Merged
            # syllabus units ("Unit 1: A\nsome text about A") are content and
            # stay in the overlap.
            if self._is_heading(block) and block.text.strip().count("\n") == 0:
                continue
            used += len(block.text) + 2
            if used > self.overlap_chars:
                break
            tail.append(block)
        return list(reversed(tail))

    def _build_chunk(
        self,
        structured: StructuredDocument,
        group: list[_Block],
        index: int,
    ) -> DocumentChunk:
        meta_block = next((b for b in group if self._is_heading(b)), group[0])
        first = meta_block
        meta = structured.metadata
        # A short self-describing header helps the embedding stay contextually
        # tied to the chunk's topic. Prefer the most specific label available.
        header = first.subtopic or first.topic or meta.title
        content = "\n\n".join(block.text for block in group)
        text = f"{header}\n\n{content}" if header else content

        return DocumentChunk(
            chunk_id=f"{structured.document_id}:{index:04d}",
            document_id=structured.document_id,
            document_type=meta.document_type,
            title=meta.title,
            author=meta.author,
            page=first.page,
            section=first.section,
            topic=first.topic,
            subtopic=first.subtopic,
            semester=meta.semester,
            subject=meta.subject,
            subject_code=meta.subject_code,
            academic_year=meta.academic_year,
            marks=first.marks if first.marks is not None else meta.marks,
            question_number=(
                first.question_number
                if first.question_number is not None
                else meta.question_number
            ),
            text=text,
        )
