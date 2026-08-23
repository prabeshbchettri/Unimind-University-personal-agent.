"""Conservative text cleaning for extracted PDF text.

Cleaning is deliberately minimal: it normalizes whitespace and removes obvious
extraction artifacts while preserving the actual academic content. No semantic
or topic-level processing happens here.
"""

from __future__ import annotations

import re
import unicodedata

# Control characters other than tab and newline are extraction noise.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Zero-width and other invisible format characters.
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
# Three or more consecutive whitespace characters collapse to a single space.
_WHITESPACE_RUN_RE = re.compile(r"[ \t]{2,}")
# Three or more consecutive blank lines collapse to one.
_BLANK_LINES_RE = re.compile(r"\n{3,}")


class TextCleaner:
    """Normalize extracted page text without changing its meaning."""

    def clean(self, text: str) -> str:
        """Clean a block of extracted text."""
        if not text:
            return ""

        # Normalize line endings to Unix style.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove non-breaking/hard spaces and invisible characters.
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00a0", " ")
        text = _INVISIBLE_RE.sub("", text)
        # Drop control characters that are not structural.
        text = _CONTROL_CHARS_RE.sub("", text)
        # Collapse runs of spaces (within a line).
        text = _WHITESPACE_RUN_RE.sub(" ", text)
        # Trim leading/trailing whitespace on each line (layout artifacts)
        # while preserving blank lines and paragraph structure.
        text = "\n".join(line.strip() for line in text.split("\n"))
        # Collapse excessive blank lines, preserving a single paragraph break.
        text = _BLANK_LINES_RE.sub("\n\n", text)
        # Trim leading/trailing whitespace.
        return text.strip()

    def clean_page(self, text: str) -> str:
        """Clean a single page's text (alias of :meth:`clean`)."""
        return self.clean(text)
