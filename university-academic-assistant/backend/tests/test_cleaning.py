"""Tests for conservative text cleaning."""

from app.ingestion import TextCleaner


def test_collapses_multiple_spaces() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("normalization  is   important") == "normalization is important"


def test_collapses_excessive_blank_lines() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("Unit 1\n\n\n\n\nUnit 2") == "Unit 1\n\nUnit 2"


def test_removes_control_characters() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("DBMS\x00\x01 \x07semester") == "DBMS semester"


def test_removes_invisible_and_nbsp_characters() -> None:
    cleaner = TextCleaner()
    # nbsp becomes a space, zero-width space is removed entirely.
    assert cleaner.clean("A\u00a0B\u200bC") == "A BC"


def test_normalizes_line_endings() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("line1\r\nline2\rline3") == "line1\nline2\nline3"


def test_preserves_paragraph_breaks_and_content() -> None:
    cleaner = TextCleaner()
    text = "What is DBMS?\n\nA DBMS is a software system."
    assert cleaner.clean(text) == text


def test_empty_input_returns_empty() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("") == ""
    assert cleaner.clean("   \n\n  ") == ""


def test_does_not_rewrite_academic_content() -> None:
    cleaner = TextCleaner()
    text = "BCS-502 Database Management Systems (3 credits)"
    assert cleaner.clean(text) == text