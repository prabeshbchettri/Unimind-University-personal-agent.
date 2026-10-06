"""Generate the deterministic demo corpus used throughout the project.

The corpus is small, fully synthetic and university-themed so the demo never
depends on private or random internet documents. Re-running the script
regenerates identical PDFs (fixed layout, no timestamps, no randomness).

Usage (from backend/, so the project virtualenv is used)::

    python ../scripts/generate_demo_corpus.py
"""

from __future__ import annotations

from pathlib import Path

import fitz

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"

PAGE_WIDTH, PAGE_HEIGHT = 595, 842  # A4 in points
MARGIN = 60
FONT = "helv"
FONT_SIZE = 11
LINE_HEIGHT = 18


def render_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    """Write one PDF with the given paragraphs laid out deterministically."""
    document = fitz.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN

    page.insert_text((MARGIN, y), title, fontname="helv", fontsize=15)
    y += 2 * LINE_HEIGHT

    for paragraph in paragraphs:
        for line in _wrap(paragraph, width_chars=92):
            if y > PAGE_HEIGHT - MARGIN:
                page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                y = MARGIN
            page.insert_text((MARGIN, y), line, fontname=FONT, fontsize=FONT_SIZE)
            y += LINE_HEIGHT
        y += LINE_HEIGHT // 2

    document.save(path)
    document.close()
    print(f"wrote {path.name}")


def _wrap(text: str, width_chars: int) -> list[str]:
    """Wrap text into lines of at most ``width_chars`` characters."""
    lines: list[str] = []
    for paragraph_line in text.splitlines():
        words = paragraph_line.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) > width_chars:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}"
        lines.append(current)
    return lines


def main() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    render_pdf(
        DOCUMENTS_DIR / "attendance_policy.pdf",
        "University Attendance Policy",
        [
            "Students are required to attend at least 75 percent of all scheduled "
            "sessions in every registered course. Attendance is recorded at the "
            "start of each lecture, laboratory session, and seminar.",
            "A student whose attendance falls below the 75 percent threshold in any "
            "course becomes ineligible to sit the final examination in that course. "
            "The student must then apply for condonation through the department "
            "office and, if condonation is approved, may be permitted to take the "
            "examination subject to the conditions stated by the examination "
            "committee.",
            "Medical leave supported by a certificate from the university health "
            "center is excluded from the attendance calculation. Certificates must "
            "be submitted within seven working days of the last day of leave.",
            "Requests for attendance relaxation are reviewed once per semester, "
            "after the final week of teaching. Decisions are communicated by the "
            "department office by email.",
        ],
    )

    render_pdf(
        DOCUMENTS_DIR / "examination_rules.pdf",
        "Examination Rules and Eligibility",
        [
            "Eligibility to sit a semester-end examination requires registration in "
            "good standing in the course, settlement of all outstanding fees, and "
            "satisfactory attendance as defined in the University Attendance "
            "Policy.",
            "Students must carry their university identity card to every "
            "examination. Entry to the examination hall closes fifteen minutes "
            "after the scheduled start time, and no candidate may leave during the "
            "first thirty minutes.",
            "Examination scores are released through the student portal within "
            "three weeks of the examination date. A re-evaluation request may be "
            "filed within ten working days of score release and requires a fee "
            "that is refunded if the score changes.",
            "Any candidate found in possession of unauthorized material during an "
            "examination is referred to the academic integrity committee.",
        ],
    )

    render_pdf(
        DOCUMENTS_DIR / "cs201_syllabus.pdf",
        "CS201 Data Structures - Course Syllabus",
        [
            "CS201 Data Structures is a four-credit core course offered in the "
            "third semester. The course introduces linear structures including "
            "arrays, linked lists, stacks, and queues, followed by trees, hash "
            "tables, and elementary graph algorithms.",
            "Assessment for CS201 consists of two midterm tests worth 20 percent "
            "each, five programming assignments worth 20 percent in total, and a "
            "final examination worth 40 percent. The final examination is a "
            "closed-book written paper lasting three hours.",
            "The prerequisite for CS201 is CS101 Introduction to Programming. "
            "Students who have not passed CS101 must obtain consent from the "
            "course instructor before registering.",
            "The recommended textbook is 'Data Structures and Algorithm Analysis' "
            "by Mark Allen Weiss. Lecture notes are published weekly on the course "
            "portal.",
        ],
    )

    render_pdf(
        DOCUMENTS_DIR / "library_policy.pdf",
        "Library and Study Facilities Policy",
        [
            "Undergraduate students may borrow up to six books at a time for a "
            "loan period of fourteen days. Renewal is permitted twice provided no "
            "reservation exists against the title.",
            "Overdue items accrue a fine of two units per day per item. Borrowing "
            "privileges are suspended while fines remain unpaid.",
            "The library is open from 08:00 to 22:00 on teaching days and from "
            "10:00 to 18:00 on weekends. Group study rooms may be reserved up to "
            "seven days in advance for sessions of up to two hours.",
        ],
    )

    print(f"Demo corpus written to {DOCUMENTS_DIR}")


if __name__ == "__main__":
    main()
