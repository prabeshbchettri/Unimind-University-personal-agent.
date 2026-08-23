"""End-to-end acceptance test through a running server (Phase 13).

Runs the six acceptance scenario types against the real HTTP API:

    S1  syllabus question       -> NORMAL dense RAG
    S2  regulation exact match  -> HYBRID (dense + BM25)
    S3  structural graph query  -> GRAPH (knowledge graph + dense fallback)
    S4  past questions          -> GRAPH (question -> topic)
    S5  book recommendation     -> /recommendations coverage scoring
    S6  web search              -> WEB (current/external info)

It builds small PDFs from scratch, uploads them through the real
``POST /documents/index`` pipeline (classify -> structure -> chunk -> embed ->
index -> graph), then exercises every scenario plus the chat session
lifecycle and operational endpoints. All checks run against a live server,
so it also exercises the request logging and caching layers.

Usage:
    python scripts/e2e.py [--url http://localhost:8000] [--out e2e_results.json]

Exit code 0 only when every scenario passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pymupdf

# ---------------------------------------------------------------------------
# Scenario documents (text designed for the deterministic pipeline)
# ---------------------------------------------------------------------------

S1_SYLLABUS = {
    "filename": "syllabus-dbms.pdf",
    "text": (
        "Database Management System Syllabus\n\n"
        "Semester: 5\n\n"
        "Unit 1: Introduction\n1.1 Data Models\n"
        "The unit covers the core ideas of Introduction.\n\n"
        "Unit 2: Normalization\n2.1 Normal Forms\n"
        "Normalization removes redundancy in tables.\n\n"
        "Unit 3: Transactions\n3.1 ACID Properties\n"
        "Transactions keep data consistent.\n"
    ),
    "expect_type": "syllabus",
}

S1B_SYLLABUS_DSA = {
    "filename": "syllabus-dsa.pdf",
    "text": (
        "Data Structures and Algorithms Syllabus\n\n"
        "Semester: 5\n\n"
        "Unit 1: Arrays\n1.1 Array Operations\n"
        "Unit 2: Linked Lists\n2.1 Singly Linked Lists\n"
        "Unit 3: Trees\n3.1 Binary Trees\n"
        "Unit 4: Sorting\n4.1 Quick Sort\n"
    ),
    "expect_type": "syllabus",
}

S2_REGULATIONS = {
    "filename": "regulations-exam.pdf",
    "text": (
        "Examination Regulations 2024\n\n"
        "Rule 8: Grading\n"
        "Marks are converted to grade points: 80-100 is A, 60-79 is B, "
        "40-59 is C, below 40 is F. A student failing a subject must "
        "retake it in the next semester.\n\n"
        "Rule 12: the use of mobile phones inside the examination hall is "
        "prohibited.\n"
    ),
    "expect_type": "rules_regulations",
}

S3_PAST_PAPER = {
    "filename": "pastpaper-dbms-2080.pdf",
    "text": (
        "DBMS Past Question Paper 2080\n"
        "Database Management System\n"
        "Semester: 5\n\n"
        "Q1. Explain the concept of normalization and its normal forms. "
        "(10 marks)\n"
        "Q2. Write SQL queries for the student database. (8 marks)\n"
        "Q3. Discuss ACID properties of transactions. (7 marks)\n"
    ),
    "expect_type": "past_question",
}

S5_BOOK = {
    "filename": "book-dbms-concepts.pdf",
    "text": (
        "Database System Concepts\n"
        "Author: Abraham Silberschatz\n\n"
        "Chapter 1: Introduction\n1.1 Data Models\n\n"
        "Chapter 2: Normalization\n2.1 Normal Forms\n"
        "Normalization removes redundancy in tables.\n\n"
        "Chapter 3: Transactions\n3.1 ACID Properties\n"
        "Transactions keep data consistent.\n"
    ),
    "expect_type": "library_book",
}

S2B_NOTICE = {
    "filename": "notice-attendance.pdf",
    "text": (
        "Notice\n"
        "Subject: Attendance Requirement 2026\n\n"
        "All students must attend at least 75 percent of classes in every "
        "subject. Students below the threshold are not eligible to sit the "
        "final examination. Attendance is recorded per lecture by the class "
        "teacher.\n"
    ),
    "expect_type": "notice",
}

S2C_CALENDAR = {
    "filename": "calendar-2024.pdf",
    "text": (
        "Academic Calendar 2024\n\n"
        "First semester: 2024-02-12 to 2024-06-28. Exam week starts on "
        "2024-06-17. Spring holidays: 2024-04-09 to 2024-04-15. Second "
        "semester begins 2024-07-15.\n"
    ),
    "expect_type": "academic_calendar",
}

DOCUMENTS = [S1_SYLLABUS, S1B_SYLLABUS_DSA, S2_REGULATIONS, S3_PAST_PAPER, S5_BOOK, S2B_NOTICE, S2C_CALENDAR]

# ---------------------------------------------------------------------------
# Scenario checks
# ---------------------------------------------------------------------------


def _write_pdfs() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="e2e_docs_"))
    for doc in DOCUMENTS:
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), doc["text"], fontsize=11)
        pdf.save(str(directory / doc["filename"]))
        pdf.close()
    return directory


def _check(label: str, ok: bool, detail: str) -> dict:
    return {"scenario": label, "pass": bool(ok), "detail": detail}


def run_e2e(url: str) -> tuple[list[dict], dict]:
    results: list[dict] = []
    client = httpx.Client(base_url=url, timeout=120.0)
    errors: list[str] = []

    def record(label: str, ok: bool, detail: str) -> None:
        entry = _check(label, ok, detail)
        results.append(entry)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {detail[:200]}")
        if not ok:
            errors.append(label)

    # --- Operational: health -------------------------------------------------
    health = client.get("/health")
    record("health", health.status_code == 200, f"GET /health -> {health.status_code} {health.json() if health.status_code == 200 else health.text}")

    # --- Index the scenario corpus through the real pipeline -----------------
    docs_dir = _write_pdfs()
    index_results: list[dict] = []
    for doc in DOCUMENTS:
        with open(docs_dir / doc["filename"], "rb") as handle:
            response = client.post(
                "/documents/index",
                files={"file": (doc["filename"], handle, "application/pdf")},
            )
        body = response.json() if response.status_code == 200 else {}
        ok = response.status_code == 200 and body.get("document_type") == doc["expect_type"]
        record(f"index {doc['filename']}", ok, f"status={response.status_code}, type={body.get('document_type')}, chunks={body.get('chunk_count')}")
        index_results.append({"file": doc["filename"], "status": response.status_code, **body})
    graph_summary = client.get("/graph/summary")
    record("graph summary", graph_summary.status_code == 200, f"GET /graph/summary -> {graph_summary.status_code} {graph_summary.text[:160]}")

    # --- S1: syllabus question (NORMAL) --------------------------------------
    r = client.post("/chat", json={"message": "Explain normalization."})
    body = r.json() if r.status_code == 200 else {}
    ok = r.status_code == 200 and body.get("strategy") == "NORMAL" and body.get("sources")
    record("S1 syllabus/NORMAL", ok, f"strategy={body.get('strategy')}, sources={len(body.get('sources', []))}, answer={str(body.get('answer'))[:80]!r}")

    # --- S2: regulation exact match (HYBRID) ---------------------------------
    r = client.post("/chat", json={"message": "What does rule 8 say about grading?"})
    body = r.json() if r.status_code == 200 else {}
    titles = [s.get("metadata", {}).get("title") or "" for s in body.get("sources", [])]
    ok = r.status_code == 200 and body.get("strategy") == "HYBRID" and any("Examination Regulations" in t for t in titles)
    record("S2 regulation/HYBRID", ok, f"strategy={body.get('strategy')}, sources={len(body.get('sources', []))}, titles={titles[:3]}")

    # --- S3: structural graph query (GRAPH) ----------------------------------
    r = client.post("/chat", json={"message": "Which subjects are in semester 5?"})
    body = r.json() if r.status_code == 200 else {}
    ok = r.status_code == 200 and body.get("strategy") == "GRAPH" and body.get("sources")
    record("S3 graph/GRAPH", ok, f"strategy={body.get('strategy')}, sources={len(body.get('sources', []))}, answer={str(body.get('answer'))[:100]!r}")

    # --- S4: past questions (GRAPH) ------------------------------------------
    r = client.post("/chat", json={"message": "Which past questions are about normalization?"})
    body = r.json() if r.status_code == 200 else {}
    ok = r.status_code == 200 and body.get("sources")
    record("S4 past questions", ok, f"strategy={body.get('strategy')}, sources={len(body.get('sources', []))}, answer={str(body.get('answer'))[:100]!r}")

    # --- S5: book recommendation ---------------------------------------------
    r = client.post("/recommendations", json={"query": "Which book is best for normalization?"})
    body = r.json() if r.status_code == 200 else {}
    top = (body.get("recommendations") or [{}])[0]
    ok = r.status_code == 200 and top.get("book") == "Database System Concepts" and top.get("score", 0) > 0
    record("S5 recommendation", ok, f"top={top.get('book')}, score={top.get('score')}, count={len(body.get('recommendations', []))}")

    # --- S6: web search (WEB) ------------------------------------------------
    r = client.post("/chat", json={"message": "What is the latest Python version?"})
    body = r.json() if r.status_code == 200 else {}
    web_kinds = [s.get("kind") for s in body.get("sources", [])]
    ok = r.status_code == 200 and body.get("strategy") == "WEB" and any(k == "web" for k in web_kinds)
    record("S6 web/WEB", ok, f"strategy={body.get('strategy')}, sources={len(body.get('sources', []))}, kinds={web_kinds[:3]}")

    # --- Session lifecycle ----------------------------------------------------
    r = client.post("/chat", json={"message": "What are ACID properties?"})
    session_id = r.json().get("session_id") if r.status_code == 200 else None
    record("session create", r.status_code == 200 and bool(session_id), f"POST /chat -> session_id={session_id}")
    r = client.post("/chat", json={"message": "And transactions?", "session_id": session_id}) if session_id else None
    record("session continue", bool(r and r.status_code == 200 and r.json().get("session_id") == session_id), f"POST /chat with session_id -> {r.status_code if r else 'n/a'}")
    r = client.get(f"/chat/sessions/{session_id}") if session_id else None
    record("session fetch", bool(r and r.status_code == 200 and len(r.json().get("messages", [])) == 4), f"GET /chat/sessions/{session_id} -> {r.status_code if r else 'n/a'}")
    r = client.delete(f"/chat/sessions/{session_id}") if session_id else None
    record("session delete", bool(r and r.status_code == 204), f"DELETE -> {r.status_code if r else 'n/a'}")
    r = client.get(f"/chat/sessions/{session_id}") if session_id else None
    record("session gone", bool(r and r.status_code == 404), f"GET after delete -> {r.status_code if r else 'n/a'}")

    summary = {
        "total": len(results),
        "passed": sum(1 for entry in results if entry["pass"]),
        "failed": [entry for entry in results if not entry["pass"]],
        "indexed_documents": index_results,
        "graph_summary": graph_summary.json() if graph_summary.status_code == 200 else None,
        "elapsed_seconds": None,
    }
    return results, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL.")
    parser.add_argument("--out", default="e2e_results.json", help="JSON results file.")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only upload the scenario documents (no scenario checks). Useful to prepare a live demo corpus.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if args.seed_only:
        client = httpx.Client(base_url=args.url, timeout=120.0)
        health = client.get("/health")
        print(f"health: {health.status_code} {health.json() if health.status_code == 200 else health.text}")
        docs_dir = _write_pdfs()
        for doc in DOCUMENTS:
            with open(docs_dir / doc["filename"], "rb") as handle:
                response = client.post(
                    "/documents/index",
                    files={"file": (doc["filename"], handle, "application/pdf")},
                )
            body = response.json() if response.status_code == 200 else {}
            print(f"index {doc['filename']}: {response.status_code} type={body.get('document_type')} chunks={body.get('chunk_count')}")
        print(f"seeded {len(DOCUMENTS)} documents in {round(time.perf_counter() - started, 1)}s")
        return 0

    results, summary = run_e2e(args.url)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 2)

    output = Path(args.out)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n{summary['passed']}/{summary['total']} checks passed in {summary['elapsed_seconds']}s -> {output}")
    if summary["failed"]:
        print("Failed:", ", ".join(entry["scenario"] for entry in summary["failed"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())