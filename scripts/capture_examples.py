"""Capture example responses from the running backend for before/after comparison.

Usage (backend server must be running on 127.0.0.1:8000):

    python ../scripts/capture_examples.py            # writes scripts/captured_answers.json
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS = [
    ("simple_factual", "What is Chapter 1 of Simulation?"),
    ("definition", "What is Monte Carlo simulation?"),
    ("summary", "Give me a summary of Chapter 1 of Simulation."),
    ("detailed", "Explain Monte Carlo simulation in detail."),
    ("unanswerable", "Who won the football World Cup?"),
    ("greeting", "Hello"),
]

EXTRA_QUESTIONS = [
    ("word_limit_100", "Explain Monte Carlo simulation in about 100 words."),
    ("comparison", "Compare vector and hybrid retrieval."),
    ("syllabus_lookup", "What are the main topics covered in Chapter 1 of Simulation?"),
]

API = "http://127.0.0.1:8000/api/chat"


def main() -> int:
    captured: dict[str, dict] = {}
    all_questions = QUESTIONS + EXTRA_QUESTIONS
    for label, question in all_questions:
        payload = json.dumps({"message": question}).encode("utf-8")
        request = urllib.request.Request(
            API, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
        elapsed = time.perf_counter() - started
        captured[label] = {
            "question": question,
            "answer": body["answer"],
            "provider": body["provider"],
            "strategy": body["strategy"],
            "enough_evidence": body["enough_evidence"],
            "latency_s": round(elapsed, 1),
        }
        print(f"=== {label} ({body['provider']}, {body['strategy']}, {elapsed:.1f}s) ===")
        print(body["answer"])
        print()

    out = Path(__file__).resolve().parent / "captured_answers.json"
    out.write_text(json.dumps(captured, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
