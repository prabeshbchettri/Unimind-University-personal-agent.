"""Routing evaluation set (Phase 8 + Phase 11).

A representative set of 36 academic queries (10 NORMAL, 10 HYBRID, 10 GRAPH,
6 WEB) used to measure routing accuracy, false routing and the fallback rate.
"""

from __future__ import annotations

from app.router.router import AdaptiveRouter

ROUTING_EVAL_SET: list[dict[str, str]] = [
    # --- NORMAL: semantic / conceptual --------------------------------------
    {"query": "Explain normalization.", "expected": "NORMAL"},
    {"query": "What is DBMS?", "expected": "NORMAL"},
    {"query": "Define transaction management.", "expected": "NORMAL"},
    {"query": "What is the difference between 1NF and 2NF?", "expected": "NORMAL"},
    {"query": "Describe ACID properties.", "expected": "NORMAL"},
    {"query": "What is SQL used for?", "expected": "NORMAL"},
    {"query": "Explain the role of a database administrator.", "expected": "NORMAL"},
    {"query": "What is a primary key?", "expected": "NORMAL"},
    {"query": "How does indexing improve query performance?", "expected": "NORMAL"},
    {"query": "What is data redundancy?", "expected": "NORMAL"},
    # --- HYBRID: exact-match tokens -----------------------------------------
    {"query": "What does regulation 12 say about attendance?", "expected": "HYBRID"},
    {"query": "What is CSIT 325 about?", "expected": "HYBRID"},
    {"query": "What happened on 2024-05-01?", "expected": "HYBRID"},
    {"query": "Which year past paper is 2080?", "expected": "HYBRID"},
    {"query": "What is question 5 about?", "expected": "HYBRID"},
    {"query": "How many marks is Q3 worth?", "expected": "HYBRID"},
    {"query": "Where can I find rule 8 of the exam regulations?", "expected": "HYBRID"},
    {"query": "What does CSIT 210 cover?", "expected": "HYBRID"},
    {"query": "Explain the notice dated 2023-12-15.", "expected": "HYBRID"},
    {"query": "Students must attend 75 percent of classes.", "expected": "HYBRID"},
    # --- GRAPH: structural relationships -------------------------------------
    {"query": "Which subjects are in semester 5?", "expected": "GRAPH"},
    {"query": "List the topics of Database Management System.", "expected": "GRAPH"},
    {"query": "What are the subtopics of Normalization?", "expected": "GRAPH"},
    {"query": "Which past questions are about Normalization?", "expected": "GRAPH"},
    {"query": "Which subjects contain the topic Normalization?", "expected": "GRAPH"},
    {"query": "What topics are included in DBMS?", "expected": "GRAPH"},
    {"query": "Find past questions related to SQL.", "expected": "GRAPH"},
    {"query": "Which courses are offered in semester 5 of BSc CSIT?", "expected": "GRAPH"},
    {"query": "List subjects in semester 6.", "expected": "GRAPH"},
    {"query": "What are the subtopics of Transaction Management?", "expected": "GRAPH"},
    # --- WEB: current / external information (Phase 11) ------------------------
    {"query": "What is the latest Python version?", "expected": "WEB"},
    {"query": "What happened in today's AI news?", "expected": "WEB"},
    {"query": "What is the current OpenAI API pricing?", "expected": "WEB"},
    {"query": "What is the current attendance requirement?", "expected": "WEB"},
    {"query": "What is the weather forecast for Kathmandu?", "expected": "WEB"},
    {"query": "Explain the latest AI research paper.", "expected": "WEB"},
]


def evaluate_router(router: AdaptiveRouter) -> dict:
    """Route every query in the eval set and compute routing metrics.

    Returns counts plus accuracy / false-routing / fallback rates.
    """
    results = [
        {"query": item["query"], "expected": item["expected"], "plan": router.analyze(item["query"])}
        for item in ROUTING_EVAL_SET
    ]
    total = len(results)
    correct = sum(1 for result in results if result["plan"].strategy.value == result["expected"])
    false_routing = sum(1 for result in results if result["plan"].strategy.value != result["expected"])
    fallback = sum(1 for result in results if result["plan"].fallback)
    return {
        "total": total,
        "correct": correct,
        "false_routing": false_routing,
        "fallback": fallback,
        "routing_accuracy": round(correct / total, 4) if total else 0.0,
        "false_routing_rate": round(false_routing / total, 4) if total else 0.0,
        "fallback_rate": round(fallback / total, 4) if total else 0.0,
        "results": results,
    }
