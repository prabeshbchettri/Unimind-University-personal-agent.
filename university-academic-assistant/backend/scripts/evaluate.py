"""Phase 13 evaluation runner.

Runs every evaluation against the hermetic offline environment and prints a
complete report: router accuracy, retrieval metrics (Recall@K / Precision@K
/ MRR per strategy), answer metrics (faithfulness / relevance / source
correctness / hallucination rate), OCR quality (digital + scanned), and the
library recommendation scenario checks.

Run (from the backend directory):

    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.answers import evaluate_answers  # noqa: E402
from app.evaluation.corpus import RETRIEVAL_QUERIES  # noqa: E402
from app.evaluation.environment import build_environment  # noqa: E402
from app.evaluation.ocr import evaluate_ocr  # noqa: E402
from app.evaluation.recommendations import evaluate_recommendations  # noqa: E402
from app.evaluation.retrieval import evaluate_retrieval  # noqa: E402
from app.router.eval_set import evaluate_router  # noqa: E402


def _line(char: str = "=") -> None:
    print(char * 72)


def _table(rows: list[tuple[str, str, str]]) -> None:
    for left, center, right in rows:
        print(f"{left:<34} {center:<22} {right}")


def main() -> None:
    env = build_environment()

    _line()
    print("PHASE 13 - FULL SYSTEM EVALUATION (hermetic, deterministic)")
    print("backend: in-memory Qdrant | deterministic embedder | stub LLM | in-memory graph | BM25")
    _line()

    # --- 1. Router evaluation ------------------------------------------------
    router_metrics = evaluate_router(env.router)
    print("\n1) ROUTER EVALUATION")
    print(f"   eval set            : {router_metrics['total']} labeled queries "
          f"(NORMAL/HYBRID/GRAPH/WEB)")
    print(f"   routing accuracy    : {router_metrics['routing_accuracy']:.2%}")
    print(f"   false routing rate  : {router_metrics['false_routing_rate']:.2%}")
    print(f"   fallback rate       : {router_metrics['fallback_rate']:.2%}")
    for result in router_metrics["results"]:
        if result["plan"].strategy.value != result["expected"]:
            print(f"   MISROUTE: expected={result['expected']:<6} got={result['plan'].strategy.value:<6} query={result['query']}")

    # --- 2. Retrieval evaluation ---------------------------------------------
    print("\n2) RETRIEVAL EVALUATION (document-level, labeled corpus)")
    print(f"   corpus queries      : {len(RETRIEVAL_QUERIES)} across 6 categories")
    print("   each strategy scored on its labeled queries only; empty-rate on")
    print("   other queries guards against out-of-scope results")
    _table([("strategy", "recall@1/3/5", "precision@3/5   mrr   empty-other")])
    retrieval = evaluate_retrieval(env)
    for name, metrics in retrieval["strategies"].items():
        print(
            f"{name:<34} {metrics['recall@1']:.2f}/{metrics['recall@3']:.2f}/{metrics['recall@5']:.2f}"
            f"{'':>6} {metrics['precision@3']:.2f}/{metrics['precision@5']:.2f}   {metrics['mrr']:.2f}   {metrics['empty_rate_on_other_queries']:.2f}"
        )

    # --- 3. Answer evaluation ------------------------------------------------
    answers = evaluate_answers(env)
    print("\n3) ANSWER EVALUATION (grounded stub generator; methodology applies to Ollama)")
    print(f"   queries evaluated   : {answers['count']}")
    print(f"   mean faithfulness   : {answers['mean_faithfulness']:.2%}")
    print(f"   relevance (evidence): {answers['relevance']:.2%}")
    print(f"   source correctness  : {answers['source_correctness']:.2%}")
    print(f"   hallucination rate  : {answers['hallucination_rate']:.2%}")

    # --- 4. OCR evaluation ----------------------------------------------------
    ocr = evaluate_ocr(env)
    print("\n4) OCR EVALUATION (digital vs scanned)")
    print(f"   tesseract available : {ocr['tesseract_available']}")
    for name, scenario in ocr["scenarios"].items():
        if scenario.get("status") == "skipped":
            print(f"   {name:<14}: skipped - {scenario['reason']}")
        else:
            print(
                f"   {name:<14}: method={scenario['extraction_method']:<8} "
                f"char_accuracy={scenario['character_accuracy']:.2%} word_recall={scenario['word_recall']:.2%}"
            )

    # --- 5. Recommendation evaluation -----------------------------------------
    recs = evaluate_recommendations(env)
    print("\n5) LIBRARY RECOMMENDATION EVALUATION")
    print(f"   formula             : {recs['formula']}")
    for name, scenario in recs["scenarios"].items():
        top_books = ", ".join(f"{b['book']} ({b['score']:.2f})" for b in scenario["books"][:3])
        print(f"   {name:<24}: {top_books or '(none)'}")
    print(f"   checks passed       : {recs['checks_passed']} ({sum(recs['checks'].values())}/{len(recs['checks'])})")
    for check, passed in recs["checks"].items():
        if not passed:
            print(f"   FAILED CHECK: {check}")

    _line()
    print("Evaluation complete. See docs/EVALUATION.md for the methodology.")
    _line()


if __name__ == "__main__":
    main()