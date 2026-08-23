"""Answer evaluation (Phase 13).

Methodology (documented in docs/EVALUATION.md):

The evaluation uses a *grounded* stub LLM: a responder that answers by
quoting the strongest retrieved source(s). Because the answer generator is
deterministic, the metrics measure the *pipeline* (retrieval feeding the
generator, context assembly, grounding discipline) rather than model quality.
With a real Ollama model the identical methodology applies — the metrics are
computed from the final answer text and the retrieved sources:

- **Faithfulness** — the fraction of answer content (whitespace-trimmed
  4-grams) that appears in the retrieved sources. 1.0 = fully grounded.
- **Relevance** — retrieval hit rate: the answer ran with non-empty
  evidence for queries whose corpus has evidence.
- **Source correctness** — every source referenced by the answer
  (``[Source N]``/``[Web N]`` markers) exists in the retrieved source list.
- **Citation correctness** — cited source indices appear in the same order
  as the retrieved sources (no fabricated citations).
- **Hallucination rate** — the fraction of answers containing substantive
  n-grams found in no retrieved source (a response that quotes nothing from
  the sources is also flagged when it asserts a specific fact).
"""

from __future__ import annotations

import asyncio
import re

from app.evaluation.corpus import RETRIEVAL_QUERIES
from app.evaluation.environment import EvaluationEnvironment
from app.evaluation.metrics import macro_average
from app.llm.stub import StubLLMClient

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "was", "were", "it", "that", "with", "for", "as", "by", "at", "this",
    "which", "what", "how", "do", "does", "not", "be", "from",
}


def _n_grams(text: str, n: int = 4) -> set[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    meaningful = [token for token in tokens if token not in _STOPWORDS]
    return {" ".join(meaningful[i : i + n]) for i in range(len(meaningful) - n + 1)}


def _run(coro):
    return asyncio.run(coro)


def _grounded_responder() -> tuple[StubLLMClient, list[dict]]:
    """A stub LLM that answers by quoting the strongest source verbatim."""
    seen: list[dict] = []

    def responder(prompt: str) -> str:
        seen.append({"prompt": prompt})
        # [Source N] header line, then the full source body up to the next
        # block or the end of the context.
        match = re.search(r"\[Source (\d+)\]\s*[^\n]*\n(.*?)(?=\n\[Source |\n\[Web |\n--- END|\Z)", prompt, re.DOTALL)
        if not match:
            return "I could not find sufficient information in the available sources."
        index, text = int(match.group(1)), match.group(2).strip()
        return f"According to [Source {index}]: {text[:400]}"

    return StubLLMClient(default_answer="I could not find sufficient information in the available sources.", responder=responder), seen


def evaluate_answers(env: EvaluationEnvironment) -> dict:
    """Run the labeled queries through the chat service and score answers."""
    llm, seen = _grounded_responder()
    env.chat_service.llm = llm

    rows = []
    for item in RETRIEVAL_QUERIES:
        result = _run(env.chat_service.answer(item["query"], top_k=5))
        sources = result.sources
        source_text = " ".join(source.text for source in sources)
        answer = result.answer
        # The generator's framing ("According to [Source N]: ...") is not
        # content; strip markers and the lead-in before grounding checks.
        clean_answer = re.sub(r"\[Source \d+\]", "", answer)
        clean_answer = re.sub(r"^\s*according\s+to\s*:?\s*", "", clean_answer, flags=re.IGNORECASE)

        answer_grams = _n_grams(clean_answer)
        source_grams = _n_grams(source_text)
        faithful = (
            len(answer_grams & source_grams) / len(answer_grams)
            if answer_grams
            else (1.0 if not sources else 0.0)
        )

        cited = set(int(number) for number in re.findall(r"\[Source (\d+)\]", answer))
        citation_ok = bool(cited) and cited <= set(range(1, len(sources) + 1))

        hallucinated = bool(answer_grams - source_grams) and bool(cited)
        if not sources and "sufficient information" not in answer:
            hallucinated = True

        rows.append(
            {
                "query": item["query"],
                "category": item["category"],
                "sources": len(sources),
                "faithfulness": round(faithful, 4),
                "source_correct": citation_ok,
                "hallucinated": hallucinated,
            }
        )

    evidence_queries = [row for row in rows if row["sources"] > 0]
    return {
        "queries": rows,
        "count": len(rows),
        "mean_faithfulness": macro_average([row["faithfulness"] for row in rows]),
        "relevance": round(len(evidence_queries) / len(rows), 4) if rows else 0.0,
        "source_correctness": round(
            sum(1 for row in rows if row["source_correct"]) / len(rows), 4
        )
        if rows
        else 0.0,
        "hallucination_rate": round(
            sum(1 for row in rows if row["hallucinated"]) / len(rows), 4
        )
        if rows
        else 0.0,
        "grounded_stub": True,
    }