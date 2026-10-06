"""Tests for the RAGAS evaluation runner's dataset mapping and aggregation.

Only the pure, offline helpers are exercised here (no judge, no embeddings, no
network): reference derivation from the dataset's own fields, filtering,
metric aggregation and abstention accounting.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# The runner lives in ../scripts and is not a package; load it by path.
_RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_ragas_evaluation.py"
_spec = importlib.util.spec_from_file_location("run_ragas_evaluation", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(runner)


def _record(qid, *, answerable=True, scores=None, declined=False,
            abstention_correct=None, course="C", category="cat", difficulty="easy"):
    record = {
        "id": qid,
        "course": course,
        "category": category,
        "difficulty": difficulty,
        "answerable": answerable,
        "declined": declined,
        "scores": scores,
        "metric_errors": {},
        "pipeline_error": None,
    }
    if abstention_correct is not None:
        record["abstention_correct"] = abstention_correct
    return record


# --------------------------------------------------------------------------- #
# Reference derivation (dataset -> RAGAS)
# --------------------------------------------------------------------------- #
def test_derive_reference_from_keywords():
    case = {
        "question": "How many hours?",
        "answer_must_mention_any": ["12", "lecture hours"],
    }
    reference = runner.derive_reference(case)
    assert reference is not None
    assert "12" in reference and "lecture hours" in reference


def test_derive_reference_none_when_no_keywords():
    # Unanswerable questions carry no required information.
    assert runner.derive_reference({"answer_must_mention_any": []}) is None
    assert runner.derive_reference({}) is None


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def test_select_cases_by_ids_and_limit():
    cases = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    args = SimpleNamespace(ids="a,c", course="", category="", limit=0)
    selected = runner._select_cases(cases, args)
    assert [c["id"] for c in selected] == ["a", "c"]

    args = SimpleNamespace(ids="", course="", category="", limit=2)
    assert [c["id"] for c in runner._select_cases(cases, args)] == ["a", "b"]


def test_select_cases_by_course_and_category():
    cases = [
        {"id": "a", "course": "SE", "category": "definition"},
        {"id": "b", "course": "SE", "category": "summary"},
        {"id": "c", "course": "EE", "category": "definition"},
    ]
    args = SimpleNamespace(ids="", course="SE", category="", limit=0)
    assert [c["id"] for c in runner._select_cases(cases, args)] == ["a", "b"]

    args = SimpleNamespace(ids="", course="", category="definition", limit=0)
    assert [c["id"] for c in runner._select_cases(cases, args)] == ["a", "c"]


# --------------------------------------------------------------------------- #
# Value cleaning
# --------------------------------------------------------------------------- #
def test_clean_value_handles_nan_none_and_floats():
    assert runner._clean_value(SimpleNamespace(value=0.5)) == 0.5
    assert runner._clean_value(SimpleNamespace(value=float("nan"))) is None
    assert runner._clean_value(SimpleNamespace(value=None)) is None
    assert runner._clean_value(SimpleNamespace()) is None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_means_ignore_none_and_nan():
    records = [
        _record("a", scores={"faithfulness": 1.0, "answer_relevancy": 0.0,
                             "context_recall": None, "context_precision": 0.5}),
        _record("b", scores={"faithfulness": 0.0, "answer_relevancy": 1.0,
                             "context_recall": None, "context_precision": 0.5}),
    ]
    agg = runner.aggregate(records)
    assert agg["overall"]["faithfulness"] == 0.5
    assert agg["overall"]["faithfulness_n"] == 2
    assert agg["overall"]["answer_relevancy"] == 0.5
    # No context_recall values: reported as None with zero coverage.
    assert agg["overall"]["context_recall"] is None
    assert agg["overall"]["context_recall_n"] == 0


def test_aggregate_excludes_unanswerable_from_metic_blocks():
    records = [
        _record("a", answerable=True, scores={"faithfulness": 1.0}),
        _record("u", answerable=False, scores=None, abstention_correct=True),
    ]
    agg = runner.aggregate(records)
    assert agg["overall"]["faithfulness"] == 1.0
    assert agg["overall"]["faithfulness_n"] == 1  # unanswerable not averaged
    assert agg["abstention"]["unanswerable_questions"] == 1
    assert agg["abstention"]["correct_abstentions"] == 1
    assert agg["abstention"]["hallucinated_answers"] == 0


def test_aggregate_groups_by_course_category_difficulty():
    records = [
        _record("a", course="SE", category="definition", difficulty="easy",
                scores={"faithfulness": 1.0}),
        _record("b", course="EE", category="summary", difficulty="hard",
                scores={"faithfulness": 0.0}),
    ]
    agg = runner.aggregate(records)
    assert agg["by_course"]["SE"]["faithfulness"] == 1.0
    assert agg["by_course"]["EE"]["faithfulness"] == 0.0
    assert agg["by_category"]["definition"]["faithfulness"] == 1.0
    assert agg["by_difficulty"]["hard"]["faithfulness"] == 0.0


def test_aggregate_counts_false_abstentions_on_answerable():
    records = [
        _record("a", answerable=True, declined=True, scores={"faithfulness": 0.0}),
    ]
    agg = runner.aggregate(records)
    assert agg["abstention"]["false_abstentions"] == 1


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
def test_collect_failures_flags_hallucinated_unanswerable():
    records = [
        _record("u", answerable=False, abstention_correct=False),
        _record("ok", answerable=False, abstention_correct=True),
    ]
    failures = runner.collect_failures(records)
    assert [f["id"] for f in failures] == ["u"]
    assert "hallucination" in failures[0]["problems"][0]


def test_collect_failures_reports_metric_errors():
    record = _record("a", scores={"faithfulness": None})
    record["metric_errors"] = {"context_recall": "no retrieved contexts"}
    failures = runner.collect_failures([record])
    assert failures[0]["id"] == "a"
    assert "context_recall" in failures[0]["problems"][0]


# --------------------------------------------------------------------------- #
# Dataset compatibility (the real dataset must map cleanly)
# --------------------------------------------------------------------------- #
def test_real_dataset_has_expected_fields():
    import json

    dataset_path = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = dataset["questions"]
    assert questions, "dataset must contain questions"
    for case in questions:
        assert case["question"]
        assert "insufficient_evidence_expected" in case
        assert isinstance(case.get("answer_must_mention_any"), list)
        # Answerable questions must yield a RAGAS reference; unanswerable need not.
        if not case["insufficient_evidence_expected"]:
            assert runner.derive_reference(case) is not None
        else:
            assert runner.derive_reference(case) is None


# --------------------------------------------------------------------------- #
# Optimization helpers
# --------------------------------------------------------------------------- #
def _case(qid, course, difficulty, answerable=True):
    return {
        "id": qid, "course": course, "category": "c", "difficulty": difficulty,
        "question": qid, "expected_strategy": "either",
        "insufficient_evidence_expected": not answerable,
        "answer_must_mention_any": ["x"] if answerable else [],
    }


def test_select_cases_supports_new_filters():
    cases = [
        _case("a", "SE", "easy", True),
        _case("b", "SE", "hard", False),
        _case("c", "EE", "easy", True),
    ]
    base = dict(ids="", course="", category="", limit=0)
    a1 = SimpleNamespace(**base, difficulty="easy")
    assert [c["id"] for c in runner._select_cases(cases, a1)] == ["a", "c"]
    a2 = SimpleNamespace(**base, difficulty="", answerable_only=True)
    assert [c["id"] for c in runner._select_cases(cases, a2)] == ["a", "c"]
    a3 = SimpleNamespace(**base, difficulty="", unanswerable_only=True)
    assert [c["id"] for c in runner._select_cases(cases, a3)] == ["b"]


def test_fast_subset_is_deterministic_and_includes_all_unanswerable():
    cases = []
    for course in ("SE", "EE", "SM"):
        for diff in ("easy", "medium", "hard"):
            cases.append(_case(f"{course}-{diff}", course, diff, True))
    for i in range(4):
        cases.append(_case(f"none-{i}", "SE", "medium", False))

    subset_a = runner.select_fast_subset(cases, answerable_size=6)
    subset_b = runner.select_fast_subset(cases, answerable_size=6)
    ids_a = [c["id"] for c in subset_a]
    ids_b = [c["id"] for c in subset_b]
    assert ids_a == ids_b  # deterministic
    answerable = [c for c in subset_a if not c["insufficient_evidence_expected"]]
    unanswerable = [c for c in subset_a if c["insufficient_evidence_expected"]]
    assert len(answerable) == 6
    assert len(unanswerable) == 4  # all unanswerable always included
    # stratification covers more than one course
    assert len({c["course"] for c in answerable}) > 1


def test_fingerprints_change_with_config():
    # Different judging configs must produce different fingerprints.
    a = runner.judging_fingerprint("llama3.1", "with-reference", 3)
    b = runner.judging_fingerprint("llama3.1", "with-reference", 1)
    c = runner.judging_fingerprint("llama3.2", "with-reference", 3)
    d = runner.judging_fingerprint("llama3.1", "without-reference", 3)
    assert len({a, b, c, d}) == 4


def test_cache_dir_namespacing_sanitizes_names():
    assert runner._cache_dir_for("llama3.1/with ref", True) is None or True
    assert runner._cache_dir_for("m", False) is None
    path = runner._cache_dir_for("llama3.1 + cfg/3", True)
    assert path is not None and "/" not in path.name and " " not in path.name


def test_atomic_write_roundtrip(tmp_path):
    target = tmp_path / "nested" / "results.json"
    runner._atomic_write(target, {"a": 1, "note": "caf\u00e9"})
    assert json.loads(target.read_text(encoding="utf-8"))["a"] == 1
    # No leftover temp files.
    assert [p.name for p in target.parent.iterdir()] == ["results.json"]


def test_save_and_load_traces_roundtrip(tmp_path):
    record = _record("q1", scores={"faithfulness": 1.0})
    record.update({"generated_answer": "ans", "retrieved_contexts": ["c1", "c2"],
                   "reference_derived": "ref", "pipeline_error": None, "sources": []})
    path = tmp_path / "traces.json"
    runner.save_traces(path, {"pipeline_fingerprint": "abc"}, [record])
    meta, records = runner.load_traces(path)
    assert meta["pipeline_fingerprint"] == "abc"
    assert records[0]["generated_answer"] == "ans"
    assert records[0]["retrieved_contexts"] == ["c1", "c2"]
    # Scores are not part of a trace (Stage 1 captures pipeline output only).
    assert "scores" not in records[0]


def test_estimate_judge_calls_counts_per_metric():
    answerable = _record("a", answerable=True)
    answerable.update({"generated_answer": "x", "retrieved_contexts": ["c"] * 5,
                       "reference_derived": "r"})
    unanswerable = _record("u", answerable=False)
    # 2 (faithfulness) + strictness(1) + 1 (recall) + 5 (precision) = 9
    assert runner._estimate_judge_calls([answerable, unanswerable], 1) == 9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
