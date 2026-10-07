"""Document-type intent ranking tests.

Proves the syllabus boost is intent-aware (only fires for syllabus-style
queries), course-aware (the named course's syllabus wins), and inert for
ordinary content queries (vector/BM25/RRF order untouched).
"""

from __future__ import annotations

from app import document_intent as di
from app.retriever import RetrievedChunk

SE = "docs/software_engineering/"
CN = "docs/computer_networks/"
SE_SYLLABUS = "ENCT352SOFTWAREENGINEERING_2026_04_25_10_01_01.pdf"
CN_SYLLABUS = "ENCT355COMPUTERNETWORKS_2026_05_01_00_00_00.pdf"


def _chunk(
    document: str,
    source_path: str,
    *,
    score: float = 0.8,
    index: int = 0,
    bm25: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document}::p1::c{index}",
        text=f"content of {document}",
        document=document,
        page=1,
        score=score,
        chunk_index=index,
        semantic_score=score,
        bm25_score=bm25,
        source_path=source_path,
    )


# --------------------------------------------------------------------------- #
# Classification + intent detection
# --------------------------------------------------------------------------- #
def test_classify_document_by_filename_and_folder() -> None:
    assert di.classify_document(SE_SYLLABUS, SE + SE_SYLLABUS) == di.SYLLABUS
    assert di.classify_document("ch_1_slide.pdf", SE + "ch_1_slide.pdf") == di.SLIDE
    assert di.classify_document("Chapter 1.pdf", "docs/simulation/Chapter 1.pdf") == di.CHAPTER
    assert di.classify_document("past_question_software.pdf", SE + "past_question_software.pdf") == di.PAST_PAPER
    assert di.classify_document("holiday_notice_october_7.pdf", "docs/notices/holiday_notice_october_7.pdf") == di.NOTICE
    assert di.classify_document("university_rules_and_regulations.pdf", "docs/rules/university_rules.pdf") == di.RULES


def test_syllabus_intent_detects_outline_phrases() -> None:
    for query in (
        "syllabus of the software engineering",
        "software engineering syllabus",
        "show me the course outline",
        "what are the course contents",
        "curriculum of simulation",
        "course structure",
    ):
        assert di.detect_syllabus_intent(query) is True


def test_syllabus_intent_ignores_content_queries() -> None:
    for query in (
        "explain software testing in software engineering",
        "software engineering chapter 1",
        "what is a use case diagram",
        "compare verification and validation",
    ):
        assert di.detect_syllabus_intent(query) is False


# --------------------------------------------------------------------------- #
# Ranking behavior
# --------------------------------------------------------------------------- #
def test_syllabus_query_ranks_matching_syllabus_first() -> None:
    chunks = [
        _chunk("ch_1_slide.pdf", SE + "ch_1_slide.pdf"),
        _chunk("ch_3_slide.pdf", SE + "ch_3_slide.pdf"),
        _chunk(SE_SYLLABUS, SE + SE_SYLLABUS, index=2),
    ]
    out = di.apply_intent_ranking("syllabus of the software engineering", chunks)

    assert out[0].document == SE_SYLLABUS
    assert out[0].intent_boost == di.SYLLABUS_MATCH_BOOST
    assert [c.document for c in out[1:]] == ["ch_1_slide.pdf", "ch_3_slide.pdf"]


def test_software_engineering_syllabus_query_ranks_syllabus_first() -> None:
    chunks = [
        _chunk("ch_2_slide.pdf", SE + "ch_2_slide.pdf"),
        _chunk(SE_SYLLABUS, SE + SE_SYLLABUS, index=1),
    ]
    out = di.apply_intent_ranking("software engineering syllabus", chunks)
    assert out[0].document == SE_SYLLABUS


def test_content_query_leaves_ranking_untouched() -> None:
    chunks = [
        _chunk("ch_1_slide.pdf", SE + "ch_1_slide.pdf"),
        _chunk(SE_SYLLABUS, SE + SE_SYLLABUS, index=1),
    ]
    out = di.apply_intent_ranking("explain software testing in software engineering", chunks)

    # Same objects, same order, no boost: vector/BM25/RRF order is preserved.
    assert out is chunks
    assert all(c.intent_boost == 0.0 for c in out)


def test_chapter_query_does_not_prioritise_syllabus() -> None:
    chunks = [
        _chunk("ch_1_slide.pdf", SE + "ch_1_slide.pdf"),
        _chunk(SE_SYLLABUS, SE + SE_SYLLABUS, index=1),
    ]
    out = di.apply_intent_ranking("software engineering chapter 1", chunks)
    assert out is chunks
    assert out[0].document == "ch_1_slide.pdf"


def test_course_specific_syllabus_query_picks_matching_course() -> None:
    chunks = [
        _chunk(SE_SYLLABUS, SE + SE_SYLLABUS),
        _chunk(CN_SYLLABUS, CN + CN_SYLLABUS, index=1),
        _chunk("ch_1_slide.pdf", SE + "ch_1_slide.pdf", index=2),
    ]
    out = di.apply_intent_ranking("syllabus of computer networks", chunks)

    assert out[0].document == CN_SYLLABUS
    assert out[0].intent_boost == di.SYLLABUS_MATCH_BOOST
    # The Software Engineering syllabus must not receive the course-match boost.
    se_out = next(c for c in out if c.document == SE_SYLLABUS)
    assert se_out.intent_boost == 0.0


def test_unmatchable_course_query_does_not_boost_other_syllabus() -> None:
    chunks = [
        _chunk(SE_SYLLABUS, SE + SE_SYLLABUS),
        _chunk("ch_1_slide.pdf", SE + "ch_1_slide.pdf", index=1),
    ]
    out = di.apply_intent_ranking("syllabus of computer networks", chunks)

    assert out is chunks  # no matching course -> ranking unchanged
    assert all(c.intent_boost == 0.0 for c in out)


def test_generic_syllabus_query_boosts_any_syllabus() -> None:
    chunks = [
        _chunk("ch_1_slide.pdf", SE + "ch_1_slide.pdf"),
        _chunk(SE_SYLLABUS, SE + SE_SYLLABUS, index=1),
    ]
    out = di.apply_intent_ranking("give me the syllabus", chunks)

    assert out[0].document == SE_SYLLABUS
    assert out[0].intent_boost == di.SYLLABUS_GENERIC_BOOST


def test_boosted_chunk_keeps_its_retrieval_scores() -> None:
    syllabus = _chunk(SE_SYLLABUS, SE + SE_SYLLABUS, score=0.61, index=2, bm25=4.2)
    chunks = [
        _chunk("ch_1_slide.pdf", SE + "ch_1_slide.pdf"),
        _chunk("ch_3_slide.pdf", SE + "ch_3_slide.pdf", index=1),
        syllabus,
    ]
    out = di.apply_intent_ranking("software engineering syllabus", chunks)
    top = out[0]

    assert top.chunk_id == syllabus.chunk_id
    assert top.semantic_score == 0.61
    assert top.bm25_score == 4.2
    assert top.page == syllabus.page


def test_empty_retrieval_is_a_noop() -> None:
    assert di.apply_intent_ranking("syllabus of the software engineering", []) == []


# --------------------------------------------------------------------------- #
# Pipeline integration: the citations returned to the API follow the boost
# --------------------------------------------------------------------------- #
def test_pipeline_reorders_citations_for_syllabus_query() -> None:
    from app.config import Settings
    from app.rag import answer_question
    from app.routing import QueryRouter
    from tests.conftest import FakeLLM, FakeRetriever, make_retrieved

    settings = Settings(
        qdrant_url="http://fake",
        qdrant_collection="test_docs",
        embedding_dim=64,
        retrieval_top_k=4,
        min_relevance_score=0.0,
        max_context_chars=4000,
        _env_file=None,
    )
    llm = FakeLLM(text="The course objectives and assessment scheme are [1].")
    hybrid = FakeRetriever(
        [
            make_retrieved(
                "lecture slide content", document="ch_1_slide.pdf", score=0.9,
                source_path=SE + "ch_1_slide.pdf",
            ),
            make_retrieved(
                "course objectives and assessment scheme", document=SE_SYLLABUS,
                score=0.7, chunk_id="se-syllabus", source_path=SE + SE_SYLLABUS,
            ),
        ]
    )

    result = answer_question(
        "syllabus of the software engineering",
        settings=settings,
        retriever=FakeRetriever([]),
        llm=llm,
        router=QueryRouter(),
        hybrid_retriever=hybrid,
    )

    assert result.strategy == "hybrid"
    assert result.sources[0].document == SE_SYLLABUS
