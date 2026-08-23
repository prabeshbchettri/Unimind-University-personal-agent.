"""Recommendation endpoints (Phase 9)."""

from fastapi import APIRouter, Request

from app.schemas.recommendations import (
    BookRecommendationSchema,
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse, summary="Recommend library books for a query")
async def recommend(request: Request, body: RecommendationRequest) -> RecommendationResponse:
    """Recommend library books ranked by syllabus/topic coverage.

    The coverage score is computed from the indexed content (matched required
    topics / required topics, weighted with subtopic coverage) and each book
    reports its matched and missing topics as evidence.
    """
    result = await request.app.state.recommendation_service.recommend(body.query)
    return RecommendationResponse(
        query=result.query,
        subject=result.subject,
        required_topics=result.required_topics,
        required_subtopics=result.required_subtopics,
        recommendations=[
            BookRecommendationSchema(
                book=coverage.book_title,
                author=coverage.author,
                score=coverage.score,
                matched_topics=coverage.matched_topics,
                missing_topics=coverage.missing_topics,
                matched_subtopics=coverage.matched_subtopics,
                missing_subtopics=coverage.missing_subtopics,
                topic_coverage=coverage.topic_coverage,
                subtopic_coverage=coverage.subtopic_coverage,
                chapter_count=coverage.chapter_count,
                evidence=coverage.evidence,
            )
            for coverage in result.recommendations
        ],
        explanation=result.explanation,
        no_syllabus_evidence=result.no_syllabus_evidence,
    )