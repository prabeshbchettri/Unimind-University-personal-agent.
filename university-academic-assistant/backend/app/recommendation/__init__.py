"""Book recommendation package.

Phase 9: evidence-based library book recommendation. Books are recommended by
how well their indexed chapter/subtopic content covers the university
syllabus topics required for the user query.

Imports stay lazy (only lightweight modules here) to avoid import cycles:
``TopicResolver`` depends on the Qdrant repository, so it is imported by the
service directly.
"""

from app.recommendation.coverage import CoverageCalculator, normalize_name, rank
from app.recommendation.models import BookCoverage, BookProfile, RecommendationResult

__all__ = [
    "BookCoverage",
    "BookProfile",
    "CoverageCalculator",
    "normalize_name",
    "rank",
    "RecommendationResult",
]