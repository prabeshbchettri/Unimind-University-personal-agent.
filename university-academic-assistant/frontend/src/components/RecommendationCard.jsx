import { formatPercent } from "../utils/format";

// One book recommendation with its coverage score, matched/missing topics and
// the evidence (sources) that backs the score.
export default function RecommendationCard({ recommendation }) {
  const {
    book,
    author,
    score,
    matched_topics = [],
    missing_topics = [],
    matched_subtopics = [],
    missing_subtopics = [],
    chapter_count,
    evidence = [],
  } = recommendation;

  return (
    <article className="rec-card">
      <header className="rec-card__head">
        <div className="rec-card__titles">
          <h4 className="rec-card__title">{book}</h4>
          {author && <p className="rec-card__author">{author}</p>}
        </div>
        <span className="rec-card__score" title="Syllabus coverage score">
          {formatPercent(score)}
        </span>
      </header>

      <div className="rec-card__badges">
        {matched_topics.slice(0, 6).map((topic) => (
          <span key={`m-${topic}`} className="rec-card__badge rec-card__badge--matched" title="Covered">
            {topic}
          </span>
        ))}
        {missing_topics.slice(0, 4).map((topic) => (
          <span key={`x-${topic}`} className="rec-card__badge rec-card__badge--missing" title="Not covered">
            {topic}
          </span>
        ))}
      </div>

      <ul className="rec-card__detail">
        <li>Matched subtopics: {matched_subtopics.length}</li>
        <li>Missing subtopics: {missing_subtopics.length}</li>
        {chapter_count != null && <li>Chapters indexed: {chapter_count}</li>}
      </ul>

      {evidence.length > 0 && (
        <p className="rec-card__evidence">
          <strong>Based on:</strong> {evidence.join(", ")}
        </p>
      )}
    </article>
  );
}
