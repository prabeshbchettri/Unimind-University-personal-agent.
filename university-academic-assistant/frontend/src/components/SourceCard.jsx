import { formatDateTime, documentTypeLabel } from "../utils/format";

// One retrieved source. University sources come from the RAG pipeline with
// provenance metadata; web sources are live search results with a URL.
export default function SourceCard({ source }) {
  const meta = source.metadata || {};
  const isWeb = source.kind === "web";
  const label = documentTypeLabel(meta.document_type);

  return (
    <article className="source-card">
      <header className="source-card__head">
        <span className={`source-card__kind ${isWeb ? "source-card__kind--web" : ""}`}>
          {isWeb ? "Web" : label}
        </span>
        <span className="source-card__score">{Math.round((source.score || 0) * 100)}%</span>
      </header>

      <h4 className="source-card__title">{meta.title || source.text?.slice(0, 80) || "Source"}</h4>

      {isWeb ? (
        <a
          className="source-card__link"
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
        >
          {source.url}
        </a>
      ) : (
        <p className="source-card__meta">
          {meta.page != null && `Page ${meta.page}`}
          {meta.page != null && meta.topic ? " · " : ""}
          {meta.topic || ""}
        </p>
      )}

      <p className="source-card__text">{source.text}</p>

      <footer className="source-card__foot">
        {isWeb
          ? source.retrieved_at && `Retrieved ${formatDateTime(source.retrieved_at)}`
          : meta.document_id || ""}
      </footer>
    </article>
  );
}
