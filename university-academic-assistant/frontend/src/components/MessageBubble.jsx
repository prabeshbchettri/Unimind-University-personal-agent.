import SourceCard from "./SourceCard";
import RecommendationCard from "./RecommendationCard";

// One chat message: a user question or the assistant reply. Assistant replies
// may carry retrieved sources, the routing strategy, book recommendations, or
// a friendly error banner when something failed.
export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const isTyping = message.typing;

  return (
    <div className={`message ${isUser ? "message--user" : "message--assistant"}`}>
      <div className="message__bubble">
        {isTyping ? (
          <span className="message__typing" aria-label="Assistant is typing">
            <span />
            <span />
            <span />
          </span>
        ) : isUser ? (
          message.content
        ) : (
          <>
            <div className="message__content">{message.content}</div>

            {message.error && <div className="message__error">{message.error}</div>}

            {message.strategy && (
              <span className="message__strategy">
                Strategy: {message.strategy}
                {message.fallback ? " (fallback)" : ""}
              </span>
            )}

            {message.recommendations && (
              <section className="message__recommendations">
                {message.recommendations.explanation && (
                  <p className="message__rec-explanation">
                    {message.recommendations.explanation}
                  </p>
                )}
                <div className="message__rec-list">
                  {message.recommendations.results.map((recommendation, index) => (
                    <RecommendationCard
                      key={`${recommendation.book}-${index}`}
                      recommendation={recommendation}
                    />
                  ))}
                </div>
              </section>
            )}

            {message.sources && message.sources.length > 0 && (
              <section className="message__sources">
                <h4 className="message__sources-title">Sources ({message.sources.length})</h4>
                <div className="message__sources-list">
                  {message.sources.map((source, index) => (
                    <SourceCard key={`${source.metadata?.document_id || source.url}-${index}`} source={source} />
                  ))}
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}
