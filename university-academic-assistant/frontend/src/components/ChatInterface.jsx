import { useState } from "react";
import MessageBubble from "./MessageBubble";

const SUGGESTIONS = [
  "What is the attendance requirement?",
  "Find the latest Python release and its pricing",
  "Recommend books for COE 301",
  "What are the university holidays in the academic calendar?",
];

export default function ChatInterface({
  messages,
  pending,
  mode,
  onSend,
  onModeChange,
}) {
  const [input, setInput] = useState("");

  function submit() {
    const text = input.trim();
    if (!text || pending) return;
    setInput("");
    onSend(text);
  }

  return (
    <main className="chat">
      {messages.length === 0 ? (
        <section className="chat__welcome">
          <h1>University Academic Assistant</h1>
          <p>
            Ask about attendance, regulation, academic calendar, syllabus topics,
            past questions — or get book recommendations and live web results.
          </p>
          <div className="chat__suggestions">
            {SUGGESTIONS.map((suggestion) => (
              <button
                type="button"
                key={suggestion}
                className="chat__suggestion"
                onClick={() => {
                  setInput(suggestion);
                }}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </section>
      ) : (
        <section className="chat__messages" aria-live="polite">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </section>
      )}

      <div className="chat__composer">
        <div className="chat__composer-mode">
          <button
            type="button"
            className={`chat__mode ${mode === "recommend" ? "chat__mode--active" : ""}`}
            onClick={() => onModeChange(mode === "recommend" ? "chat" : "recommend")}
          >
            {mode === "recommend" ? "Book recommendations: on" : "Book recommendations: off"}
          </button>
        </div>
        <div className="chat__composer-row">
          <textarea
            className="chat__input"
            placeholder={
              mode === "recommend"
                ? "Which books do you recommend for…"
                : "Ask about attendance, regulation, syllabus, past questions, the web…"
            }
            rows={1}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
          />
          <button
            type="button"
            className="chat__send"
            disabled={!input.trim() || pending}
            onClick={submit}
          >
            Send
          </button>
        </div>
        <p className="chat__hint">
          Enter to send, Shift+Enter for a new line. Sources and recommendations
          appear under each answer.
        </p>
      </div>
    </main>
  );
}
