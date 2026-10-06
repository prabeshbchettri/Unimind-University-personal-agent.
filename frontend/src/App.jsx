import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import Markdown from './api-markdown.jsx'
import { ApiError, checkHealth, sendChatMessageStream } from './api.js'

/** Short strategy label shown next to the full reason. */
function strategyLabel(strategy) {
  return strategy ? strategy.toUpperCase() : '—'
}

/** "groq" -> "Groq", "ollama_fallback" -> "Ollama fallback". */
function providerLabel(provider) {
  if (!provider || provider === 'none') return '—'
  const [base, modifier] = provider.split('_')
  const label = base.charAt(0).toUpperCase() + base.slice(1)
  return modifier === 'fallback' ? `${label} fallback` : label
}

/** Small inline SVG mark reusing the existing app favicon. */
function UniMindMark() {
  return (
    <svg className="unimind-mark" viewBox="0 0 48 46" aria-hidden="true" focusable="false">
      <path
        fill="currentColor"
        d="M25.946 44.938c-.664.845-2.021.375-2.021-.698V33.937a2.26 2.26 0 0 0-2.262-2.262H10.287c-.92 0-1.456-1.04-.92-1.788l7.48-10.471c1.07-1.497 0-3.578-1.842-3.578H1.237c-.92 0-1.456-1.04-.92-1.788L10.013.474c.214-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.471c-1.07 1.498 0 3.579 1.842 3.579h11.377c.943 0 1.473 1.088.89 1.83L25.947 44.94z"
      />
    </svg>
  )
}

/** Strategy, provider and sources of one answer -- exactly what the backend reported. */
function AnswerMeta({ meta }) {
  return (
    <dl className="metadata">
      <div>
        <dt>Strategy</dt>
        <dd title={meta.strategyReason || undefined}>{strategyLabel(meta.strategy)}</dd>
      </div>
      <div>
        <dt>Provider</dt>
        <dd>{providerLabel(meta.provider)}</dd>
      </div>
      <div>
        <dt>Sources</dt>
        <dd>
          {meta.sources.length === 0 ? (
            <span className="muted">None</span>
          ) : (
            <ul className="sources">
              {meta.sources.map((source) => (
                <li key={`${source.document}-${source.page}`}>
                  {source.document} · Page {source.page}
                </li>
              ))}
            </ul>
          )}
        </dd>
      </div>
    </dl>
  )
}

/** One left- or right-aligned conversation message. */
function Message({ entry }) {
  const isUser = entry.role === 'user'
  return (
    <div className={`turn ${isUser ? 'turn-user' : 'turn-assistant'}`}>
      {isUser ? (
        <div className="message message-user">
          <p>{entry.question}</p>
        </div>
      ) : (
        <>
          <div className="assistant-identity">
            <UniMindMark />
            <span className="assistant-name">UniMind</span>
          </div>
          <div className="message message-assistant">
            {entry.answer === null && !entry.error && !entry.stopped && (
              <p className="streaming-note muted">
                <span className="thinking-dot" aria-hidden="true" /> UniMind is thinking…
              </p>
            )}
            {entry.error && <p className="error-text">{entry.error}</p>}
            {entry.answer !== null && entry.answer !== '' && (
              <>
                <Markdown text={entry.answer} />
                {entry.streaming && <span className="stream-cursor" aria-hidden="true" />}
              </>
            )}
            {entry.stopped && !entry.error && (
              <p className="muted stopped-note">Generation stopped.</p>
            )}
            {entry.meta && !entry.streaming && <AnswerMeta meta={entry.meta} />}
          </div>
        </>
      )}
    </div>
  )
}

/** Welcome screen with clickable example prompts. */
function EmptyState({ onPick }) {
  const suggestions = [
    'Explain Monte Carlo simulation',
    'What is Chapter 1 of Simulation about?',
    'Summarize Chapter 2 for my exam',
    'What are the main topics in the Simulation syllabus?',
  ]
  return (
    <div className="empty-state">
      <h2>Welcome to UniMind</h2>
      <p className="empty-subtitle">How can I help you study today?</p>
      <p className="empty-hint muted">
        Ask questions about your courses, notes, syllabus, and university study materials.
      </p>
      <div className="suggestions" role="list">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            className="suggestion"
            role="listitem"
            onClick={() => onPick(suggestion)}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [entries, setEntries] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [backendOnline, setBackendOnline] = useState(null) // null = checking
  const scrollRef = useRef(null)
  const stickToBottomRef = useRef(true)
  const abortRef = useRef(null)

  // Real health check -- no static "Connected" badge.
  useEffect(() => {
    let active = true
    const probe = async () => {
      const online = await checkHealth()
      if (active) setBackendOnline(online)
    }
    probe()
    const interval = setInterval(probe, 15_000)
    return () => {
      active = false
      clearInterval(interval)
    }
  }, [])

  // Follow the stream only while the user is near the bottom; never yank the
  // view if they scrolled up to read.
  useEffect(() => {
    if (stickToBottomRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [entries])

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }, [])

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const ask = useCallback(async (question) => {
    const trimmed = question.trim()
    if (!trimmed || abortRef.current) return

    const userEntry = { id: crypto.randomUUID(), role: 'user', question: trimmed }
    const answerEntry = {
      id: crypto.randomUUID(),
      role: 'assistant',
      answer: null,
      meta: null,
      error: '',
      streaming: true,
      stopped: false,
    }
    setEntries((current) => [...current, userEntry, answerEntry])
    setInput('')
    setStreaming(true)
    stickToBottomRef.current = true

    const controller = new AbortController()
    abortRef.current = controller
    const patch = (update) =>
      setEntries((current) =>
        current.map((entry) => (entry.id === answerEntry.id ? { ...entry, ...update } : entry)),
      )

    try {
      const final = await sendChatMessageStream(trimmed, {
        signal: controller.signal,
        onMeta: (meta) =>
          patch({
            meta: {
              strategy: meta.strategy,
              strategyReason: meta.strategy_reason,
              provider: meta.provider,
              model: meta.model,
              sources: [],
            },
          }),
        onDelta: (text) =>
          setEntries((current) =>
            current.map((entry) =>
              entry.id === answerEntry.id
                ? { ...entry, answer: (entry.answer ?? '') + text }
                : entry,
            ),
          ),
      })
      // The final payload is authoritative: apply sources and metadata from
      // the terminal done event (covers answers with no deltas, such as
      // greetings and insufficient-evidence declines).
      patch({
        streaming: false,
        answer: final.answer,
        meta: final.meta ?? {
          strategy: final.strategy,
          strategyReason: final.strategy_reason,
          provider: final.provider,
          model: final.model,
          sources: Array.isArray(final.sources) ? final.sources : [],
        },
      })
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        patch({ streaming: false, stopped: true })
      } else {
        const message =
          err instanceof ApiError
            ? err.message
            : 'Something went wrong while answering. Please try again.'
        patch({ streaming: false, error: message })
      }
    } finally {
      abortRef.current = null
      setStreaming(false)
    }
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <UniMindMark />
          <div>
            <h1>UniMind</h1>
            <p className="subtitle">Your Personal University AI Assistant</p>
          </div>
        </div>
        <p
          className={`status ${backendOnline === false ? 'status-down' : ''}`}
          role="status"
          aria-live="polite"
        >
          {backendOnline === null
            ? 'Checking backend…'
            : backendOnline
              ? 'Connected to your university library'
              : 'Backend unavailable — start it with: cd backend && python -m uvicorn app.main:app'}
        </p>
      </header>

      <main className="chat" aria-live="polite" onScroll={handleScroll} ref={scrollRef}>
        {entries.length === 0 && !streaming && <EmptyState onPick={ask} />}
        {entries.map((entry) => (
          <Message key={entry.id} entry={entry} />
        ))}
      </main>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          ask(input)
        }}
      >
        <div className="composer-box">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                ask(input)
              }
            }}
            placeholder="Ask UniMind anything about your university courses…"
            aria-label="Question"
            disabled={streaming}
            rows={1}
            maxLength={2000}
          />
          {streaming ? (
            <button type="button" className="stop-button" onClick={stop} aria-label="Stop generating">
              ■ Stop
            </button>
          ) : (
            <button
              type="submit"
              className="send-button"
              disabled={!input.trim()}
              aria-label="Send message"
            >
              ➤
            </button>
          )}
        </div>
        <p className="composer-hint muted">Enter to send · Shift + Enter for a new line</p>
      </form>
    </div>
  )
}
