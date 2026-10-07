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

/** Sources + retrieval metadata for one answer, kept compact and collapsible. */
function AnswerMeta({ meta }) {
  const hasSources = meta.sources && meta.sources.length > 0
  return (
    <div className="answer-meta">
      {hasSources && (
        <details className="sources-details">
          <summary>
            <span>Sources ({meta.sources.length})</span>
            <span className="chevron" aria-hidden="true">▾</span>
          </summary>
          <ol className="sources-list">
            {meta.sources.map((source) => (
              <li key={`${source.document}-${source.page}`}>
                {source.document} · Page {source.page}
              </li>
            ))}
          </ol>
        </details>
      )}
      <div className="meta-tags">
        <span title={meta.strategyReason || undefined}>Strategy: {strategyLabel(meta.strategy)}</span>
        <span>Provider: {providerLabel(meta.provider)}</span>
      </div>
    </div>
  )
}

/** One user or assistant turn. */
function Message({ entry }) {
  if (entry.role === 'user') {
    return (
      <div className="turn turn-user" id={`turn-${entry.id}`}>
        <div className="user-bubble">{entry.question}</div>
      </div>
    )
  }

  return (
    <div className="turn turn-assistant" id={`turn-${entry.id}`}>
      <div className="assistant-avatar">
        <UniMindMark />
      </div>
      <div className="assistant-body">
        <div className="assistant-name">UniMind</div>
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
      <h2>UniMind</h2>
      <p className="empty-subtitle">University Academic Assistant</p>
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
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const scrollRef = useRef(null)
  const stickToBottomRef = useRef(true)
  const abortRef = useRef(null)
  const entriesRef = useRef([])
  const textareaRef = useRef(null)
  useEffect(() => {
    entriesRef.current = entries
  }, [entries])

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

  // Grow the composer with its content, up to the CSS max-height.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [input])

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }, [])

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  // New chat clears the in-session conversation. History is not persisted, so
  // this simply starts an empty thread.
  const startNewChat = useCallback(() => {
    if (abortRef.current) return // do not drop an in-flight answer
    setEntries([])
    setSidebarOpen(false)
    stickToBottomRef.current = true
  }, [])

  const scrollToEntry = useCallback((id) => {
    setSidebarOpen(false)
    const el = document.getElementById(`turn-${id}`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const ask = useCallback(async (question) => {
    const trimmed = question.trim()
    if (!trimmed || abortRef.current) return

    // Stateless multi-turn: send the visible conversation back with every
    // request (bounded to the most recent turns); the backend keeps nothing.
    const history = entriesRef.current
      .slice(-10)
      .flatMap((entry) => {
        if (entry.role === 'user') return [{ role: 'user', content: entry.question }]
        if (entry.role === 'assistant' && entry.answer && !entry.error)
          return [{ role: 'assistant', content: entry.answer }]
        return []
      })

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
        history,
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

  const askedQuestions = entries.filter((entry) => entry.role === 'user')

  return (
    <div className="app">
      <aside className={`sidebar${sidebarOpen ? ' open' : ''}`}>
        <div className="sidebar-brand">
          <UniMindMark />
          <div className="brand-text">
            <h1>UniMind</h1>
            <p className="subtitle">University Academic Assistant</p>
          </div>
        </div>

        <button type="button" className="new-chat" onClick={startNewChat} disabled={streaming}>
          <span aria-hidden="true">+</span> New chat
        </button>

        <div className="sidebar-section">
          <h2>This session</h2>
          {askedQuestions.length === 0 ? (
            <p className="sidebar-empty">No questions yet.</p>
          ) : (
            <ul className="session-list">
              {askedQuestions.map((entry) => (
                <li key={entry.id}>
                  <button
                    type="button"
                    className="session-item"
                    title={entry.question}
                    onClick={() => scrollToEntry(entry.id)}
                  >
                    {entry.question}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="sidebar-footer">
          Answers are grounded in your uploaded university materials. History is
          kept for this browser session only.
        </div>
      </aside>

      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          role="presentation"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="main">
        <header className="topbar">
          <button
            type="button"
            className="menu-button"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-label="Toggle sidebar"
          >
            ☰
          </button>
          <span className="topbar-title">UniMind</span>
          <p
            className={`status${backendOnline === false ? ' status-down' : ''}`}
            role="status"
            aria-live="polite"
          >
            {backendOnline === null
              ? 'Checking backend…'
              : backendOnline
                ? 'Connected'
                : 'Backend unavailable — start it with: cd backend && python -m uvicorn app.main:app'}
          </p>
        </header>

        <main className="chat" aria-live="polite" onScroll={handleScroll} ref={scrollRef}>
          <div className="thread">
            {entries.length === 0 && !streaming && <EmptyState onPick={ask} />}
            {entries.map((entry) => (
              <Message key={entry.id} entry={entry} />
            ))}
          </div>
        </main>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault()
            ask(input)
          }}
        >
          <div className="composer-inner">
            <div className="composer-box">
              <textarea
                ref={textareaRef}
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
                  Stop
                </button>
              ) : (
                <button
                  type="submit"
                  className="send-button"
                  disabled={!input.trim()}
                  aria-label="Send message"
                >
                  ↑
                </button>
              )}
            </div>
            <p className="composer-hint muted">Enter to send · Shift + Enter for a new line</p>
          </div>
        </form>
      </div>
    </div>
  )
}
