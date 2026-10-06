import { useEffect, useRef, useState } from 'react'
import './App.css'
import { ApiError, checkHealth, sendChatMessage } from './api.js'

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

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [backendOnline, setBackendOnline] = useState(null) // null = checking
  const endRef = useRef(null)

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

  // Keep the newest message in view.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, busy])

  async function handleSubmit(event) {
    event.preventDefault()
    const question = input.trim()
    if (!question || busy) return

    const entry = { id: crypto.randomUUID(), question, answer: null, meta: null, error: '' }
    setMessages((current) => [...current, entry])
    setInput('')
    setBusy(true)

    try {
      const data = await sendChatMessage(question)
      setMessages((current) =>
        current.map((item) =>
          item.id === entry.id
            ? {
                ...item,
                answer: data.answer,
                meta: {
                  strategy: data.strategy,
                  strategyReason: data.strategy_reason,
                  provider: data.provider,
                  sources: Array.isArray(data.sources) ? data.sources : [],
                },
              }
            : item,
        ),
      )
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Something went wrong. Please try again.'
      setMessages((current) =>
        current.map((item) => (item.id === entry.id ? { ...item, error: message } : item)),
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Adaptive University RAG Assistant</h1>
        <p className="subtitle">
          Ask questions about university academic documents and get grounded, cited answers.
        </p>
        <p className={`status ${backendOnline === false ? 'status-down' : ''}`}>
          {backendOnline === null
            ? 'Checking backend…'
            : backendOnline
              ? 'Backend connected'
              : 'Backend unavailable — start it with: cd backend && python -m uvicorn app.main:app'}
        </p>
      </header>

      <main className="chat" aria-live="polite">
        {messages.length === 0 && !busy && (
          <p className="muted">
            Try: “What is the minimum attendance requirement?” or “What are the credit
            requirements for CS201?”
          </p>
        )}
        {messages.map((entry) => (
          <div key={entry.id} className="turn">
            <div className="message message-user">
              <span className="message-role">You</span>
              <p>{entry.question}</p>
            </div>
            <div className="message message-assistant">
              <span className="message-role">Assistant</span>
              {entry.answer === null && !entry.error && <p className="muted">Thinking…</p>}
              {entry.error && <p className="error-text">{entry.error}</p>}
              {entry.answer && <p>{entry.answer}</p>}
              {entry.meta && <AnswerMeta meta={entry.meta} />}
            </div>
          </div>
        ))}
        {busy && (
          <div className="message message-assistant">
            <span className="message-role">Assistant</span>
            <p className="muted">Retrieving evidence and generating an answer…</p>
          </div>
        )}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask about an academic policy…"
          aria-label="Question"
          disabled={busy}
          maxLength={2000}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          {busy ? 'Sending…' : 'Send'}
        </button>
      </form>
    </div>
  )
}

export default App
