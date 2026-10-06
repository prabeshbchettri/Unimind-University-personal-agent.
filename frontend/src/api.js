// Dedicated API client: all HTTP communication lives here, not in the UI
// components. The UI only consumes the returned data or the ApiError message.

// Must stay above the backend generation budget (LLM_TIMEOUT, default 120s):
// Ollama can legitimately take a long time on first generation, so the browser
// must not abort while the model is still working.
const CHAT_TIMEOUT_MS = 150_000

// Empty by default: requests go to the same origin (Vite dev proxy forwards
// them to FastAPI). Set VITE_API_BASE_URL to call the backend directly.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  /**
   * @param {string} message user-presentable explanation
   * @param {'network'|'timeout'|'server'} kind error category
   */
  constructor(message, kind) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
  }
}

/** Extract a useful, user-safe message from a FastAPI error body. */
async function describeHttpError(response) {
  let detail = ''
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') detail = body.detail
  } catch {
    // Non-JSON error body (e.g. a proxy page): fall through to the generic text.
  }
  if (response.status === 422) {
    return detail || 'That question could not be processed. Please rephrase it.'
  }
  return detail || `The assistant backend reported an error (HTTP ${response.status}).`
}

/** Real health check -- the UI shows "connected" only when this succeeds. */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(5_000),
    })
    return response.ok
  } catch {
    return false
  }
}

/**
 * Send one chat message and return the grounded answer payload
 * (answer, sources, strategy, strategy_reason, provider, ...).
 * Throws ApiError with a user-presentable message on any failure.
 */
export async function sendChatMessage(message) {
  let response
  try {
    response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: AbortSignal.timeout(CHAT_TIMEOUT_MS),
    })
  } catch (error) {
    if (error instanceof DOMException && (error.name === 'TimeoutError' || error.name === 'AbortError')) {
      throw new ApiError(
        'The assistant took too long to answer and the request was stopped. Please try again.',
        'timeout',
      )
    }
    throw new ApiError(
      'Cannot reach the assistant backend. Make sure the API server is running.',
      'network',
    )
  }

  if (!response.ok) {
    throw new ApiError(await describeHttpError(response), 'server')
  }

  let data
  try {
    data = await response.json()
  } catch {
    throw new ApiError('The assistant backend sent a malformed response.', 'server')
  }

  if (!data || typeof data.answer !== 'string' || !data.answer.trim()) {
    throw new ApiError('The assistant returned an empty answer. Please try again.', 'server')
  }
  return data
}
