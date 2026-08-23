// Single fetch wrapper for the whole app.
//
// - One place for the base URL, timeouts, JSON handling and error mapping.
// - Errors become ApiError instances with a user-facing `message` and a
//   `category` ("network" | "timeout" | "server" | "not-found" | ...).
// - Internal details (stack traces, raw server internals) are never exposed.

const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const DEFAULT_TIMEOUT_MS = 60000;

export class ApiError extends Error {
  constructor(message, { category = "server", status = 0, retryable = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.category = category;
    this.status = status;
    this.retryable = retryable;
  }
}

function friendlyDetail(detail) {
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }
  return null;
}

function messageFor(status, detail) {
  const text = friendlyDetail(detail);
  switch (status) {
    case 503:
      if (text && /llm/i.test(text)) {
        return "The language model (Ollama) is unavailable right now. Please try again later.";
      }
      if (text && /embed/i.test(text)) {
        return "The embedding service is unavailable right now. Please try again later.";
      }
      return text || "A backend service (model, search index or database) is temporarily unavailable.";
    case 404:
      return text || "The requested conversation was not found. It may have been deleted.";
    case 422:
      return "The request was invalid. Please check your input and try again.";
    case 501:
      return "This feature is not implemented yet.";
    case 401:
    case 403:
      return "You are not allowed to perform this action.";
    case 500:
      return "The server ran into an unexpected problem. Please try again.";
    default:
      return text || `The request failed (HTTP ${status}).`;
  }
}

export async function request(path, { method = "GET", body, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const options = {
      method,
      headers: { Accept: "application/json" },
      signal: controller.signal,
    };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(`${API_BASE}${path}`, options);
    if (response.status === 204) {
      return null;
    }
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      // Non-JSON body (or none); keep null.
    }
    if (!response.ok) {
      throw new ApiError(messageFor(response.status, payload?.detail), {
        category: response.status >= 500 ? "server" : "request",
        status: response.status,
        retryable: response.status >= 500,
      });
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error.name === "AbortError") {
      throw new ApiError(
        "The request timed out. The assistant may be busy — please try again.",
        { category: "timeout", retryable: true },
      );
    }
    throw new ApiError(
      "Cannot reach the backend. Make sure the API server is running.",
      { category: "network", retryable: true },
    );
  } finally {
    clearTimeout(timer);
  }
}
