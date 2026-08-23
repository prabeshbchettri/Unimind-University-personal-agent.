import { request } from "./apiClient";

// Chat + session API (backend routes: /chat, /chat/sessions).

export async function sendMessage(message, sessionId, topK) {
  const body = { message };
  if (sessionId) body.session_id = sessionId;
  if (topK !== undefined && topK !== null) body.top_k = topK;
  return request("/chat", { method: "POST", body });
}

export async function listSessions({ limit = 50, offset = 0 } = {}) {
  return request(`/chat/sessions?limit=${limit}&offset=${offset}`);
}

export async function getSession(sessionId) {
  return request(`/chat/sessions/${encodeURIComponent(sessionId)}`);
}

export async function deleteSession(sessionId) {
  return request(`/chat/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}
