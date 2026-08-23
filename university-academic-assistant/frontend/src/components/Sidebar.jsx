import { formatRelativeTime } from "../utils/format";

export default function Sidebar({
  sessions,
  activeSessionId,
  loading,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  open,
  onClose,
}) {
  return (
    <aside className={`sidebar ${open ? "sidebar--open" : ""}`}>
      <div className="sidebar__header">
        <button type="button" className="sidebar__new-chat" onClick={onNewChat}>
          + New chat
        </button>
      </div>

      <nav className="sidebar__list" aria-label="Chat history">
        {loading && <p className="sidebar__hint">Loading conversations…</p>}
        {!loading && sessions.length === 0 && (
          <p className="sidebar__hint">No conversations yet.</p>
        )}
        <ul>
          {sessions.map((session) => (
            <li key={session.session_id}>
              <button
                type="button"
                className={`sidebar__item ${session.session_id === activeSessionId ? "sidebar__item--active" : ""}`}
                onClick={() => {
                  onSelectSession(session.session_id);
                  onClose();
                }}
                title={session.title}
              >
                <span className="sidebar__item-title">{session.title || "Untitled chat"}</span>
                <span className="sidebar__item-meta">
                  {session.message_count} messages · {formatRelativeTime(session.updated_at)}
                </span>
              </button>
              <button
                type="button"
                className="sidebar__delete"
                aria-label={`Delete conversation: ${session.title}`}
                onClick={(event) => {
                  event.stopPropagation();
                  onDeleteSession(session.session_id);
                }}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}
