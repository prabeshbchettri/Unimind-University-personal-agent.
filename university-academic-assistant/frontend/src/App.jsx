import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatInterface from "./components/ChatInterface";
import { ApiError } from "./services/apiClient";
import * as chatApi from "./services/chatApi";
import * as recommendationApi from "./services/recommendationApi";

let messageSequence = 0;
function nextMessageId() {
  messageSequence += 1;
  return `msg-${Date.now()}-${messageSequence}`;
}

const TYPING_MESSAGE = { id: "typing", role: "assistant", typing: true };

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const [mode, setMode] = useState("chat");
  const [globalError, setGlobalError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refreshSessions = useCallback(async () => {
    try {
      const response = await chatApi.listSessions();
      setSessions(response.items || []);
    } catch (error) {
      setGlobalError(error instanceof ApiError ? error.message : "Could not load conversations.");
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  const newChat = useCallback(() => {
    setActiveSessionId(null);
    setMessages([]);
    setGlobalError(null);
  }, []);

  const handleNewChat = useCallback(() => {
    newChat();
    setSidebarOpen(false);
  }, [newChat]);

  const selectSession = useCallback(
    async (sessionId) => {
      setGlobalError(null);
      try {
        const detail = await chatApi.getSession(sessionId);
        setActiveSessionId(sessionId);
        setMessages(
          (detail.messages || []).map((entry) => ({
            id: entry.message_id,
            role: entry.role,
            content: entry.content,
          })),
        );
      } catch (error) {
        setGlobalError(error instanceof ApiError ? error.message : "Could not open the conversation.");
      }
    },
    [],
  );

  const deleteSession = useCallback(
    async (sessionId) => {
      try {
        await chatApi.deleteSession(sessionId);
        if (sessionId === activeSessionId) {
          newChat();
        }
        refreshSessions();
      } catch (error) {
        setGlobalError(error instanceof ApiError ? error.message : "Could not delete the conversation.");
      }
    },
    [activeSessionId, newChat, refreshSessions],
  );

  const send = useCallback(
    async (text) => {
      setPending(true);
      setGlobalError(null);
      setMessages((previous) => [
        ...previous,
        { id: nextMessageId(), role: "user", content: text },
        TYPING_MESSAGE,
      ]);

      try {
        if (mode === "recommend") {
          const result = await recommendationApi.recommendBooks(text);
          setMessages((previous) => [
            ...previous.filter((entry) => !entry.typing),
            {
              id: nextMessageId(),
              role: "assistant",
              content: "",
              recommendations: result,
            },
          ]);
          return;
        }

        const result = await chatApi.sendMessage(text, activeSessionId);
        setMessages((previous) => [
          ...previous.filter((entry) => !entry.typing),
          {
            id: nextMessageId(),
            role: "assistant",
            content: result.answer,
            sources: result.sources || [],
            strategy: result.strategy,
            fallback: result.fallback,
          },
        ]);
        if (result.session_id && result.session_id !== activeSessionId) {
          setActiveSessionId(result.session_id);
        }
        refreshSessions();
      } catch (error) {
        setMessages((previous) => [
          ...previous.filter((entry) => !entry.typing),
          {
            id: nextMessageId(),
            role: "assistant",
            content: "",
            error: error instanceof ApiError ? error.message : "Something went wrong. Please try again.",
          },
        ]);
      } finally {
        setPending(false);
      }
    },
    [activeSessionId, mode, refreshSessions],
  );

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        loading={sessionsLoading}
        onNewChat={handleNewChat}
        onSelectSession={selectSession}
        onDeleteSession={deleteSession}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <div className="app__main">
        <header className="app__header">
          <button
            type="button"
            className="app__menu"
            aria-label="Toggle conversations"
            onClick={() => setSidebarOpen((open) => !open)}
          >
            ☰
          </button>
          <h1 className="app__title">University Academic Assistant</h1>
          <span className="app__status">Connected</span>
        </header>
        {globalError && (
          <div className="app__banner" role="alert">
            <span>{globalError}</span>
            <button type="button" className="app__banner-close" onClick={() => setGlobalError(null)}>
              Dismiss
            </button>
          </div>
        )}
        <ChatInterface
          messages={messages}
          pending={pending}
          mode={mode}
          onSend={send}
          onModeChange={setMode}
        />
      </div>
    </div>
  );
}