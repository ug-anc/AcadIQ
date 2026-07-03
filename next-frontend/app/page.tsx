"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/* ─── API endpoints ─── */
const API_QUERY = "/api/v1/query";
const API_FEEDBACK = "/api/v1/feedback";
const API_SESSIONS = "/api/v1/sessions";

/* ─── Types ─── */
type ConfidenceBand = "found" | "low_confidence" | "not_found";

interface Citation {
  document_name: string;
  section: string;
  page_number: number;
  excerpt?: string;
  file_url?: string;
}

interface BotResponse {
  answer: string;
  confidence_band: ConfidenceBand;
  citations?: Citation[];
  confidence_score?: number;
  retrieval_latency_ms?: number;
  generation_latency_ms?: number;
  cached?: boolean;
  standalone_query?: string;
}

interface UserTurn {
  type: "user";
  text: string;
}

interface BotTurn {
  type: "bot";
  data: BotResponse;
  query: string;
  feedbackDone?: boolean;
}

type Turn = UserTurn | BotTurn;

interface SessionSummary {
  session_id: string;
  title: string;
  status: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
}

/* ─── Helpers ─── */

const BAND_LABEL: Record<ConfidenceBand, string> = {
  found: "Verified",
  low_confidence: "Low confidence",
  not_found: "Not found",
};

function newSessionId(): string {
  // Fix #1: full UUID4 instead of short random string
  return crypto.randomUUID();
}

/* ─── Components ─── */

function BandPill({ band }: { band: ConfidenceBand }) {
  const colors: Record<ConfidenceBand, { bg: string; fg: string }> = {
    found: { bg: "var(--navy-soft)", fg: "var(--navy)" },
    low_confidence: { bg: "var(--amber-soft)", fg: "var(--amber)" },
    not_found: { bg: "var(--stop-soft)", fg: "var(--stop)" },
  };
  const c = colors[band];
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 7,
        fontSize: 12,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: "0.04em",
        padding: "4px 10px",
        borderRadius: 999,
        marginBottom: 10,
        background: c.bg,
        color: c.fg,
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: "currentColor",
          display: "inline-block",
        }}
      />
      {BAND_LABEL[band]}
    </div>
  );
}

function BotMessage({
  turn,
  onFeedback,
}: {
  turn: BotTurn;
  onFeedback: (helpful: boolean) => void;
}) {
  const { data, feedbackDone } = turn;
  // const score = (data.confidence_score ?? 0).toFixed(3);
  const lat =
    (data.retrieval_latency_ms || 0) + (data.generation_latency_ms || 0);

  return (
    <div style={{ marginBottom: 26 }}>
      <BandPill band={data.confidence_band} />
      <div
        style={{
          fontFamily: 'Georgia, "Times New Roman", serif',
          fontSize: 17,
          whiteSpace: "pre-wrap",
          borderLeft: "3px solid var(--line)",
          padding: "2px 0 2px 16px",
        }}
      >
        {data.answer}
      </div>

      {/* Citations */}
      {data.citations && data.citations.length > 0 && (
        <details
          style={{
            marginTop: 12,
            border: "1px solid var(--line)",
            borderRadius: "var(--radius)",
            background: "#fff",
            overflow: "hidden",
          }}
        >
          <summary
            style={{
              cursor: "pointer",
              padding: "10px 14px",
              fontSize: 13,
              fontWeight: 600,
              color: "var(--navy)",
              listStyle: "none",
            }}
          >
            {data.citations.length} source
            {data.citations.length > 1 ? "s" : ""} referenced ▸
          </summary>
          {data.citations.map((c, i) => (
            <a
              key={i}
              href={c.file_url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "block",
                textDecoration: "none",
                color: "inherit",
                borderTop: "1px solid var(--line)",
                padding: "12px 14px",
                fontSize: 13,
              }}
            >
              <div style={{ fontWeight: 600, color: "var(--navy)" }}>
                {c.document_name} · Section {c.section} · Page {c.page_number}
              </div>
              {c.excerpt && (
                <div
                  style={{
                    color: "var(--muted)",
                    fontStyle: "italic",
                    marginTop: 3,
                  }}
                >
                  &ldquo;{c.excerpt}&rdquo;
                </div>
              )}
            </a>
          ))}
        </details>
      )}

      {/* Metadata */}
      {/* <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 10 }}> */}
        {/* confidence {score} · {lat} ms{data.cached ? " · cached" : ""} */}
        {/* {data.standalone_query && */}
          {/* data.standalone_query !== turn.query && */}
          {/* ` · interpreted as: "${data.standalone_query}"`} */}
      {/* </div> */}

      {/* <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 10 }}>
        {lat} ms{data.cached ? " · cached" : ""}
        {data.standalone_query &&
          data.standalone_query !== turn.query &&
          ` · interpreted as: "${data.standalone_query}"`}
      </div> */}

      {data.standalone_query && data.standalone_query !== turn.query && (
        <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 10 }}>
          interpreted as: "{data.standalone_query}"
        </div>
      )}

      {/* Feedback */}
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        {/* {feedbackDone ? (
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            Thanks — feedback recorded.
          </span>
        ) : ( */}
        {feedbackDone ? null : (
          <>
            <button
              onClick={() => onFeedback(true)}
              style={{
                border: "1px solid var(--line)",
                background: "#fff",
                borderRadius: 8,
                padding: "4px 10px",
                cursor: "pointer",
                fontSize: 14,
              }}
            >
              👍 Helpful
            </button>
            <button
              onClick={() => onFeedback(false)}
              style={{
                border: "1px solid var(--line)",
                background: "#fff",
                borderRadius: 8,
                padding: "4px 10px",
                cursor: "pointer",
                fontSize: 14,
              }}
            >
              👎 Not helpful
            </button>
          </>
        )}
      </div>
    </div>
  );
}

/* ─── Main Page ─── */

export default function Home() {
  /* ── State ── */
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoaded, setSessionsLoaded] = useState(false);
  const [archivedSessions, setArchivedSessions] = useState<SessionSummary[]>(
    []
  );
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [archivedOpen, setArchivedOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /* ── Scroll to bottom on new turns ── */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  /* ── Fetch sessions on mount ── */
  const fetchSessions = useCallback(async () => {
    try {
      const [activeRes, archivedRes] = await Promise.all([
        fetch(`${API_SESSIONS}?status=active`),
        fetch(`${API_SESSIONS}?status=archived`),
      ]);
      if (activeRes.ok) {
        const data = await activeRes.json();
        setSessions(data.sessions || []);
      }
      if (archivedRes.ok) {
        const data = await archivedRes.json();
        setArchivedSessions(data.sessions || []);
      }
    } catch {
      /* offline — start fresh */
    } finally {
      setSessionsLoaded(true);
    }
  }, []);

  useEffect(() => {
    fetchSessions().then(() => {
      /* handled in the effect below */
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── Auto-select or create session on mount ── */
  useEffect(() => {
    if (!sessionsLoaded) return;
    if (activeSessionId) return; // already picked
    if (sessions.length > 0) {
      switchSession(sessions[0].session_id);
    } else {
      handleNewChat();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions, sessionsLoaded]);

  /* ── Load turn history for a session ── */
  async function switchSession(sid: string) {
    setActiveSessionId(sid);
    try {
      const res = await fetch(`${API_SESSIONS}/${sid}`);
      if (res.ok) {
        const data = await res.json();
        const loaded: Turn[] = (data.turns || []).map(
          (t: { role: string; content: string }) =>
            t.role === "user"
              ? ({ type: "user", text: t.content } as UserTurn)
              : ({
                  type: "bot",
                  query: "",
                  data: {
                    answer: t.content,
                    confidence_band: "found" as ConfidenceBand,
                    citations: [],
                    confidence_score: 1,
                  },
                  feedbackDone: true,
                } as BotTurn)
        );
        setTurns(loaded);
      } else {
        setTurns([]);
      }
    } catch {
      setTurns([]);
    }
    inputRef.current?.focus();
  }

  /* ── New chat ── */
  async function handleNewChat() {
    const sid = newSessionId();
    try {
      await fetch(`${API_SESSIONS}?session_id=${sid}`, { method: "POST" });
    } catch {
      /* will be auto-created on first query */
    }
    setActiveSessionId(sid);
    setTurns([]);
    await fetchSessions();
    inputRef.current?.focus();
  }

  /* ── Archive ── */
  async function handleArchive(sid: string, e?: React.MouseEvent) {
    e?.stopPropagation();
    try {
      await fetch(`${API_SESSIONS}/${sid}/archive`, { method: "PATCH" });
    } catch {
      /* ignore */
    }
    if (activeSessionId === sid) {
      setActiveSessionId("");
      setTurns([]);
    }
    await fetchSessions();
  }

  /* ── Feedback ── */
  async function handleFeedback(index: number, helpful: boolean) {
    const turn = turns[index] as BotTurn;
    await fetch(API_FEEDBACK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: activeSessionId,
        query: turn.query,
        answer: turn.data.answer,
        helpful,
      }),
    }).catch(() => {});
    setTurns((prev) =>
      prev.map((t, i) =>
        i === index ? ({ ...t, feedbackDone: true } as BotTurn) : t
      )
    );
  }

  /* ── Submit query ── */
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || !activeSessionId) return;

    setTurns((prev) => [...prev, { type: "user", text: q }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(API_QUERY, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, session_id: activeSessionId }),
      });

      let data: BotResponse;
      if (res.status === 422) {
        data = {
          answer: "That query was rejected by the input guard.",
          confidence_band: "not_found",
          citations: [],
          confidence_score: 0,
        };
      } else {
        data = await res.json();
      }

      setTurns((prev) => [...prev, { type: "bot", data, query: q }]);

      // Refresh session list (title may have updated)
      fetchSessions();
    } catch {
      setTurns((prev) => [
        ...prev,
        {
          type: "bot",
          query: q,
          data: {
            answer: "Could not reach the server. Is the API running?",
            confidence_band: "not_found",
            citations: [],
            confidence_score: 0,
          },
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  /* ── Check if current session has messages ── */
  const hasMessages = turns.length > 0;

  /* ── Render ── */
  return (
    <div className="app-shell">
      {/* Mobile overlay */}
      <div
        className={`sidebar-overlay ${sidebarOpen ? "" : "hidden"}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* ─── Sidebar ─── */}
      <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="sidebar-header">
          <h2>Conversations</h2>
          <button className="btn-new-chat" onClick={handleNewChat}>
            <span style={{ fontSize: 18 }}>＋</span> New Chat
          </button>
        </div>

        <div className="session-list">
          {sessions.map((s) => (
            // <button
            //   key={s.session_id}
            //   className={`session-item ${
            //     s.session_id === activeSessionId ? "active" : ""
            //   }`}
            //   onClick={() => switchSession(s.session_id)}
            // >
            <div
              key={s.session_id}
              role="button"
              tabIndex={0}
              className={`session-item ${
                s.session_id === activeSessionId ? "active" : ""
              }`}
              onClick={() => switchSession(s.session_id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") switchSession(s.session_id);
              }}
            >
              <div className="session-item-content">
                <div className="session-title">
                  {s.title || "New chat"}
                </div>
                <div className="session-meta">
                  {s.turn_count} turn{s.turn_count !== 1 ? "s" : ""}
                </div>
              </div>
              <button
                className="session-archive-btn"
                onClick={(e) => handleArchive(s.session_id, e)}
                title="Archive"
              >
                🗃
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <div
              style={{
                padding: "20px 16px",
                color: "var(--muted)",
                fontSize: 13,
                textAlign: "center",
              }}
            >
              No conversations yet
            </div>
          )}
        </div>

        {/* Archived section */}
        {archivedSessions.length > 0 && (
          <div className="archived-section">
            <button
              className="archived-toggle"
              onClick={() => setArchivedOpen(!archivedOpen)}
            >
              <span
                style={{
                  transform: archivedOpen ? "rotate(90deg)" : "rotate(0)",
                  transition: "transform 0.15s ease",
                  display: "inline-block",
                }}
              >
                ▸
              </span>
              Archived ({archivedSessions.length})
            </button>
            {archivedOpen && (
              <div className="archived-list">
                {archivedSessions.map((s) => (
                  <button
                    key={s.session_id}
                    className="session-item"
                    onClick={() => switchSession(s.session_id)}
                  >
                    <div className="session-item-content">
                      <div className="session-title">
                        {s.title || "Archived chat"}
                      </div>
                      <div className="session-meta">
                        {s.turn_count} turn{s.turn_count !== 1 ? "s" : ""}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </aside>

      {/* ─── Chat Area ─── */}
      <div className="chat-area">
        {/* Header */}
        <header className="chat-header">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            ☰
          </button>
          <div className="chat-header-title">
            Ask<span style={{ color: "var(--navy)" }}>IITK</span>
          </div>
          <div style={{ color: "var(--muted)", fontSize: 13 }}>
            Every claim cited from official documents.
          </div>
        </header>

        {/* Messages or welcome */}
        {!hasMessages ? (
          <div className="welcome-screen">
            <div className="welcome-icon">📚</div>
            <h2>
              Ask<span style={{ color: "var(--navy)" }}>IITK</span>
            </h2>
            <p>
              Ask about credits, grading, attendance, backlogs, or branch-change
              rules. If the documents don&apos;t cover it, I&apos;ll say so
              rather than guess.
            </p>
          </div>
        ) : (
          <div className="chat-thread">
            <div className="chat-thread-inner">
              {turns.map((turn, i) =>
                turn.type === "user" ? (
                  <div key={i} style={{ textAlign: "right", marginBottom: 26 }}>
                    <div
                      style={{
                        display: "inline-block",
                        background: "var(--navy)",
                        color: "#fff",
                        padding: "10px 14px",
                        borderRadius:
                          "var(--radius) var(--radius) 2px var(--radius)",
                        maxWidth: "80%",
                        textAlign: "left",
                      }}
                    >
                      {turn.text}
                    </div>
                  </div>
                ) : (
                  <BotMessage
                    key={i}
                    turn={turn}
                    onFeedback={(helpful) => handleFeedback(i, helpful)}
                  />
                )
              )}

              {loading && (
                <div style={{ marginBottom: 26 }}>
                  <BandPill band="found" />
                  <div
                    style={{
                      fontFamily: 'Georgia, "Times New Roman", serif',
                      fontSize: 17,
                      borderLeft: "3px solid var(--line)",
                      padding: "2px 0 2px 16px",
                      color: "var(--muted)",
                    }}
                  >
                    Thinking…
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </div>
        )}

        {/* Input bar */}
        <footer className="chat-footer">
          <div className="chat-footer-inner">
            <form className="chat-form" onSubmit={handleSubmit}>
              <input
                ref={inputRef}
                className="chat-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                autoComplete="off"
                placeholder="e.g. What is the minimum attendance requirement?"
                disabled={loading}
              />
              <button
                type="submit"
                className="chat-submit"
                disabled={loading || !input.trim()}
              >
                Ask
              </button>
            </form>
            <div className="chat-tagline">
              A correct &ldquo;not found&rdquo; beats a confident wrong answer.
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
