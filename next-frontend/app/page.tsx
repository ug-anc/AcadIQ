"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";

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
  const colors: Record<ConfidenceBand, string> = {
    found: "bg-[var(--navy-soft)] text-[var(--navy)]",
    low_confidence: "bg-[var(--amber-soft)] text-[var(--amber)]",
    not_found: "bg-[var(--stop-soft)] text-[var(--stop)]",
  };
  return (
    <div
      className={`inline-flex items-center gap-[7px] text-xs font-semibold uppercase tracking-[0.04em] px-2.5 py-1 rounded-full mb-2.5 ${colors[band]}`}
    >
      <span className="w-[7px] h-[7px] rounded-full bg-current inline-block" />
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
    <div className="mb-[26px]">
      <BandPill band={data.confidence_band} />
      <div className="font-doc text-[17px] whitespace-pre-wrap border-l-[3px] border-[var(--line)] pt-[2px] pr-0 pb-[2px] pl-4">
        {data.answer}
      </div>

      {/* Citations */}
      {data.citations && data.citations.length > 0 && (
        <details className="mt-3 border border-[var(--line)] rounded-[var(--radius)] bg-white overflow-hidden">
          <summary className="cursor-pointer px-[14px] py-[10px] text-[13px] font-semibold text-[var(--navy)] list-none">
            {data.citations.length} source
            {data.citations.length > 1 ? "s" : ""} referenced ▸
          </summary>
          {data.citations.map((c, i) => (
            <a
              key={i}
              href={c.file_url}
              target="_blank"
              rel="noopener noreferrer"
              className="block no-underline text-inherit border-t border-[var(--line)] px-[14px] py-3 text-[13px]"
            >
              <div className="font-semibold text-[var(--navy)]">
                {c.document_name} · Section {c.section} · Page {c.page_number}
              </div>
              {c.excerpt && (
                <div className="text-[var(--muted)] italic mt-[3px]">
                  &ldquo;{c.excerpt}&rdquo;
                </div>
              )}
            </a>
          ))}
        </details>
      )}

      {data.standalone_query && data.standalone_query !== turn.query && (
        <div className="text-[var(--muted)] text-xs mt-2.5">
          interpreted as: "{data.standalone_query}"
        </div>
      )}

      {/* Feedback */}
      <div className="mt-2.5 flex gap-2">
        {/* {feedbackDone ? (
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            Thanks — feedback recorded.
          </span>
        ) : ( */}
        {feedbackDone ? null : (
          <>
            <button
              onClick={() => onFeedback(true)}
              className="border border-[var(--line)] bg-white rounded-lg px-2.5 py-1 cursor-pointer text-sm"
            >
              👍 Helpful
            </button>
            <button
              onClick={() => onFeedback(false)}
              className="border border-[var(--line)] bg-white rounded-lg px-2.5 py-1 cursor-pointer text-sm"
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
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /* ── Scroll to bottom on new turns ── */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

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
      {/* ─── Chat Area ─── */}
      <div className="chat-area">
        {/* Header */}
        <header className="flex m-2 items-center">
            <Image src="https://anc-website-blond.vercel.app/_next/image?url=%2Fimages%2Flogo%2Fanc_logo2.png&w=128&q=75"
            width={40}
            height={40}
            alt="AnC logo"
            className="m-2"
            />
            <h2 className="text-3xl m-2">
            Ask<span className="text-[var(--navy)]">IITK</span>
            </h2>
        </header>

        {/* Messages or welcome */}
        {!hasMessages ? (
          <div className="welcome-screen">
            <h2>
              Ask<span className="text-[var(--navy)]">IITK</span>
            </h2>
            <p>
              Ask questions from the UG Manual
            </p>
          </div>
        ) : (
          <div className="chat-thread">
            <div className="chat-thread-inner">
              {turns.map((turn, i) =>
                turn.type === "user" ? (
                  <div key={i} className="text-right mb-[26px]">
                    <div className="inline-block bg-[var(--navy)] text-white px-[14px] py-2.5 rounded-[var(--radius)_var(--radius)_2px_var(--radius)] max-w-[80%] text-left">
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
                <div className="mb-[26px]">
                  <BandPill band="found" />
                  <div className="font-doc text-[17px] border-l-[3px] border-[var(--line)] pt-[2px] pr-0 pb-[2px] pl-4 text-[var(--muted)]">
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
