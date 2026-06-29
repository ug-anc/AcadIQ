"use client";

import { useEffect, useRef, useState } from "react";

const API = "/api/v1/query";
const FEEDBACK = "/api/v1/feedback";
const SESSION = "web-" + Math.random().toString(36).slice(2, 10);

type ConfidenceBand = "found" | "low_confidence" | "not_found";

interface Citation {
  document_name: string;
  section: string;
  page_number: number;
  excerpt?: string;
}

interface BotResponse {
  answer: string;
  confidence_band: ConfidenceBand;
  citations?: Citation[];
  confidence_score?: number;
  retrieval_latency_ms?: number;
  generation_latency_ms?: number;
  cached?: boolean;
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

const BAND_LABEL: Record<ConfidenceBand, string> = {
  found: "Verified",
  low_confidence: "Low confidence",
  not_found: "Not found",
};

function BandPill({ band }: { band: ConfidenceBand }) {
  const styles: Record<ConfidenceBand, React.CSSProperties> = {
    found: { background: "var(--navy-soft)", color: "var(--navy)" },
    low_confidence: { background: "var(--amber-soft)", color: "var(--amber)" },
    not_found: { background: "var(--stop-soft)", color: "var(--stop)" },
  };
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
        ...styles[band],
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
  const score = (data.confidence_score ?? 0).toFixed(3);
  const lat = (data.retrieval_latency_ms || 0) + (data.generation_latency_ms || 0);

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

      {/* Citations Block */}
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
            {data.citations.length} source{data.citations.length > 1 ? "s" : ""} referenced ▸
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
                <div style={{ color: "var(--muted)", fontStyle: "italic", marginTop: 3 }}>
                  &ldquo;{c.excerpt}&rdquo;
                </div>
              )}
            </a>
          ))}
        </details>
      )}

      {/* Metadata */}
      <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 10 }}>
        confidence {score} · {lat} ms{data.cached ? " · cached" : ""}
      </div>

      {/* Feedback Buttons */}
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        {feedbackDone ? (
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            Thanks — feedback recorded.
          </span>
        ) : (
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








export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([
    {
      type: "bot",
      query: "",
      data: {
        answer:
          "Ask about credits, grading, attendance, backlogs, or branch-change rules. If the documents don't cover it, I'll say so rather than guess.",
        confidence_band: "found",
        citations: [],
        confidence_score: 1,
        retrieval_latency_ms: 0,
        generation_latency_ms: 0,
      },
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function handleFeedback(index: number, helpful: boolean) {
    const turn = turns[index] as BotTurn;
    await fetch(FEEDBACK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: SESSION,
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

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q) return;

    setTurns((prev) => [...prev, { type: "user", text: q }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, session_id: SESSION }),
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

  return (
    <>
      <header
        style={{
          borderBottom: "1px solid var(--line)",
          padding: "22px 20px",
          textAlign: "center",
        }}
      >
        <div style={{ fontWeight: 700, letterSpacing: "-0.02em", fontSize: 22 }}>
          Ask<span style={{ color: "var(--navy)" }}>IITK</span>
        </div>
        <div style={{ color: "var(--muted)", fontSize: 13, marginTop: 2 }}>
          Answers only from official college documents — every claim cited.
        </div>
      </header>

      <main
        style={{
          maxWidth: 760,
          margin: "0 auto",
          padding: "24px 20px 140px",
          width: "100%",
        }}
      >
        {turns.map((turn, i) =>
          turn.type === "user" ? (
            <div key={i} style={{ textAlign: "right", marginBottom: 26 }}>
              <div
                style={{
                  display: "inline-block",
                  background: "var(--navy)",
                  color: "#fff",
                  padding: "10px 14px",
                  borderRadius: "var(--radius) var(--radius) 2px var(--radius)",
                  maxWidth: "80%",
                  textAlign: "left",
                }}
              >
                {turn.text}
              </div>
            </div>
          ) : (
            // <BotMessage
            //   key={i}
            //   turn={turn}
            //   onFeedback={(helpful) => handleFeedback(i, helpful)}
            // />
            <BotMessage
              key={i}
              turn={turn}
              // Pass the citations data here
              citations={turn.citations}
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
      </main>

      <footer
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          background: "linear-gradient(transparent, var(--paper) 24px)",
          padding: "18px 20px",
        }}
      >
        <form
          onSubmit={handleSubmit}
          style={{
            maxWidth: 760,
            margin: "0 auto",
            display: "flex",
            gap: 10,
            background: "#fff",
            border: "1px solid var(--line)",
            borderRadius: "var(--radius)",
            padding: "8px 8px 8px 14px",
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            autoComplete="off"
            placeholder="e.g. What is the minimum attendance requirement?"
            disabled={loading}
            style={{
              flex: 1,
              border: 0,
              outline: 0,
              fontSize: 16,
              background: "transparent",
              color: "var(--ink)",
            }}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            style={{
              background: "var(--navy)",
              color: "#fff",
              border: 0,
              borderRadius: 8,
              padding: "0 18px",
              fontWeight: 600,
              cursor: loading || !input.trim() ? "default" : "pointer",
              opacity: loading || !input.trim() ? 0.5 : 1,
            }}
          >
            Ask
          </button>
        </form>
        <div
          style={{
            textAlign: "center",
            color: "var(--muted)",
            fontSize: 12,
            marginTop: 8,
          }}
        >
          A correct &ldquo;not found&rdquo; beats a confident wrong answer.
        </div>
      </footer>
    </>
  );
}
