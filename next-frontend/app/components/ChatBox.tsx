"use client";

import { useState, useRef, useEffect, FormEvent, KeyboardEvent } from "react";

interface Message {
  question: string;
  answer: string;
  isError?: boolean;
}

export default function ChatBox() {
  const [messages, setMessages] = useState<Message[]>([
    { question: "What all can you do?", answer: "I can answer questions from the UG Manual" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Creates a stable session ID per component lifecycle to preserve history on the backend
  const [sessionId] = useState(() => crypto.randomUUID());

  // Dynamically expands or shrinks the textarea height to fit your text exactly
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [input]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e?: FormEvent) {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput(""); // Empty field immediately for a snappy user experience
    setIsLoading(true);

    // Optimistically push user query with a temporary visual loading indicator
    setMessages((prev) => [...prev, { question: userQuery, answer: "Thinking..." }]);

    try {
      // Routes seamlessly via the path proxy middleware configured in your next.config.ts
      console.log({
        "query": userQuery,
        "session_id": sessionId
      })
      const response = await fetch("http://localhost:8000/api/v1/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: userQuery,
          session_id: "75e6d6d8-f860-466d-a70d-f28e2025b682",
        }),
      });

      // Explicitly intercept your FastAPI @limiter.limit("30/minute") barrier code
      if (response.status === 429) {
        setMessages((prev) => [
          ...prev.slice(0, -1),
          { 
            question: userQuery, 
            answer: "Slow down a bit! You've reached the request limit. Please try again in a minute.", 
            isError: true 
          }
        ]);
        return;
      }

      if (!response.ok) {
        throw new Error(`Server returned structural error status: ${response.status}`);
      }

      const data = await response.json();

      // Replace the loading placeholder with the definitive text from your model
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { question: userQuery, answer: data.answer }
      ]);

    } catch (error) {
      console.error("API Communication Failure:", error);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { 
          question: userQuery, 
          answer: "Unable to establish a connection with the server. Please verify your backend server status.", 
          isError: true 
        }
      ]);
    } finally {
      setIsLoading(false);
      // Put focus back into text entry after response injection completes
      textareaRef.current?.focus();
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Normal desktop enter fires submission, Shift+Enter passes standard newlines
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="mt-8 w-full max-w-2xl mx-auto rounded-2xl border border-slate-100 bg-white shadow-sm overflow-hidden h-full flex flex-col">
      {/* Messages Render Timeline */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-4">
        {messages.map((msg, idx) => (
          <div key={idx} className="flex flex-col gap-1">
            <p className="font-semibold text-sm text-slate-800">You: {msg.question}</p>
            <p className={`text-sm whitespace-pre-wrap px-4 py-3 rounded-xl max-w-[100%] self-start ${
              msg.answer === "Thinking..." ? "text-slate-400 animate-pulse bg-slate-50" : 
              msg.isError ? "text-red-700 bg-red-50 border border-red-100" : 
              "text-slate-600 bg-slate-100"
            }`}>
              {msg.answer}
            </p>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Dynamic Textarea Ingestion Engine */}
      <div className="p-4">
        <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row items-end">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about IITK..."
            disabled={isLoading}
            className="flex-1 resize-none overflow-hidden max-h-48 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white focus:ring-2 focus:ring-slate-200 disabled:opacity-60"
          />

          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="inline-flex h-[46px] min-w-[90px] items-center justify-center rounded-xl bg-slate-900 px-5 text-sm font-medium text-white transition hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-400 disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed"
          >
            {isLoading ? "..." : "Ask"}
          </button>
        </form>
      </div>
    </div>
  );
}