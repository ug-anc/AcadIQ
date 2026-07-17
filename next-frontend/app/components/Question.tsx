"use client";

import { useState } from "react";

type QuestionFormProps = {
  onAsk?: (question: string) => void;
  placeholder?: string;
};

export default function QuestionForm({
  onAsk,
  placeholder = "Ask anything about IITK...",
}: QuestionFormProps) {
  const [question, setQuestion] = useState("");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = question.trim();

    if (!trimmed) return;

    onAsk?.(trimmed);
    setQuestion("");
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="w-full max-w-3xl"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={placeholder}
          className="flex-1 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white focus:ring-2 focus:ring-slate-200"
        />

        <button
          type="submit"
          className="inline-flex items-center justify-center rounded-xl bg-slate-900 px-5 py-3 text-sm font-medium text-white transition hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-400"
        >
          Ask
        </button>
      </div>
    </form>
  );
}
