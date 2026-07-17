"use client";

import Header from "../components/Header";
import ChatBox from "../components/ChatBox";

export default function Chat() {
  return (
    <div className="min-h-screen bg-slate-50">
      <Header />
      <main className="flex h-[calc(100vh-80px)] items-end justify-center px-4 pb-8 sm:px-6 lg:px-8">
        <ChatBox />
      </main>
    </div>
  );
}