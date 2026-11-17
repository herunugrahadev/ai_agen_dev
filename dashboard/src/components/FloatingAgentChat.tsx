"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";

interface ChatStepTrace {
  id: string;
  event: string;
  message: string;
  stepId?: number | null;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "streaming" | "clarification" | "completed" | "failed";
  traces?: ChatStepTrace[];
  questions?: string[];
}

interface FloatingAgentChatProps {
  apiBase: string;
  onExecutionComplete?: () => void;
}

const SAMPLE_PROMPTS = [
  "Cari berita teknologi AI terbaru hari ini dan simpan ke file news.txt",
  "Lihat daftar file yang ada di dalam workspace saat ini",
  "Carikan ringkasan tentang 9Router AI gateway lalu simpan ke 9router_notes.md",
];

export default function FloatingAgentChat({
  apiBase,
  onExecutionComplete,
}: FloatingAgentChatProps) {
  const [isMinimized, setIsMinimized] = useState(true);
  const [input, setInput] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Halo! Saya OpenMausBot. Ketik instruksi bahasa alami, dan saya akan menyusun rencana DAG serta mengeksekusi tool melalui 9Router secara real-time.",
      status: "completed",
    },
  ]);

  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isMinimized && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isMinimized]);

  const sendInstruction = async (instructionText: string) => {
    const trimmed = instructionText.trim();
    if (!trimmed || isRunning) return;

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: trimmed,
    };

    const assistantId = `a-${Date.now()}`;
    const initialAssistant: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "Menghubungkan ke OpenMausBot Execution Engine...",
      status: "streaming",
      traces: [],
    };

    setMessages((prev) => [...prev, userMsg, initialAssistant]);
    setInput("");
    setIsRunning(true);

    const updateAssistant = (updater: (msg: ChatMessage) => ChatMessage) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? updater(m) : m))
      );
    };

    try {
      const response = await fetch(`${apiBase}/api/v1/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instruction: trimmed,
          context: { source: "floating-chat" },
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status} dari ${apiBase}/api/v1/execute`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const dataLine = frame
            .split("\n")
            .find((line) => line.startsWith("data: "));
          if (!dataLine) continue;

          const rawJson = dataLine.slice(6).trim();
          if (!rawJson) continue;

          try {
            const payload = JSON.parse(rawJson) as {
              event: string;
              message: string;
              step_id?: number | null;
              data?: Record<string, unknown>;
            };

            updateAssistant((msg) => {
              const nextTraces = [
                ...(msg.traces ?? []),
                {
                  id: `${Date.now()}-${Math.random()}`,
                  event: payload.event,
                  message: payload.message,
                  stepId: payload.step_id,
                },
              ];

              if (payload.event === "clarification_needed") {
                const questions = Array.isArray(
                  payload.data?.clarification_questions
                )
                  ? (payload.data?.clarification_questions as string[])
                  : [];
                return {
                  ...msg,
                  status: "clarification",
                  content: payload.message,
                  questions,
                  traces: nextTraces,
                };
              }

              if (payload.event === "completed") {
                const results = payload.data?.results;
                return {
                  ...msg,
                  status: "completed",
                  content: results
                    ? `Eksekusi selesai.\n${JSON.stringify(results, null, 2)}`
                    : "Eksekusi selesai.",
                  traces: nextTraces,
                };
              }

              if (payload.event === "failed") {
                return {
                  ...msg,
                  status: "failed",
                  content: payload.message,
                  traces: nextTraces,
                };
              }

              return {
                ...msg,
                content: payload.message,
                traces: nextTraces,
              };
            });
          } catch {
            // Ignore malformed SSE frame
          }
        }
      }

      onExecutionComplete?.();
    } catch (err) {
      updateAssistant((msg) => ({
        ...msg,
        status: "failed",
        content:
          err instanceof Error
            ? `Gagal menjalankan instruksi: ${err.message}`
            : "Terjadi kesalahan saat menghubungi API.",
      }));
    } finally {
      setIsRunning(false);
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void sendInstruction(input);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendInstruction(input);
    }
  };

  if (isMinimized) {
    return (
      <div className="fixed bottom-5 right-5 z-50">
        <button
          type="button"
          onClick={() => setIsMinimized(false)}
          className="flex items-center gap-2.5 rounded-full border border-sky-500/40 bg-slate-900/95 px-4 py-3 text-sm font-semibold text-white shadow-xl shadow-sky-950/50 backdrop-blur transition hover:border-sky-400 hover:bg-slate-800"
        >
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              isRunning ? "bg-amber-400 animate-ping" : "bg-emerald-400"
            }`}
          />
          <span>Chat Agen OpenMausBot</span>
          <span className="rounded bg-sky-500/20 px-1.5 py-0.5 text-[10px] font-medium text-sky-300">
            Buka
          </span>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-5 right-5 z-50 flex h-[540px] w-[400px] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-2xl border border-slate-700/80 bg-slate-950/95 shadow-2xl shadow-black/70 backdrop-blur">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900/90 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              isRunning ? "bg-amber-400 animate-pulse" : "bg-emerald-400"
            }`}
          />
          <div>
            <h3 className="text-sm font-semibold text-white">
              OpenMausBot Agent Chat
            </h3>
            <p className="text-[11px] text-slate-400">
              {isRunning ? "Sedang mengeksekusi via 9Router..." : "Siap menerima instruksi"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() =>
              setMessages([
                {
                  id: "welcome",
                  role: "assistant",
                  content:
                    "Riwayat chat dibersihkan. Silakan kirim instruksi baru.",
                  status: "completed",
                },
              ])
            }
            disabled={isRunning}
            className="rounded-md px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
            title="Clear chat"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={() => setIsMinimized(true)}
            className="rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-200 hover:bg-slate-700"
            title="Minimize chat"
          >
            Minimize
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${
              msg.role === "user" ? "items-end" : "items-start"
            }`}
          >
            <div
              className={`max-w-[90%] rounded-xl px-3.5 py-2.5 text-xs leading-relaxed ${
                msg.role === "user"
                  ? "bg-sky-600 text-white"
                  : msg.status === "failed"
                  ? "border border-rose-500/40 bg-rose-950/40 text-rose-200"
                  : msg.status === "clarification"
                  ? "border border-amber-500/40 bg-amber-950/30 text-amber-100"
                  : "border border-slate-800 bg-slate-900 text-slate-200"
              }`}
            >
              <pre className="whitespace-pre-wrap font-sans">{msg.content}</pre>

              {msg.questions && msg.questions.length > 0 && (
                <ul className="mt-2 list-disc space-y-1 pl-4 text-amber-200">
                  {msg.questions.map((q, idx) => (
                    <li key={idx}>{q}</li>
                  ))}
                </ul>
              )}

              {msg.traces && msg.traces.length > 0 && (
                <details className="mt-2 border-t border-slate-800 pt-2">
                  <summary className="cursor-pointer text-[11px] font-medium text-slate-400 hover:text-slate-200">
                    Trace ReAct ({msg.traces.length} event)
                  </summary>
                  <div className="mt-1.5 space-y-1">
                    {msg.traces.map((t) => (
                      <div
                        key={t.id}
                        className="rounded bg-slate-950/70 px-2 py-1 font-mono text-[10px] text-slate-300"
                      >
                        <span className="text-sky-400">[{t.event}]</span>{" "}
                        {t.stepId ? `Step ${t.stepId}: ` : ""}
                        {t.message}
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>
        ))}

        {messages.length === 1 && (
          <div className="space-y-1.5 pt-2">
            <p className="text-[11px] font-medium text-slate-500">
              Contoh instruksi cepat:
            </p>
            {SAMPLE_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => void sendInstruction(prompt)}
                className="block w-full rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2 text-left text-[11px] text-slate-300 transition hover:border-sky-500/40 hover:bg-slate-800"
              >
                {prompt}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Input Area */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-slate-800 bg-slate-900/80 p-3"
      >
        <div className="flex items-end gap-2">
          <textarea
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isRunning}
            placeholder="Ketik instruksi untuk agen... (Enter untuk kirim)"
            className="flex-1 resize-none rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-sky-500 focus:outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isRunning || !input.trim()}
            className="rounded-xl bg-sky-600 px-3.5 py-2.5 text-xs font-semibold text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isRunning ? "..." : "Kirim"}
          </button>
        </div>
      </form>
    </div>
  );
}
