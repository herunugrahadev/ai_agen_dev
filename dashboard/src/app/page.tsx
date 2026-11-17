"use client";

import { useCallback, useEffect, useState } from "react";
import FloatingAgentChat from "@/components/FloatingAgentChat";

interface AgentTokenMetric {
  agent: string;
  label: string;
  tokens: number;
}

interface PendingApprovalInfo {
  approval_id: string;
  task_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  reason: string;
  status: "pending" | "approved" | "rejected";
}

interface TaskRecord {
  task_id: string;
  status: string;
  objective: string;
  pending_approval?: PendingApprovalInfo | null;
  token_usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  started_at: string;
  completed_at?: string | null;
  error?: string | null;
}

interface MetricsOverview {
  queue_depth: number;
  total_tasks: number;
  routing_success_rate: number;
  status_counts: Record<string, number>;
  token_summary: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  agent_tokens: AgentTokenMetric[];
  active_tasks: TaskRecord[];
  recent_tasks: TaskRecord[];
  nine_router: {
    reachable: boolean;
    base_url: string;
    rate_limit_429_fallbacks: number;
  };
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const STATUS_BADGE: Record<string, string> = {
  queued: "bg-slate-700/70 text-slate-200 border-slate-600",
  planning: "bg-indigo-500/20 text-indigo-300 border-indigo-500/40",
  executing: "bg-sky-500/20 text-sky-300 border-sky-500/40",
  awaiting_approval: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  reviewing: "bg-purple-500/20 text-purple-300 border-purple-500/40",
  completed: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  rejected: "bg-rose-500/20 text-rose-300 border-rose-500/40",
  failed: "bg-red-500/20 text-red-300 border-red-500/40",
  needs_revision: "bg-orange-500/20 text-orange-300 border-orange-500/40",
};

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<MetricsOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("-");
  const [busyApprovalId, setBusyApprovalId] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/metrics/overview`, {
        cache: "no-store",
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data: MetricsOverview = await res.json();
      setMetrics(data);
      setError(null);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to connect to OpenMausBot API"
      );
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    const timer = setInterval(fetchMetrics, 3000);
    return () => clearInterval(timer);
  }, [fetchMetrics]);

  const handleApprovalDecision = async (
    approvalId: string,
    decision: "approve" | "reject"
  ) => {
    setBusyApprovalId(approvalId);
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/webhook/approvals/${approvalId}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            decision,
            reviewer: "dashboard-operator",
            comment: `${decision}d via OpenMausBot Dashboard`,
          }),
        }
      );
      if (!res.ok) {
        throw new Error(`Approval action failed: HTTP ${res.status}`);
      }
      await fetchMetrics();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Approval request failed");
    } finally {
      setBusyApprovalId(null);
    }
  };

  const maxAgentTokens = Math.max(
    1,
    ...(metrics?.agent_tokens.map((a) => a.tokens) ?? [1])
  );

  return (
    <main className="mx-auto max-w-7xl px-6 py-8">
      {/* Header */}
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            OpenMausBot + 9Router Control Plane
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Real-time agent token telemetry, 9Router routing success rate, and HITL task queue
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${
              metrics?.nine_router.reachable
                ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
                : "border-amber-500/40 bg-amber-500/15 text-amber-300"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                metrics?.nine_router.reachable
                  ? "bg-emerald-400 animate-pulse"
                  : "bg-amber-400"
              }`}
            />
            9Router: {metrics?.nine_router.reachable ? "Online" : "Offline / Standby"}
          </span>
          <span className="text-xs text-slate-400">Updated: {lastUpdated}</span>
          <button
            onClick={fetchMetrics}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
          >
            Refresh
          </button>
        </div>
      </header>

      {error && (
        <div className="mb-6 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          API Connection Warning ({API_BASE}): {error}
        </div>
      )}

      {/* KPI Cards */}
      <section className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Total Token Usage
          </p>
          <p className="mt-2 text-3xl font-bold text-white">
            {(metrics?.token_summary.total_tokens ?? 0).toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Prompt: {(metrics?.token_summary.prompt_tokens ?? 0).toLocaleString()} · Completion:{" "}
            {(metrics?.token_summary.completion_tokens ?? 0).toLocaleString()}
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Routing Success Rate
          </p>
          <p className="mt-2 text-3xl font-bold text-emerald-400">
            {metrics?.routing_success_rate ?? 100}%
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Completed: {metrics?.status_counts?.completed ?? 0} / Total:{" "}
            {metrics?.total_tasks ?? 0} tasks
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Redis Queue Depth
          </p>
          <p className="mt-2 text-3xl font-bold text-sky-400">
            {metrics?.queue_depth ?? 0}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Active Workers Running: {metrics?.active_tasks.length ?? 0}
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Pending HITL Approvals
          </p>
          <p className="mt-2 text-3xl font-bold text-amber-400">
            {metrics?.status_counts?.awaiting_approval ?? 0}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Rejected: {metrics?.status_counts?.rejected ?? 0} · Failed:{" "}
            {metrics?.status_counts?.failed ?? 0}
          </p>
        </div>
      </section>

      {/* Charts Row */}
      <section className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Token Usage per Agent Bar Chart */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-300">
            Token Usage per Agent Tier
          </h2>
          <div className="space-y-4">
            {(metrics?.agent_tokens ?? []).map((item) => {
              const pct = Math.round((item.tokens / maxAgentTokens) * 100);
              return (
                <div key={item.agent}>
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="font-medium text-slate-200">
                      {item.label}{" "}
                      <span className="text-slate-500">({item.agent})</span>
                    </span>
                    <span className="font-mono text-slate-300">
                      {item.tokens.toLocaleString()} tok
                    </span>
                  </div>
                  <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-sky-400 transition-all duration-500"
                      style={{ width: `${Math.max(pct, item.tokens > 0 ? 4 : 0)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Status Distribution */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-300">
            Workflow & Routing Status Breakdown
          </h2>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {Object.entries(metrics?.status_counts ?? {}).map(([statusKey, count]) => (
              <div
                key={statusKey}
                className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"
              >
                <span
                  className={`inline-block rounded-md border px-2 py-0.5 text-[11px] font-medium ${
                    STATUS_BADGE[statusKey] ?? "border-slate-700 text-slate-300"
                  }`}
                >
                  {statusKey}
                </span>
                <p className="mt-2 text-2xl font-semibold text-white">{count}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Active Queue & HITL Table */}
      <section className="mb-8 rounded-xl border border-slate-800 bg-slate-900/70 p-6">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-300">
          Active Queue & Pending Human Approvals (Real-Time)
        </h2>
        {metrics && metrics.active_tasks.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-800 text-xs uppercase text-slate-400">
                <tr>
                  <th className="py-2 pr-4">Task ID</th>
                  <th className="py-2 pr-4">Status</th>
                  <th className="py-2 pr-4">Objective</th>
                  <th className="py-2 pr-4">Tokens</th>
                  <th className="py-2">HITL Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {metrics.active_tasks.map((task) => (
                  <tr key={task.task_id} className="hover:bg-slate-800/30">
                    <td className="py-3 pr-4 font-mono text-xs text-slate-300">
                      {task.task_id.slice(0, 12)}...
                    </td>
                    <td className="py-3 pr-4">
                      <span
                        className={`inline-block rounded-md border px-2 py-0.5 text-xs font-medium ${
                          STATUS_BADGE[task.status] ?? "border-slate-700 text-slate-300"
                        }`}
                      >
                        {task.status}
                      </span>
                    </td>
                    <td className="py-3 pr-4 text-slate-200">
                      <p className="line-clamp-1">{task.objective}</p>
                      {task.pending_approval && (
                        <p className="mt-0.5 text-xs text-amber-300">
                          Tool: <code>{task.pending_approval.tool_name}</code> —{" "}
                          {JSON.stringify(task.pending_approval.arguments)}
                        </p>
                      )}
                    </td>
                    <td className="py-3 pr-4 font-mono text-xs text-slate-300">
                      {task.token_usage.total_tokens.toLocaleString()}
                    </td>
                    <td className="py-3">
                      {task.status === "awaiting_approval" && task.pending_approval ? (
                        <div className="flex gap-2">
                          <button
                            disabled={busyApprovalId === task.pending_approval.approval_id}
                            onClick={() =>
                              handleApprovalDecision(
                                task.pending_approval!.approval_id,
                                "approve"
                              )
                            }
                            className="rounded-md bg-emerald-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
                          >
                            Approve
                          </button>
                          <button
                            disabled={busyApprovalId === task.pending_approval.approval_id}
                            onClick={() =>
                              handleApprovalDecision(
                                task.pending_approval!.approval_id,
                                "reject"
                              )
                            }
                            className="rounded-md bg-rose-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-rose-500 disabled:opacity-50"
                          >
                            Reject
                          </button>
                        </div>
                      ) : (
                        <span className="text-xs text-slate-500">Running...</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-slate-500">
            No active or queued tasks right now. Submit a task via POST /api/v1/tasks.
          </p>
        )}
      </section>

      {/* Recent Tasks History */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-300">
          Recent Execution History
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 text-xs uppercase text-slate-400">
              <tr>
                <th className="py-2 pr-4">Task ID</th>
                <th className="py-2 pr-4">Status</th>
                <th className="py-2 pr-4">Objective</th>
                <th className="py-2 pr-4">Tokens</th>
                <th className="py-2">Started</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {(metrics?.recent_tasks ?? []).map((task) => (
                <tr key={task.task_id} className="hover:bg-slate-800/30">
                  <td className="py-2.5 pr-4 font-mono text-xs text-slate-400">
                    {task.task_id.slice(0, 12)}...
                  </td>
                  <td className="py-2.5 pr-4">
                    <span
                      className={`inline-block rounded-md border px-2 py-0.5 text-xs font-medium ${
                        STATUS_BADGE[task.status] ?? "border-slate-700 text-slate-300"
                      }`}
                    >
                      {task.status}
                    </span>
                  </td>
                  <td className="py-2.5 pr-4 text-slate-200 line-clamp-1">
                    {task.objective}
                  </td>
                  <td className="py-2.5 pr-4 font-mono text-xs text-slate-300">
                    {task.token_usage.total_tokens.toLocaleString()}
                  </td>
                  <td className="py-2.5 text-xs text-slate-400">
                    {new Date(task.started_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <FloatingAgentChat apiBase={API_BASE} onExecutionComplete={fetchMetrics} />
    </main>
  );
}
