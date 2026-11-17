"""ExecutorAgent: executes subtasks using custom tools with HITL approval gating."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agent_core.agents.base import BaseAgent
from agent_core.config import get_settings
from agent_core.llm_client import NineRouterClient
from agent_core.logging_utils import get_logger
from agent_core.models import (
    DagPlanOutput,
    DagStep,
    ExecutionEvent,
    ExecutorOutput,
    InstructionRequest,
    PendingApproval,
    PlanOutput,
    TaskRequest,
    TokenUsage,
    ToolExecutionRecord,
)
from agent_core.tools.base import ToolRegistry

logger = get_logger("openmausbot.executor")

EXECUTOR_SYSTEM_PROMPT = """You are the Executor Agent in OpenMausBot.
You receive a structured plan of subtasks and must execute them using the provided tools.
Call `web_scraper` to gather web data when a URL is involved, and call `database_writer`
to persist structured extracted findings into PostgreSQL.
For destructive or external-impact tools (`database_deleter`, `email_sender`), invoke them
normally when required by the plan; the platform will automatically pause for human approval.
Once all subtasks are complete, respond with a concise JSON object:
{
  "summary": "what was executed and observed",
  "extracted_data": { ... }
}"""


@dataclass
class ExecutionStepResult:
    """Result of an ExecutorAgent run or resume, supporting HITL suspension."""

    output: ExecutorOutput | None = None
    pending_approval: PendingApproval | None = None
    messages_snapshot: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_records: list[ToolExecutionRecord] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    remaining_iterations: int = 0


class ExecutorAgent(BaseAgent):
    """Tier-2 low-latency tool-calling agent routed to `agent-executor` on 9Router."""

    def __init__(
        self,
        llm_client: NineRouterClient,
        tool_registry: ToolRegistry,
        model_alias: str | None = None,
        max_iterations: int | None = None,
        destructive_tools: set[str] | None = None,
    ) -> None:
        settings = get_settings()
        super().__init__(
            agent_id="openmausbot-executor",
            model_alias=model_alias or settings.model_alias_executor,
            llm_client=llm_client,
        )
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations or settings.executor_max_tool_iterations
        self.destructive_tools = destructive_tools or set(settings.destructive_tools)

    async def execute_plan(
        self,
        task: TaskRequest,
        plan: PlanOutput,
    ) -> ExecutionStepResult:
        """Start executing a plan, pausing if a destructive tool requires HITL approval."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({
                    "task_id": task.task_id,
                    "objective": task.objective,
                    "target_url": task.target_url,
                    "plan": plan.model_dump(mode="json"),
                }),
            },
        ]
        return await self._run_loop(
            task_id=task.task_id,
            messages=messages,
            tool_records=[],
            total_usage=TokenUsage(),
            remaining_iterations=self.max_iterations,
            pending_tool_calls=[],
            approved_tool_call_ids=set(),
        )

    async def resume_after_approval(
        self,
        *,
        task_id: str,
        messages_snapshot: list[dict[str, Any]],
        pending_tool_calls: list[dict[str, Any]],
        tool_records: list[ToolExecutionRecord],
        token_usage: TokenUsage,
        approved_tool_call_id: str,
        remaining_iterations: int,
    ) -> ExecutionStepResult:
        """Resume suspended execution after a human operator approves the pending tool call."""
        return await self._run_loop(
            task_id=task_id,
            messages=list(messages_snapshot),
            tool_records=list(tool_records),
            total_usage=token_usage.model_copy(),
            remaining_iterations=max(remaining_iterations, 1),
            pending_tool_calls=list(pending_tool_calls),
            approved_tool_call_ids={approved_tool_call_id},
        )

    async def _run_loop(
        self,
        *,
        task_id: str,
        messages: list[dict[str, Any]],
        tool_records: list[ToolExecutionRecord],
        total_usage: TokenUsage,
        remaining_iterations: int,
        pending_tool_calls: list[dict[str, Any]],
        approved_tool_call_ids: set[str],
    ) -> ExecutionStepResult:
        tools_schema = self.tool_registry.openai_schemas()

        # Process any pending tool calls from a resumed HITL turn first
        if pending_tool_calls:
            suspended = await self._process_tool_calls(
                task_id=task_id,
                messages=messages,
                tool_calls=pending_tool_calls,
                tool_records=tool_records,
                total_usage=total_usage,
                remaining_iterations=remaining_iterations,
                approved_tool_call_ids=approved_tool_call_ids,
            )
            if suspended is not None:
                return suspended

        final_content = "{}"

        for iteration_idx in range(remaining_iterations):
            response, usage = await self.llm_client.chat_completion(
                model_alias=self.model_alias,
                agent_id=self.agent_id,
                task_id=task_id,
                temperature=0.1,
                max_tokens=2048,
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
                messages=messages,
            )

            total_usage.prompt_tokens += usage.prompt_tokens
            total_usage.completion_tokens += usage.completion_tokens
            total_usage.total_tokens += usage.total_tokens

            assistant_msg = response.choices[0].message
            raw_tool_calls = assistant_msg.tool_calls or []

            if not raw_tool_calls:
                final_content = assistant_msg.content or "{}"
                break

            serialized_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tool_calls
            ]

            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": serialized_tool_calls,
            })

            suspended = await self._process_tool_calls(
                task_id=task_id,
                messages=messages,
                tool_calls=serialized_tool_calls,
                tool_records=tool_records,
                total_usage=total_usage,
                remaining_iterations=remaining_iterations - iteration_idx - 1,
                approved_tool_call_ids=approved_tool_call_ids,
            )
            if suspended is not None:
                return suspended
        else:
            final_content = json.dumps({
                "summary": f"Reached max tool iterations ({self.max_iterations}).",
                "extracted_data": {},
            })

        parsed_final = self._safe_parse_json(final_content)
        output = ExecutorOutput(
            task_id=task_id,
            summary=str(parsed_final.get("summary", final_content)),
            extracted_data=parsed_final.get("extracted_data", {})
            if isinstance(parsed_final.get("extracted_data"), dict)
            else {},
            tool_executions=tool_records,
        )
        return ExecutionStepResult(
            output=output,
            token_usage=total_usage,
            tool_records=tool_records,
        )

    async def _process_tool_calls(
        self,
        *,
        task_id: str,
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        tool_records: list[ToolExecutionRecord],
        total_usage: TokenUsage,
        remaining_iterations: int,
        approved_tool_call_ids: set[str],
    ) -> ExecutionStepResult | None:
        """Execute tool calls sequentially; pause and return ExecutionStepResult if HITL is needed."""
        for idx, tc in enumerate(tool_calls):
            tc_id = tc["id"]
            tool_name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}

            if (
                self.tool_registry.is_approval_required(tool_name, self.destructive_tools)
                and tc_id not in approved_tool_call_ids
            ):
                approval = PendingApproval(
                    task_id=task_id,
                    tool_name=tool_name,
                    tool_call_id=tc_id,
                    arguments=parsed_args,
                    reason=(
                        f"Tool '{tool_name}' is classified as destructive/external-impact "
                        "and requires human approval before execution."
                    ),
                )
                logger.warning(
                    "Pausing execution for HITL approval (tool=%s, approval_id=%s)",
                    tool_name,
                    approval.approval_id,
                    extra={
                        "event": "hitl_approval_requested",
                        "agent_id": self.agent_id,
                        "task_id": task_id,
                        "routed_model": self.model_alias,
                    },
                )
                return ExecutionStepResult(
                    pending_approval=approval,
                    messages_snapshot=messages,
                    pending_tool_calls=tool_calls[idx:],
                    tool_records=tool_records,
                    token_usage=total_usage,
                    remaining_iterations=remaining_iterations,
                )

            record = await self.tool_registry.execute_tool(
                task_id=task_id,
                tool_name=tool_name,
                arguments=parsed_args,
            )
            tool_records.append(record)

            logger.info(
                "Executed tool '%s' (success=%s)",
                tool_name,
                record.success,
                extra={
                    "event": "tool_executed",
                    "agent_id": self.agent_id,
                    "task_id": task_id,
                    "routed_model": self.model_alias,
                    "latency_ms": record.latency_ms,
                },
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps(record.output, default=str),
            })

        return None

    @staticmethod
    def _safe_parse_json(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"summary": raw}
        except json.JSONDecodeError:
            return {"summary": raw, "extracted_data": {}}

    async def execute_dag_react(
        self,
        *,
        request: InstructionRequest,
        plan: DagPlanOutput,
        max_retries_per_step: int = 3,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute a DAG plan with parallel wave scheduling and ReAct self-correction."""
        pending_steps: dict[int, DagStep] = {s.step_id: s for s in plan.steps}
        completed_outputs: dict[int, dict[str, Any]] = {}
        total_usage = TokenUsage()

        while pending_steps:
            ready_steps = [
                step
                for step in pending_steps.values()
                if all(dep in completed_outputs for dep in step.depends_on)
            ]
            if not ready_steps:
                yield ExecutionEvent(
                    event="failed",
                    task_id=request.task_id,
                    message="DAG deadlock: cyclic or unsatisfied step dependencies.",
                )
                return

            # Run ready wave steps concurrently
            wave_tasks = [
                self._execute_step_react(
                    task_id=request.task_id,
                    step=step,
                    prior_outputs=completed_outputs,
                    max_retries=max_retries_per_step,
                )
                for step in ready_steps
            ]
            wave_results = await asyncio.gather(*wave_tasks)

            for step, step_events, step_output, step_usage, succeeded in wave_results:
                total_usage.prompt_tokens += step_usage.prompt_tokens
                total_usage.completion_tokens += step_usage.completion_tokens
                total_usage.total_tokens += step_usage.total_tokens

                for ev in step_events:
                    yield ev

                if not succeeded:
                    yield ExecutionEvent(
                        event="failed",
                        task_id=request.task_id,
                        step_id=step.step_id,
                        message=f"Step {step.step_id} ({step.tool_name}) failed after {max_retries_per_step} retries.",
                    )
                    return

                completed_outputs[step.step_id] = step_output
                pending_steps.pop(step.step_id, None)

        yield ExecutionEvent(
            event="completed",
            task_id=request.task_id,
            message="Selesai",
            data={
                "results": completed_outputs,
                "token_usage": total_usage.model_dump(),
            },
        )

    async def _execute_step_react(
        self,
        *,
        task_id: str,
        step: DagStep,
        prior_outputs: dict[int, dict[str, Any]],
        max_retries: int,
    ) -> tuple[DagStep, list[ExecutionEvent], dict[str, Any], TokenUsage, bool]:
        """Run ReAct Thought -> Action -> Observation loop with self-correction retries."""
        events: list[ExecutionEvent] = []
        step_usage = TokenUsage()
        current_args = dict(step.arguments)
        thought = f"Menjalankan langkah {step.step_id}: {step.title} menggunakan {step.tool_name}."

        for attempt in range(1, max_retries + 1):
            events.append(
                ExecutionEvent(
                    event="thought",
                    task_id=task_id,
                    step_id=step.step_id,
                    message=thought,
                    data={"attempt": attempt},
                )
            )
            events.append(
                ExecutionEvent(
                    event="action",
                    task_id=task_id,
                    step_id=step.step_id,
                    message=f"Menjalankan {step.tool_name}...",
                    data={"tool_name": step.tool_name, "arguments": current_args},
                )
            )

            record = await self.tool_registry.execute_tool(
                task_id=task_id,
                tool_name=step.tool_name,
                arguments=current_args,
            )

            events.append(
                ExecutionEvent(
                    event="observation",
                    task_id=task_id,
                    step_id=step.step_id,
                    message=(
                        f"{step.tool_name} berhasil."
                        if record.success
                        else f"{step.tool_name} gagal: {record.output}"
                    ),
                    data={
                        "success": record.success,
                        "output": record.output,
                        "latency_ms": record.latency_ms,
                    },
                )
            )

            if record.success:
                output_dict = (
                    record.output
                    if isinstance(record.output, dict)
                    else {"result": record.output}
                )
                return step, events, output_dict, step_usage, True

            if attempt >= max_retries:
                break

            # Self-correction via 9Router (tier: agent-executor)
            thought, current_args, corr_usage = await self._self_correct_arguments(
                task_id=task_id,
                step=step,
                failed_args=current_args,
                error_observation=record.output,
                prior_outputs=prior_outputs,
            )
            step_usage.prompt_tokens += corr_usage.prompt_tokens
            step_usage.completion_tokens += corr_usage.completion_tokens
            step_usage.total_tokens += corr_usage.total_tokens

            events.append(
                ExecutionEvent(
                    event="retry",
                    task_id=task_id,
                    step_id=step.step_id,
                    message=f"Memperbaiki parameter {step.tool_name} (percobaan {attempt + 1}/{max_retries})...",
                    data={"corrected_arguments": current_args, "thought": thought},
                )
            )

        return step, events, {}, step_usage, False

    async def _self_correct_arguments(
        self,
        *,
        task_id: str,
        step: DagStep,
        failed_args: dict[str, Any],
        error_observation: Any,
        prior_outputs: dict[int, dict[str, Any]],
    ) -> tuple[str, dict[str, Any], TokenUsage]:
        """Ask 9Router (agent-executor) to repair invalid tool arguments from error observation."""
        tool = self.tool_registry.get(step.tool_name)
        tool_schema = tool.to_openai_schema() if tool is not None else {}

        prompt_payload = {
            "step": step.model_dump(mode="json"),
            "failed_arguments": failed_args,
            "error_observation": error_observation,
            "tool_schema": tool_schema,
            "prior_step_outputs": prior_outputs,
        }

        response, usage = await self.llm_client.chat_completion(
            model_alias=self.model_alias,
            agent_id=self.agent_id,
            task_id=task_id,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a self-correcting ReAct Executor Agent. "
                        "A tool call failed. Analyze the error_observation and tool_schema, "
                        "then return ONLY valid JSON: "
                        '{"thought": "why it failed and how you fixed it", "arguments": {...}}'
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload, default=str)},
            ],
        )

        raw = response.choices[0].message.content or "{}"
        parsed = self._safe_parse_json(raw)
        fixed_thought = str(parsed.get("thought", "Memperbaiki argumen berdasarkan error."))
        fixed_args = (
            parsed.get("arguments")
            if isinstance(parsed.get("arguments"), dict)
            else failed_args
        )
        return fixed_thought, fixed_args, usage
