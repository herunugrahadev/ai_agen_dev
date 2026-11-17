"""ExecutorAgent: executes subtasks using custom tools via 9Router tool calling."""

import json
from typing import Any

from agent_core.agents.base import BaseAgent
from agent_core.config import get_settings
from agent_core.llm_client import NineRouterClient
from agent_core.logging_utils import get_logger
from agent_core.models import (
    ExecutorOutput,
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
Once all subtasks are complete, respond with a concise JSON object:
{
  "summary": "what was executed and observed",
  "extracted_data": { ... }
}"""


class ExecutorAgent(BaseAgent):
    """Tier-2 low-latency tool-calling agent routed to `agent-executor` on 9Router."""

    def __init__(
        self,
        llm_client: NineRouterClient,
        tool_registry: ToolRegistry,
        model_alias: str | None = None,
        max_iterations: int | None = None,
    ) -> None:
        settings = get_settings()
        super().__init__(
            agent_id="openmausbot-executor",
            model_alias=model_alias or settings.model_alias_executor,
            llm_client=llm_client,
        )
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations or settings.executor_max_tool_iterations

    async def execute_plan(
        self,
        task: TaskRequest,
        plan: PlanOutput,
    ) -> tuple[ExecutorOutput, TokenUsage]:
        """Run the tool-calling loop until the model finishes or hits max_iterations."""
        total_usage = TokenUsage()
        tool_records: list[ToolExecutionRecord] = []
        tools_schema = self.tool_registry.openai_schemas()

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

        final_content = "{}"

        for iteration in range(1, self.max_iterations + 1):
            response, usage = await self.llm_client.chat_completion(
                model_alias=self.model_alias,
                agent_id=self.agent_id,
                task_id=task.task_id,
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
            tool_calls = assistant_msg.tool_calls or []

            if not tool_calls:
                final_content = assistant_msg.content or "{}"
                break

            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                try:
                    parsed_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    parsed_args = {}

                record = await self.tool_registry.execute_tool(
                    task_id=task.task_id,
                    tool_name=tc.function.name,
                    arguments=parsed_args,
                )
                tool_records.append(record)

                logger.info(
                    "Executed tool '%s' (success=%s)",
                    tc.function.name,
                    record.success,
                    extra={
                        "event": "tool_executed",
                        "agent_id": self.agent_id,
                        "task_id": task.task_id,
                        "routed_model": self.model_alias,
                        "latency_ms": record.latency_ms,
                    },
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(record.output, default=str),
                })
        else:
            final_content = json.dumps({
                "summary": f"Reached max tool iterations ({self.max_iterations}).",
                "extracted_data": {},
            })

        parsed_final = self._safe_parse_json(final_content)
        output = ExecutorOutput(
            task_id=task.task_id,
            summary=str(parsed_final.get("summary", final_content)),
            extracted_data=parsed_final.get("extracted_data", {})
            if isinstance(parsed_final.get("extracted_data"), dict)
            else {},
            tool_executions=tool_records,
        )
        return output, total_usage

    @staticmethod
    def _safe_parse_json(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"summary": raw}
        except json.JSONDecodeError:
            return {"summary": raw, "extracted_data": {}}
