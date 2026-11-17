"""PlannerAgent: decomposes incoming objectives into structured JSON subtasks."""

import json
from typing import Any

from agent_core.agents.base import BaseAgent
from agent_core.config import get_settings
from agent_core.llm_client import NineRouterClient
from agent_core.models import (
    DagPlanOutput,
    InstructionRequest,
    PlanOutput,
    TaskRequest,
    TokenUsage,
)

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent in OpenMausBot.
Your job is to decompose a user's objective into a clear, ordered list of executable subtasks.
Available tools for the Executor Agent:
- web_scraper(url: str, max_chars?: int): Scrapes public HTTP/HTTPS webpages.
- database_writer(title: str, payload: dict, category?: str, source_url?: str): Persists structured JSON data to PostgreSQL.

You MUST respond with valid JSON matching this exact schema:
{
  "goal_summary": "concise summary of the goal",
  "reasoning": "brief explanation of your decomposition strategy",
  "subtasks": [
    {
      "step_id": 1,
      "title": "short step title",
      "description": "detailed instruction for the executor",
      "recommended_tool": "web_scraper | database_writer | null",
      "tool_arguments": {}
    }
  ]
}
Return ONLY the JSON object, with no markdown code fences."""


class PlannerAgent(BaseAgent):
    """Tier-1 reasoning agent routed to `agent-planner` on 9Router."""

    def __init__(
        self,
        llm_client: NineRouterClient,
        model_alias: str | None = None,
    ) -> None:
        super().__init__(
            agent_id="openmausbot-planner",
            model_alias=model_alias or get_settings().model_alias_planner,
            llm_client=llm_client,
        )

    async def plan(self, task: TaskRequest) -> tuple[PlanOutput, TokenUsage]:
        """Generate a structured execution plan for the given TaskRequest."""
        user_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "objective": task.objective,
            "target_url": task.target_url,
            "metadata": task.metadata,
        }

        response, usage = await self.llm_client.chat_completion(
            model_alias=self.model_alias,
            agent_id=self.agent_id,
            task_id=task.task_id,
            temperature=0.2,
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
        parsed["task_id"] = task.task_id

        plan = PlanOutput.model_validate(parsed)
        return plan, usage

    async def parse_instruction(
        self,
        request: InstructionRequest,
        available_tools: list[dict[str, Any]],
    ) -> tuple[DagPlanOutput, TokenUsage]:
        """Parse natural language instruction into a DAG plan or NEEDS_CLARIFICATION."""
        system_prompt = """You are the Intent & DAG Task Planner in OpenMausBot.
Analyze the user's natural language instruction and the available tools.
If the instruction is ambiguous or missing mandatory parameters (e.g., missing recipient email,
missing webhook URL, unclear target), you MUST return:
{
  "status": "needs_clarification",
  "goal_summary": "what the user seems to want",
  "clarification_questions": ["specific question 1", ...],
  "steps": []
}

Otherwise, decompose the instruction into a valid DAG (Directed Acyclic Graph) of steps:
{
  "status": "ready",
  "goal_summary": "concise summary",
  "clarification_questions": [],
  "steps": [
    {
      "step_id": 1,
      "title": "short title",
      "description": "step instruction",
      "tool_name": "one of the available tool names",
      "arguments": {},
      "depends_on": []
    }
  ]
}
Return ONLY valid JSON."""

        user_payload = {
            "task_id": request.task_id,
            "instruction": request.instruction,
            "context": request.context,
            "available_tools": available_tools,
        }

        response, usage = await self.llm_client.chat_completion(
            model_alias=self.model_alias,
            agent_id=self.agent_id,
            task_id=request.task_id,
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
        parsed["task_id"] = request.task_id
        return DagPlanOutput.model_validate(parsed), usage
