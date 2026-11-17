"""ReviewerAgent: validates execution outputs for accuracy, completeness, and QA."""

import json

from agent_core.agents.base import BaseAgent
from agent_core.config import get_settings
from agent_core.llm_client import NineRouterClient
from agent_core.models import (
    ExecutorOutput,
    PlanOutput,
    ReviewOutput,
    TaskRequest,
    TokenUsage,
)

REVIEWER_SYSTEM_PROMPT = """You are the Reviewer Agent in OpenMausBot.
Your role is Quality Assurance: verify that the Executor Agent's output and tool executions
accurately fulfill the user's objective and the Planner's subtasks without hallucinations.

You MUST respond with valid JSON matching this exact schema:
{
  "approved": true,
  "quality_score": 0-100,
  "feedback": "concise QA assessment",
  "issues_found": ["list of any issues or empty list"],
  "final_response": "polished, user-ready final answer summarizing verified results"
}
Return ONLY the JSON object, with no markdown code fences."""


class ReviewerAgent(BaseAgent):
    """Tier-3 QA & validation agent routed to `agent-reviewer` on 9Router."""

    def __init__(
        self,
        llm_client: NineRouterClient,
        model_alias: str | None = None,
    ) -> None:
        super().__init__(
            agent_id="openmausbot-reviewer",
            model_alias=model_alias or get_settings().model_alias_reviewer,
            llm_client=llm_client,
        )

    async def review(
        self,
        *,
        task: TaskRequest,
        plan: PlanOutput,
        execution: ExecutorOutput,
    ) -> tuple[ReviewOutput, TokenUsage]:
        """Validate execution results and produce final QA verdict."""
        review_payload = {
            "task_id": task.task_id,
            "objective": task.objective,
            "plan": plan.model_dump(mode="json"),
            "execution": execution.model_dump(mode="json"),
        }

        response, usage = await self.llm_client.chat_completion(
            model_alias=self.model_alias,
            agent_id=self.agent_id,
            task_id=task.task_id,
            temperature=0.0,
            max_tokens=1536,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(review_payload)},
            ],
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
        parsed["task_id"] = task.task_id

        review_output = ReviewOutput.model_validate(parsed)
        return review_output, usage
