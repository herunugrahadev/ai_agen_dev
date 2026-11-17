"""End-to-end automated workflow: Trigger -> Planner -> Executor -> Reviewer."""

from datetime import datetime, timezone

from agent_core.agents.executor import ExecutorAgent
from agent_core.agents.planner import PlannerAgent
from agent_core.agents.reviewer import ReviewerAgent
from agent_core.llm_client import NineRouterClient
from agent_core.logging_utils import get_logger
from agent_core.models import TaskRequest, TaskStatus, TokenUsage, WorkflowResult
from agent_core.storage.postgres_store import PostgresStore
from agent_core.storage.redis_store import RedisStore
from agent_core.tools.base import ToolRegistry
from agent_core.tools.database_writer import DatabaseWriterTool
from agent_core.tools.web_scraper import WebScraperTool

logger = get_logger("openmausbot.orchestrator")


class WorkflowOrchestrator:
    """Coordinates Planner, Executor, and Reviewer agents with Redis & PostgreSQL state."""

    def __init__(
        self,
        *,
        llm_client: NineRouterClient,
        postgres_store: PostgresStore,
        redis_store: RedisStore | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.postgres_store = postgres_store
        self.redis_store = redis_store

        self.tool_registry = tool_registry or ToolRegistry(
            tools=[
                WebScraperTool(),
                DatabaseWriterTool(store=postgres_store),
            ]
        )

        self.planner = PlannerAgent(llm_client=llm_client)
        self.executor = ExecutorAgent(
            llm_client=llm_client,
            tool_registry=self.tool_registry,
        )
        self.reviewer = ReviewerAgent(llm_client=llm_client)

    async def run(self, task: TaskRequest) -> WorkflowResult:
        """Execute the full Planner -> Executor -> Reviewer pipeline for a TaskRequest."""
        result = WorkflowResult(
            task_id=task.task_id,
            status=TaskStatus.PLANNING,
            objective=task.objective,
        )
        await self.postgres_store.save_task_result(result)

        try:
            # Stage 1: Planning (Tier 1 - agent-planner)
            logger.info(
                "Starting PlannerAgent",
                extra={"event": "workflow_stage_start", "task_id": task.task_id, "agent_id": self.planner.agent_id},
            )
            plan, plan_usage = await self.planner.plan(task)
            result.plan = plan
            self._accumulate_tokens(result.token_usage, plan_usage)
            result.status = TaskStatus.EXECUTING
            await self.postgres_store.save_task_result(result)

            if self.redis_store is not None:
                await self.redis_store.set_short_term_memory(
                    task.task_id, "plan", plan.model_dump(mode="json")
                )

            # Stage 2: Execution with custom tools (Tier 2 - agent-executor)
            logger.info(
                "Starting ExecutorAgent",
                extra={"event": "workflow_stage_start", "task_id": task.task_id, "agent_id": self.executor.agent_id},
            )
            execution, exec_usage = await self.executor.execute_plan(task, plan)
            result.execution = execution
            self._accumulate_tokens(result.token_usage, exec_usage)
            result.status = TaskStatus.REVIEWING
            await self.postgres_store.save_task_result(result)

            if self.redis_store is not None:
                await self.redis_store.set_short_term_memory(
                    task.task_id, "execution", execution.model_dump(mode="json")
                )

            # Stage 3: Quality Assurance Review (Tier 3 - agent-reviewer)
            logger.info(
                "Starting ReviewerAgent",
                extra={"event": "workflow_stage_start", "task_id": task.task_id, "agent_id": self.reviewer.agent_id},
            )
            review, review_usage = await self.reviewer.review(
                task=task,
                plan=plan,
                execution=execution,
            )
            result.review = review
            self._accumulate_tokens(result.token_usage, review_usage)
            result.status = (
                TaskStatus.COMPLETED if review.approved else TaskStatus.NEEDS_REVISION
            )
            result.completed_at = datetime.now(timezone.utc)
            await self.postgres_store.save_task_result(result)

            if self.redis_store is not None:
                await self.redis_store.set_short_term_memory(
                    task.task_id, "review", review.model_dump(mode="json")
                )

            logger.info(
                "Workflow completed (status=%s, total_tokens=%d)",
                result.status.value,
                result.token_usage.total_tokens,
                extra={
                    "event": "workflow_completed",
                    "task_id": task.task_id,
                    "token_usage": result.token_usage.model_dump(),
                    "status": result.status.value,
                },
            )
            return result

        except Exception as exc:
            result.status = TaskStatus.FAILED
            result.error = str(exc)
            result.completed_at = datetime.now(timezone.utc)
            await self.postgres_store.save_task_result(result)
            logger.exception(
                "Workflow failed for task %s: %s",
                task.task_id,
                str(exc),
                extra={"event": "workflow_failed", "task_id": task.task_id},
            )
            return result

    @staticmethod
    def _accumulate_tokens(target: TokenUsage, delta: TokenUsage) -> None:
        target.prompt_tokens += delta.prompt_tokens
        target.completion_tokens += delta.completion_tokens
        target.total_tokens += delta.total_tokens
