"""End-to-end automated workflow with Human-in-the-Loop (HITL) suspension & resume."""

from datetime import datetime, timezone
from typing import Any

from agent_core.agents.executor import ExecutionStepResult, ExecutorAgent
from agent_core.agents.planner import PlannerAgent
from agent_core.agents.reviewer import ReviewerAgent
from agent_core.llm_client import NineRouterClient
from agent_core.logging_utils import get_logger
from agent_core.models import (
    ApprovalWebhookPayload,
    PendingApproval,
    PlanOutput,
    TaskRequest,
    TaskStatus,
    TokenUsage,
    ToolExecutionRecord,
    WorkflowResult,
)
from agent_core.storage.postgres_store import PostgresStore
from agent_core.storage.redis_store import RedisStore
from agent_core.tools.base import ToolRegistry
from agent_core.tools.database_deleter import DatabaseDeleterTool
from agent_core.tools.database_writer import DatabaseWriterTool
from agent_core.tools.email_sender import EmailSenderTool
from agent_core.tools.web_scraper import WebScraperTool

logger = get_logger("openmausbot.orchestrator")


class ApprovalNotFoundError(LookupError):
    """Raised when an approval_id is missing or expired in Redis."""


class WorkflowOrchestrator:
    """Coordinates Planner, Executor (with HITL), and Reviewer agents."""

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
                DatabaseDeleterTool(store=postgres_store),
                EmailSenderTool(),
            ]
        )

        self.planner = PlannerAgent(llm_client=llm_client)
        self.executor = ExecutorAgent(
            llm_client=llm_client,
            tool_registry=self.tool_registry,
        )
        self.reviewer = ReviewerAgent(llm_client=llm_client)

    async def run(self, task: TaskRequest) -> WorkflowResult:
        """Execute Planner -> Executor -> (optional HITL pause) -> Reviewer."""
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
            exec_step = await self.executor.execute_plan(task, plan)
            self._accumulate_tokens(result.token_usage, exec_step.token_usage)

            if exec_step.pending_approval is not None:
                return await self._suspend_for_approval(
                    task=task,
                    plan=plan,
                    result=result,
                    exec_step=exec_step,
                )

            return await self._finalize_with_review(
                task=task,
                plan=plan,
                result=result,
                exec_step=exec_step,
            )

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

    async def resume_approval(
        self,
        approval_id: str,
        decision: ApprovalWebhookPayload,
    ) -> WorkflowResult:
        """Resume or reject a suspended workflow based on human operator decision."""
        if self.redis_store is None:
            raise RuntimeError("RedisStore is required for HITL approval resume.")

        state = await self.redis_store.get_pending_approval_state(approval_id)
        if state is None:
            raise ApprovalNotFoundError(
                f"Pending approval '{approval_id}' not found or expired in Redis."
            )

        task = TaskRequest.model_validate(state["task"])
        plan = PlanOutput.model_validate(state["plan"])
        approval = PendingApproval.model_validate(state["approval"])
        prior_usage = TokenUsage.model_validate(state["prior_task_tokens"])

        result = await self.postgres_store.get_task_result(task.task_id) or WorkflowResult(
            task_id=task.task_id,
            status=TaskStatus.AWAITING_APPROVAL,
            objective=task.objective,
            plan=plan,
            token_usage=prior_usage,
        )

        now = datetime.now(timezone.utc)
        approval.decided_by = decision.reviewer
        approval.decision_comment = decision.comment
        approval.decided_at = now

        if decision.decision == "reject":
            approval.status = "rejected"
            await self.redis_store.delete_pending_approval_state(approval_id)
            result.pending_approval = approval
            result.status = TaskStatus.REJECTED
            result.error = (
                f"Tool '{approval.tool_name}' rejected by {decision.reviewer}"
                + (f": {decision.comment}" if decision.comment else "")
            )
            result.completed_at = now
            await self.postgres_store.save_task_result(result)

            logger.info(
                "HITL approval rejected (approval_id=%s, tool=%s)",
                approval_id,
                approval.tool_name,
                extra={"event": "hitl_rejected", "task_id": task.task_id},
            )
            return result

        # Approved: remove consumed approval key and resume ExecutorAgent
        approval.status = "approved"
        await self.redis_store.delete_pending_approval_state(approval_id)
        result.pending_approval = approval
        result.status = TaskStatus.EXECUTING
        await self.postgres_store.save_task_result(result)

        logger.info(
            "HITL approval granted (approval_id=%s, tool=%s)",
            approval_id,
            approval.tool_name,
            extra={"event": "hitl_approved", "task_id": task.task_id},
        )

        try:
            tool_records = [
                ToolExecutionRecord.model_validate(r) for r in state["tool_records"]
            ]
            exec_token_usage = TokenUsage.model_validate(state["executor_tokens"])

            exec_step = await self.executor.resume_after_approval(
                task_id=task.task_id,
                messages_snapshot=state["messages_snapshot"],
                pending_tool_calls=state["pending_tool_calls"],
                tool_records=tool_records,
                token_usage=exec_token_usage,
                approved_tool_call_id=approval.tool_call_id,
                remaining_iterations=int(state.get("remaining_iterations", 3)),
            )

            # Update total token usage with newly consumed executor tokens
            delta_prompt = exec_step.token_usage.prompt_tokens - exec_token_usage.prompt_tokens
            delta_completion = (
                exec_step.token_usage.completion_tokens - exec_token_usage.completion_tokens
            )
            delta_total = exec_step.token_usage.total_tokens - exec_token_usage.total_tokens
            result.token_usage.prompt_tokens += max(delta_prompt, 0)
            result.token_usage.completion_tokens += max(delta_completion, 0)
            result.token_usage.total_tokens += max(delta_total, 0)

            if exec_step.pending_approval is not None:
                return await self._suspend_for_approval(
                    task=task,
                    plan=plan,
                    result=result,
                    exec_step=exec_step,
                )

            return await self._finalize_with_review(
                task=task,
                plan=plan,
                result=result,
                exec_step=exec_step,
            )
        except Exception as exc:
            result.status = TaskStatus.FAILED
            result.error = str(exc)
            result.completed_at = datetime.now(timezone.utc)
            await self.postgres_store.save_task_result(result)
            return result

    async def _suspend_for_approval(
        self,
        *,
        task: TaskRequest,
        plan: PlanOutput,
        result: WorkflowResult,
        exec_step: ExecutionStepResult,
    ) -> WorkflowResult:
        if self.redis_store is None:
            raise RuntimeError(
                "RedisStore is required to suspend agent state for HITL approval."
            )

        approval = exec_step.pending_approval
        assert approval is not None

        suspended_state: dict[str, Any] = {
            "task": task.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "approval": approval.model_dump(mode="json"),
            "messages_snapshot": exec_step.messages_snapshot,
            "pending_tool_calls": exec_step.pending_tool_calls,
            "tool_records": [r.model_dump(mode="json") for r in exec_step.tool_records],
            "executor_tokens": exec_step.token_usage.model_dump(mode="json"),
            "prior_task_tokens": result.token_usage.model_dump(mode="json"),
            "remaining_iterations": exec_step.remaining_iterations,
        }
        await self.redis_store.save_pending_approval_state(
            approval.approval_id,
            suspended_state,
        )

        result.status = TaskStatus.AWAITING_APPROVAL
        result.pending_approval = approval
        await self.postgres_store.save_task_result(result)

        logger.info(
            "Task %s suspended in Redis awaiting approval %s for tool '%s'",
            task.task_id,
            approval.approval_id,
            approval.tool_name,
            extra={"event": "workflow_awaiting_approval", "task_id": task.task_id},
        )
        return result

    async def _finalize_with_review(
        self,
        *,
        task: TaskRequest,
        plan: PlanOutput,
        result: WorkflowResult,
        exec_step: ExecutionStepResult,
    ) -> WorkflowResult:
        assert exec_step.output is not None
        execution = exec_step.output
        result.execution = execution
        result.status = TaskStatus.REVIEWING
        await self.postgres_store.save_task_result(result)

        if self.redis_store is not None:
            await self.redis_store.set_short_term_memory(
                task.task_id, "execution", execution.model_dump(mode="json")
            )

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

    @staticmethod
    def _accumulate_tokens(target: TokenUsage, delta: TokenUsage) -> None:
        target.prompt_tokens += delta.prompt_tokens
        target.completion_tokens += delta.completion_tokens
        target.total_tokens += delta.total_tokens
