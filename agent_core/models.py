"""Pydantic domain models for OpenMausBot tasks, plans, tool calls, and reviews."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"


class TaskRequest(BaseModel):
    """Incoming task request from REST API or Webhook."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    objective: str = Field(..., min_length=3, description="High-level goal for the agent workflow")
    target_url: str | None = Field(default=None, description="Optional URL to inspect/scrape")
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = Field(default="api")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WebhookPayload(BaseModel):
    """Webhook event payload for automated triggers."""

    event_type: str = Field(..., examples=["task.requested", "url.monitor"])
    objective: str = Field(..., min_length=3)
    target_url: str | None = None
    callback_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubTask(BaseModel):
    """Single executable step produced by PlannerAgent."""

    step_id: int = Field(..., ge=1)
    title: str
    description: str
    recommended_tool: str | None = Field(
        default=None,
        description="Suggested tool name (e.g., web_scraper, database_writer)",
    )
    tool_arguments: dict[str, Any] = Field(default_factory=dict)


class PlanOutput(BaseModel):
    """Structured JSON output from PlannerAgent."""

    task_id: str
    goal_summary: str
    reasoning: str
    subtasks: list[SubTask] = Field(..., min_length=1)


class ToolExecutionRecord(BaseModel):
    """Record of a single tool invocation by ExecutorAgent."""

    tool_name: str
    arguments: dict[str, Any]
    success: bool
    output: dict[str, Any] | str
    latency_ms: float


class ExecutorOutput(BaseModel):
    """Consolidated output from ExecutorAgent across all subtasks."""

    task_id: str
    summary: str
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    tool_executions: list[ToolExecutionRecord] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    """Quality assurance verdict from ReviewerAgent."""

    task_id: str
    approved: bool
    quality_score: int = Field(..., ge=0, le=100)
    feedback: str
    issues_found: list[str] = Field(default_factory=list)
    final_response: str


class TokenUsage(BaseModel):
    """Aggregated token usage across LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class WorkflowResult(BaseModel):
    """Complete end-to-end execution result stored in PostgreSQL and returned to user."""

    task_id: str
    status: TaskStatus
    objective: str
    plan: PlanOutput | None = None
    execution: ExecutorOutput | None = None
    review: ReviewOutput | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
