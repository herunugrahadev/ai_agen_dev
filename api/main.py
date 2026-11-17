"""FastAPI REST API & Webhook Trigger for OpenMausBot."""

import hashlib
import hmac
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_core.agents.executor import ExecutorAgent
from agent_core.agents.planner import PlannerAgent
from agent_core.config import get_settings
from agent_core.llm_client import NineRouterClient
from agent_core.logging_utils import get_logger, setup_logging
from agent_core.models import (
    ApprovalWebhookPayload,
    ExecutionEvent,
    InstructionRequest,
    PendingApproval,
    TaskRequest,
    TaskStatus,
    WebhookPayload,
    WorkflowResult,
)
from agent_core.storage.postgres_store import PostgresStore
from agent_core.storage.redis_store import RedisStore
from agent_core.tools.tool_registry import build_default_instruction_registry
from agent_core.workflow.orchestrator import (
    ApprovalNotFoundError,
    WorkflowOrchestrator,
)

logger = get_logger("openmausbot.api")


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)

    postgres_store = PostgresStore(settings=settings)
    await postgres_store.init_db()
    redis_store = RedisStore(settings=settings) if settings.redis_enabled else None
    llm_client = NineRouterClient(settings=settings)

    app.state.settings = settings
    app.state.postgres = postgres_store
    app.state.redis = redis_store
    app.state.llm_client = llm_client
    app.state.orchestrator = WorkflowOrchestrator(
        llm_client=llm_client,
        postgres_store=postgres_store,
        redis_store=redis_store,
    )

    logger.info("OpenMausBot API started", extra={"event": "api_startup"})
    try:
        yield
    finally:
        await llm_client.close()
        if redis_store is not None:
            await redis_store.close()
        await postgres_store.close()
        logger.info("OpenMausBot API stopped", extra={"event": "api_shutdown"})


app = FastAPI(
    title="OpenMausBot + 9Router Autonomous Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check(request: Request) -> dict[str, str]:
    """Verify PostgreSQL and Redis readiness."""
    pg_ok = await request.app.state.postgres.ping()
    redis_store = request.app.state.redis
    redis_ok = await redis_store.ping() if redis_store is not None else True
    redis_status = "disabled" if redis_store is None else ("connected" if redis_ok else "error")
    return {
        "status": "ok" if (pg_ok and redis_ok) else "degraded",
        "postgres": "connected" if pg_ok else "error",
        "redis": redis_status,
        "nine_router_url": request.app.state.settings.nine_router_base_url,
    }


@app.post(
    "/api/v1/tasks",
    response_model=TaskSubmitResponse | WorkflowResult,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_task(
    task: TaskRequest,
    request: Request,
    sync: bool = Query(
        default=False,
        description="If true, execute synchronously instead of queueing to Redis worker.",
    ),
) -> TaskSubmitResponse | WorkflowResult:
    """Enqueue an autonomous agent task (or run synchronously with ?sync=true)."""
    if sync or request.app.state.redis is None:
        return await request.app.state.orchestrator.run(task)

    initial_state = WorkflowResult(
        task_id=task.task_id,
        status=TaskStatus.QUEUED,
        objective=task.objective,
    )
    await request.app.state.postgres.save_task_result(initial_state)
    await request.app.state.redis.enqueue_task(task)

    logger.info(
        "Task queued to Redis",
        extra={"event": "task_queued", "task_id": task.task_id},
    )
    return TaskSubmitResponse(
        task_id=task.task_id,
        status=TaskStatus.QUEUED,
        message="Task accepted and queued for OpenMausBot worker.",
    )


@app.post(
    "/api/v1/webhook/trigger",
    response_model=TaskSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def webhook_trigger(
    request: Request,
    payload: WebhookPayload,
    x_webhook_signature: str | None = Header(default=None),
) -> TaskSubmitResponse:
    """Webhook trigger endpoint with optional HMAC-SHA256 signature verification."""
    settings = request.app.state.settings
    if x_webhook_signature is not None:
        raw_body = await request.body()
        expected_sig = hmac.new(
            settings.webhook_secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        provided_sig = x_webhook_signature.removeprefix("sha256=")
        if not hmac.compare_digest(expected_sig, provided_sig):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature.",
            )

    task = TaskRequest(
        objective=payload.objective,
        target_url=payload.target_url,
        metadata={**payload.metadata, "event_type": payload.event_type},
        source="webhook",
    )
    initial_state = WorkflowResult(
        task_id=task.task_id,
        status=TaskStatus.QUEUED,
        objective=task.objective,
    )
    await request.app.state.postgres.save_task_result(initial_state)
    await request.app.state.redis.enqueue_task(task)

    logger.info(
        "Webhook task queued",
        extra={"event": "webhook_task_queued", "task_id": task.task_id},
    )
    return TaskSubmitResponse(
        task_id=task.task_id,
        status=TaskStatus.QUEUED,
        message=f"Webhook event '{payload.event_type}' queued.",
    )


@app.get("/api/v1/tasks/{task_id}", response_model=WorkflowResult)
async def get_task_status(task_id: str, request: Request) -> WorkflowResult:
    """Retrieve task execution state, plan, tool logs, and QA review."""
    result = await request.app.state.postgres.get_task_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    return result


@app.get("/api/v1/tasks", response_model=list[WorkflowResult])
async def list_tasks(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[WorkflowResult]:
    """List recent task executions."""
    return await request.app.state.postgres.list_recent_tasks(limit=limit)


@app.get("/api/v1/tasks/{task_id}/records")
async def get_task_records(task_id: str, request: Request) -> list[dict]:
    """List structured records persisted by the database_writer tool for a task."""
    return await request.app.state.postgres.get_extracted_records(task_id)


@app.get("/api/v1/approvals/{approval_id}", response_model=PendingApproval)
async def get_pending_approval(approval_id: str, request: Request) -> PendingApproval:
    """Inspect a suspended HITL approval stored in Redis."""
    redis_store = request.app.state.redis
    if redis_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is disabled; HITL approval state store is unavailable.",
        )
    state = await redis_store.get_pending_approval_state(approval_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pending approval '{approval_id}' not found or expired.",
        )
    return PendingApproval.model_validate(state["approval"])


@app.post("/api/v1/webhook/approvals/{approval_id}", response_model=WorkflowResult)
async def approval_webhook(
    approval_id: str,
    payload: ApprovalWebhookPayload,
    request: Request,
    x_webhook_signature: str | None = Header(default=None),
) -> WorkflowResult:
    """HITL webhook endpoint to approve or reject a suspended destructive tool call."""
    settings = request.app.state.settings
    if x_webhook_signature is not None:
        raw_body = await request.body()
        expected_sig = hmac.new(
            settings.webhook_secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        provided_sig = x_webhook_signature.removeprefix("sha256=")
        if not hmac.compare_digest(expected_sig, provided_sig):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature.",
            )

    try:
        return await request.app.state.orchestrator.resume_approval(approval_id, payload)
    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.get("/api/v1/metrics/overview")
async def metrics_overview(request: Request) -> dict[str, Any]:
    """Aggregate real-time metrics from OpenMausBot DB/Redis and 9Router API."""
    tasks = await request.app.state.postgres.list_recent_tasks(limit=50)
    redis_store = request.app.state.redis
    queue_depth = await redis_store.get_queue_length() if redis_store is not None else 0

    status_counts: dict[str, int] = {s.value: 0 for s in TaskStatus}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # Estimate per-agent token distribution across completed/in-progress stages
    agent_tokens = {
        "agent-planner": 0,
        "agent-executor": 0,
        "agent-reviewer": 0,
    }

    active_statuses = {
        TaskStatus.QUEUED,
        TaskStatus.PLANNING,
        TaskStatus.EXECUTING,
        TaskStatus.AWAITING_APPROVAL,
        TaskStatus.REVIEWING,
    }
    active_tasks: list[dict[str, Any]] = []

    for t in tasks:
        status_counts[t.status.value] = status_counts.get(t.status.value, 0) + 1
        total_prompt_tokens += t.token_usage.prompt_tokens
        total_completion_tokens += t.token_usage.completion_tokens
        total_tokens += t.token_usage.total_tokens

        # Distribute recorded task tokens proportionally across executed agent stages
        stages_present = []
        if t.plan is not None:
            stages_present.append(("agent-planner", 0.30))
        if t.execution is not None:
            stages_present.append(("agent-executor", 0.45))
        if t.review is not None:
            stages_present.append(("agent-reviewer", 0.25))

        if stages_present and t.token_usage.total_tokens > 0:
            weight_sum = sum(w for _, w in stages_present)
            for agent_name, weight in stages_present:
                agent_tokens[agent_name] += round(
                    t.token_usage.total_tokens * (weight / weight_sum)
                )

        if t.status in active_statuses:
            active_tasks.append(t.model_dump(mode="json"))

    finished_count = (
        status_counts[TaskStatus.COMPLETED.value]
        + status_counts[TaskStatus.FAILED.value]
        + status_counts[TaskStatus.REJECTED.value]
        + status_counts[TaskStatus.NEEDS_REVISION.value]
    )
    success_count = status_counts[TaskStatus.COMPLETED.value]
    routing_success_rate = (
        round((success_count / finished_count) * 100, 1) if finished_count > 0 else 100.0
    )

    # Query 9Router health & metrics endpoint
    base_root = re.sub(r"/v1/?$", "", request.app.state.settings.nine_router_base_url)
    nine_router_status: dict[str, Any] = {
        "reachable": False,
        "base_url": request.app.state.settings.nine_router_base_url,
        "health": None,
        "rate_limit_429_fallbacks": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            health_res = await client.get(f"{base_root}/api/health")
            if health_res.status_code < 400:
                nine_router_status["reachable"] = True
                nine_router_status["health"] = (
                    health_res.json()
                    if "application/json" in health_res.headers.get("content-type", "")
                    else {"status": "ok"}
                )
    except Exception:
        nine_router_status["reachable"] = False

    return {
        "queue_depth": queue_depth,
        "total_tasks": len(tasks),
        "routing_success_rate": routing_success_rate,
        "status_counts": status_counts,
        "token_summary": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        },
        "agent_tokens": [
            {"agent": "agent-planner", "label": "Planner", "tokens": agent_tokens["agent-planner"]},
            {"agent": "agent-executor", "label": "Executor", "tokens": agent_tokens["agent-executor"]},
            {"agent": "agent-reviewer", "label": "Reviewer", "tokens": agent_tokens["agent-reviewer"]},
        ],
        "active_tasks": active_tasks,
        "recent_tasks": [t.model_dump(mode="json") for t in tasks[:15]],
        "nine_router": nine_router_status,
    }


def _format_sse(event: ExecutionEvent) -> str:
    """Format an ExecutionEvent as a standard Server-Sent Event frame."""
    return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"


@app.post("/api/v1/execute")
async def execute_instruction_sse(
    payload: InstructionRequest,
    request: Request,
) -> StreamingResponse:
    """Stream real-time DAG planning and ReAct tool execution via Server-Sent Events (SSE)."""
    llm_client: NineRouterClient = request.app.state.llm_client
    registry = getattr(
        request.app.state,
        "instruction_registry",
        build_default_instruction_registry(),
    )
    planner = PlannerAgent(llm_client=llm_client)
    executor = ExecutorAgent(llm_client=llm_client, tool_registry=registry)

    async def event_stream() -> AsyncIterator[str]:
        try:
            yield _format_sse(
                ExecutionEvent(
                    event="status",
                    task_id=payload.task_id,
                    message="Sedang menyusun rencana...",
                )
            )

            dag_plan, plan_usage = await planner.parse_instruction(
                payload,
                available_tools=registry.openai_schemas(),
            )

            if dag_plan.status == "needs_clarification":
                yield _format_sse(
                    ExecutionEvent(
                        event="clarification_needed",
                        task_id=payload.task_id,
                        message="Instruksi membutuhkan klarifikasi tambahan.",
                        data={
                            "status": TaskStatus.NEEDS_CLARIFICATION.value,
                            "goal_summary": dag_plan.goal_summary,
                            "clarification_questions": dag_plan.clarification_questions,
                            "token_usage": plan_usage.model_dump(),
                        },
                    )
                )
                return

            yield _format_sse(
                ExecutionEvent(
                    event="plan_ready",
                    task_id=payload.task_id,
                    message=f"Rencana siap ({len(dag_plan.steps)} langkah).",
                    data={
                        "plan": dag_plan.model_dump(mode="json"),
                        "token_usage": plan_usage.model_dump(),
                    },
                )
            )

            async for exec_event in executor.execute_dag_react(
                request=payload,
                plan=dag_plan,
                max_retries_per_step=3,
            ):
                yield _format_sse(exec_event)

        except Exception as exc:
            yield _format_sse(
                ExecutionEvent(
                    event="failed",
                    task_id=payload.task_id,
                    message=f"Eksekusi gagal: {exc}",
                )
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
