"""FastAPI REST API & Webhook Trigger for OpenMausBot."""

import hashlib
import hmac
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from pydantic import BaseModel

from agent_core.config import get_settings
from agent_core.llm_client import NineRouterClient
from agent_core.logging_utils import get_logger, setup_logging
from agent_core.models import (
    TaskRequest,
    TaskStatus,
    WebhookPayload,
    WorkflowResult,
)
from agent_core.storage.postgres_store import PostgresStore
from agent_core.storage.redis_store import RedisStore
from agent_core.workflow.orchestrator import WorkflowOrchestrator

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
