"""Tests for Human-in-the-Loop (HITL) approval gating, Redis state suspension, and webhook resume."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionUsage
from sqlalchemy.ext.asyncio import create_async_engine

from agent_core.config import Settings
from agent_core.llm_client import NineRouterClient
from agent_core.models import TaskRequest, TaskStatus
from agent_core.storage.postgres_store import PostgresStore
from agent_core.storage.redis_store import RedisStore
from agent_core.tools.base import ToolRegistry
from agent_core.tools.database_deleter import DatabaseDeleterTool
from agent_core.tools.email_sender import EmailSenderTool
from agent_core.workflow.orchestrator import WorkflowOrchestrator
from api.main import app


class InMemoryAsyncRedis:
    """Minimal async Redis stub supporting setex, get, delete, ping, aclose."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def ping(self) -> bool:
        return True

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self.data[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def delete(self, key: str) -> int:
        return 1 if self.data.pop(key, None) is not None else 0

    async def aclose(self) -> None:
        return None


def _make_completion(
    *,
    model: str,
    content: str | None = None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
    tokens: int = 40,
) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-hitl",
        created=1700000000,
        model=model,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls" if tool_calls else "stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=tokens // 2,
            completion_tokens=tokens - (tokens // 2),
            total_tokens=tokens,
        ),
    )


@pytest.mark.asyncio
async def test_hitl_approve_suspends_in_redis_and_resumes_via_webhook() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    pg_store = PostgresStore(engine=engine)
    await pg_store.init_db()

    fake_redis = InMemoryAsyncRedis()
    redis_store = RedisStore(client=fake_redis)  # type: ignore[arg-type]

    planner_resp = _make_completion(
        model="agent-planner",
        content=json.dumps({
            "goal_summary": "Send summary email to external partner",
            "reasoning": "Invoke email_sender tool",
            "subtasks": [
                {
                    "step_id": 1,
                    "title": "Send email",
                    "description": "Email report to partner@example.com",
                    "recommended_tool": "email_sender",
                    "tool_arguments": {"recipient": "partner@example.com"},
                }
            ],
        }),
    )
    exec_tool_call = _make_completion(
        model="agent-executor",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_email_1",
                type="function",
                function=Function(
                    name="email_sender",
                    arguments=json.dumps({
                        "recipient": "partner@example.com",
                        "subject": "Weekly Report",
                        "body": "All systems nominal.",
                    }),
                ),
            )
        ],
    )
    exec_done = _make_completion(
        model="agent-executor",
        content=json.dumps({
            "summary": "Email sent to partner@example.com",
            "extracted_data": {"recipient": "partner@example.com"},
        }),
    )
    reviewer_resp = _make_completion(
        model="agent-reviewer",
        content=json.dumps({
            "approved": True,
            "quality_score": 98,
            "feedback": "Email dispatched after human approval.",
            "issues_found": [],
            "final_response": "Email successfully sent to partner@example.com.",
        }),
    )

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[planner_resp, exec_tool_call, exec_done, reviewer_resp]
    )

    settings = Settings(nine_router_base_url="http://9router:20128/v1")
    llm_client = NineRouterClient(settings=settings, client=mock_openai)
    orchestrator = WorkflowOrchestrator(
        llm_client=llm_client,
        postgres_store=pg_store,
        redis_store=redis_store,
        tool_registry=ToolRegistry(tools=[EmailSenderTool()]),
    )

    task = TaskRequest(
        task_id="hitl-approve-001",
        objective="Send weekly report email to partner@example.com",
    )

    # 1. Run until HITL suspension
    suspended_result = await orchestrator.run(task)
    assert suspended_result.status == TaskStatus.AWAITING_APPROVAL
    assert suspended_result.pending_approval is not None
    approval_id = suspended_result.pending_approval.approval_id
    assert suspended_result.pending_approval.tool_name == "email_sender"

    # Verify suspended state is safely stored in Redis
    assert f"openmausbot:approval:{approval_id}" in fake_redis.data

    # 2. Wire FastAPI app.state and test GET + POST approval webhook
    app.state.settings = settings
    app.state.postgres = pg_store
    app.state.redis = redis_store
    app.state.orchestrator = orchestrator

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        inspect_resp = await client.get(f"/api/v1/approvals/{approval_id}")
        assert inspect_resp.status_code == 200
        assert inspect_resp.json()["tool_name"] == "email_sender"

        approve_resp = await client.post(
            f"/api/v1/webhook/approvals/{approval_id}",
            json={"decision": "approve", "reviewer": "alice", "comment": "LGTM"},
        )
        assert approve_resp.status_code == 200
        final_data = approve_resp.json()

    assert final_data["status"] == TaskStatus.COMPLETED.value
    assert final_data["pending_approval"]["status"] == "approved"
    assert final_data["pending_approval"]["decided_by"] == "alice"
    assert final_data["execution"]["tool_executions"][0]["tool_name"] == "email_sender"
    assert final_data["execution"]["tool_executions"][0]["success"] is True
    assert final_data["review"]["approved"] is True

    # Redis pending key should be cleaned up after completion
    assert f"openmausbot:approval:{approval_id}" not in fake_redis.data

    await pg_store.close()


@pytest.mark.asyncio
async def test_hitl_reject_prevents_destructive_database_delete() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    pg_store = PostgresStore(engine=engine)
    await pg_store.init_db()

    # Seed a record that must NOT be deleted when rejected
    await pg_store.insert_extracted_record(
        task_id="protected-task",
        title="Important Record",
        payload={"critical": True},
    )

    fake_redis = InMemoryAsyncRedis()
    redis_store = RedisStore(client=fake_redis)  # type: ignore[arg-type]

    planner_resp = _make_completion(
        model="agent-planner",
        content=json.dumps({
            "goal_summary": "Delete protected records",
            "reasoning": "Use database_deleter",
            "subtasks": [
                {
                    "step_id": 1,
                    "title": "Delete records",
                    "description": "Delete records for protected-task",
                    "recommended_tool": "database_deleter",
                    "tool_arguments": {"target_task_id": "protected-task"},
                }
            ],
        }),
    )
    exec_tool_call = _make_completion(
        model="agent-executor",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_del_1",
                type="function",
                function=Function(
                    name="database_deleter",
                    arguments=json.dumps({"target_task_id": "protected-task"}),
                ),
            )
        ],
    )

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[planner_resp, exec_tool_call]
    )

    settings = Settings(nine_router_base_url="http://9router:20128/v1")
    llm_client = NineRouterClient(settings=settings, client=mock_openai)
    orchestrator = WorkflowOrchestrator(
        llm_client=llm_client,
        postgres_store=pg_store,
        redis_store=redis_store,
        tool_registry=ToolRegistry(tools=[DatabaseDeleterTool(store=pg_store)]),
    )

    task = TaskRequest(
        task_id="hitl-reject-001",
        objective="Delete records for protected-task",
    )

    suspended_result = await orchestrator.run(task)
    assert suspended_result.status == TaskStatus.AWAITING_APPROVAL
    approval_id = suspended_result.pending_approval.approval_id  # type: ignore[union-attr]

    app.state.settings = settings
    app.state.postgres = pg_store
    app.state.redis = redis_store
    app.state.orchestrator = orchestrator

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        reject_resp = await client.post(
            f"/api/v1/webhook/approvals/{approval_id}",
            json={
                "decision": "reject",
                "reviewer": "bob",
                "comment": "Protected production data",
            },
        )
        assert reject_resp.status_code == 200
        rejected_data = reject_resp.json()

    assert rejected_data["status"] == TaskStatus.REJECTED.value
    assert rejected_data["pending_approval"]["status"] == "rejected"
    assert rejected_data["pending_approval"]["decided_by"] == "bob"

    # Protected record MUST still exist in PostgreSQL
    records = await pg_store.get_extracted_records("protected-task")
    assert len(records) == 1
    assert records[0]["title"] == "Important Record"

    await pg_store.close()
