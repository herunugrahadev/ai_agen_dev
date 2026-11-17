"""End-to-end workflow test: Trigger -> Planner -> Executor (tools) -> Reviewer."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
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
from agent_core.workflow.orchestrator import WorkflowOrchestrator


def _make_completion(
    *,
    model: str,
    content: str | None = None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
    tokens: int = 50,
) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-e2e",
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
@respx.mock
async def test_end_to_end_workflow_planner_executor_reviewer() -> None:
    respx.get("https://example.com/product").mock(
        return_value=httpx.Response(
            200,
            text="<html><head><title>AI Router Pro</title></head><body><h1>9Router</h1><p>Latency 12ms</p></body></html>",
        )
    )

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    store = PostgresStore(engine=engine)
    await store.init_db()

    # Sequence of mocked 9Router responses:
    # 1. PlannerAgent -> JSON plan
    # 2. ExecutorAgent (turn 1) -> calls web_scraper
    # 3. ExecutorAgent (turn 2) -> calls database_writer
    # 4. ExecutorAgent (turn 3) -> returns final JSON summary
    # 5. ReviewerAgent -> returns approved QA review JSON
    planner_resp = _make_completion(
        model="agent-planner",
        content=json.dumps({
            "goal_summary": "Scrape product page and store summary in DB",
            "reasoning": "First scrape target URL, then persist extracted metrics.",
            "subtasks": [
                {
                    "step_id": 1,
                    "title": "Scrape target URL",
                    "description": "Fetch https://example.com/product",
                    "recommended_tool": "web_scraper",
                    "tool_arguments": {"url": "https://example.com/product"},
                },
                {
                    "step_id": 2,
                    "title": "Save to PostgreSQL",
                    "description": "Store extracted title and latency",
                    "recommended_tool": "database_writer",
                    "tool_arguments": {"title": "AI Router Pro"},
                },
            ],
        }),
        tokens=100,
    )

    exec_turn_1 = _make_completion(
        model="agent-executor",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="web_scraper",
                    arguments=json.dumps({"url": "https://example.com/product"}),
                ),
            )
        ],
        tokens=60,
    )

    exec_turn_2 = _make_completion(
        model="agent-executor",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_2",
                type="function",
                function=Function(
                    name="database_writer",
                    arguments=json.dumps({
                        "title": "AI Router Pro",
                        "category": "benchmark",
                        "source_url": "https://example.com/product",
                        "payload": {"latency": "12ms"},
                    }),
                ),
            )
        ],
        tokens=60,
    )

    exec_turn_3 = _make_completion(
        model="agent-executor",
        content=json.dumps({
            "summary": "Scraped AI Router Pro and persisted latency=12ms.",
            "extracted_data": {"title": "AI Router Pro", "latency": "12ms"},
        }),
        tokens=40,
    )

    reviewer_resp = _make_completion(
        model="agent-reviewer",
        content=json.dumps({
            "approved": True,
            "quality_score": 96,
            "feedback": "All subtasks executed and persisted accurately.",
            "issues_found": [],
            "final_response": "Successfully scraped AI Router Pro (Latency 12ms) and saved to PostgreSQL.",
        }),
        tokens=80,
    )

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[planner_resp, exec_turn_1, exec_turn_2, exec_turn_3, reviewer_resp]
    )

    from agent_core.tools.base import ToolRegistry
    from agent_core.tools.database_writer import DatabaseWriterTool
    from agent_core.tools.web_scraper import WebScraperTool

    llm_client = NineRouterClient(
        settings=Settings(nine_router_base_url="http://9router:20128/v1"),
        client=mock_openai,
    )
    registry = ToolRegistry(
        tools=[
            WebScraperTool(trust_env=False),
            DatabaseWriterTool(store=store),
        ]
    )
    orchestrator = WorkflowOrchestrator(
        llm_client=llm_client,
        postgres_store=store,
        redis_store=None,
        tool_registry=registry,
    )

    task = TaskRequest(
        task_id="e2e-task-001",
        objective="Scrape https://example.com/product and save latency metrics to database.",
        target_url="https://example.com/product",
    )

    result = await orchestrator.run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.plan is not None
    assert len(result.plan.subtasks) == 2
    assert result.execution is not None
    assert len(result.execution.tool_executions) == 2
    assert result.execution.tool_executions[0].tool_name == "web_scraper"
    assert result.execution.tool_executions[0].success is True
    assert result.execution.tool_executions[1].tool_name == "database_writer"
    assert result.execution.tool_executions[1].success is True
    assert result.review is not None
    assert result.review.approved is True
    assert result.review.quality_score == 96
    assert result.token_usage.total_tokens == 340

    # Verify PostgreSQL persistence
    persisted = await store.get_task_result("e2e-task-001")
    assert persisted is not None
    assert persisted.status == TaskStatus.COMPLETED

    records = await store.get_extracted_records("e2e-task-001")
    assert len(records) == 1
    assert records[0]["payload"] == {"latency": "12ms"}

    await store.close()
