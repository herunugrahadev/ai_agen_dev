"""Tests for User Instruction Execution Engine: DAG Planner, StrictToolRegistry, ReAct, and SSE."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from agent_core.agents.executor import ExecutorAgent
from agent_core.agents.planner import PlannerAgent
from agent_core.config import Settings
from agent_core.llm_client import NineRouterClient
from agent_core.models import InstructionRequest
from agent_core.tools.tool_registry import (
    FileOperatorTool,
    StrictToolRegistry,
    WebhookDispatcherTool,
    WebSearchTool,
)
from api.main import app


def _make_completion(model: str, content: str, tokens: int = 30) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-engine",
        created=1700000000,
        model=model,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=tokens // 2,
            completion_tokens=tokens - (tokens // 2),
            total_tokens=tokens,
        ),
    )


@pytest.mark.asyncio
async def test_file_operator_and_strict_validation(tmp_path: Path) -> None:
    registry = StrictToolRegistry(tools=[FileOperatorTool(workspace_dir=tmp_path)])

    # Invalid action should fail Pydantic validation cleanly
    bad_rec = await registry.execute_tool(
        task_id="t1",
        tool_name="file_operator",
        arguments={"action": "delete_everything", "path": "notes.txt"},
    )
    assert bad_rec.success is False
    assert "Pydantic validation failed" in str(bad_rec.output)

    # Path traversal must be rejected
    trav_rec = await registry.execute_tool(
        task_id="t1",
        tool_name="file_operator",
        arguments={"action": "read", "path": "../../etc/passwd"},
    )
    assert trav_rec.success is False
    assert "escapes workspace boundary" in str(trav_rec.output)

    # Valid write & read
    write_rec = await registry.execute_tool(
        task_id="t1",
        tool_name="file_operator",
        arguments={"action": "write", "path": "summary.txt", "content": "Tech news"},
    )
    assert write_rec.success is True
    read_rec = await registry.execute_tool(
        task_id="t1",
        tool_name="file_operator",
        arguments={"action": "read", "path": "summary.txt"},
    )
    assert read_rec.success is True
    assert read_rec.output["content"] == "Tech news"


@pytest.mark.asyncio
async def test_planner_returns_needs_clarification_for_missing_params() -> None:
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_make_completion(
            model="agent-planner",
            content=json.dumps({
                "status": "needs_clarification",
                "goal_summary": "Cari 5 berita teknologi lalu kirim ke email",
                "clarification_questions": [
                    "Alamat email tujuan Anda apa?"
                ],
                "steps": [],
            }),
        )
    )
    llm_client = NineRouterClient(
        settings=Settings(nine_router_base_url="http://9router:20128/v1"),
        client=mock_openai,
    )
    planner = PlannerAgent(llm_client=llm_client)

    plan, _ = await planner.parse_instruction(
        InstructionRequest(
            task_id="clarify-1",
            instruction="Carikan 5 berita teknologi terbaru hari ini, rangkum, lalu kirim ke email saya",
        ),
        available_tools=[],
    )
    assert plan.status == "needs_clarification"
    assert len(plan.clarification_questions) == 1
    assert "email" in plan.clarification_questions[0].lower()


@pytest.mark.asyncio
@respx.mock
async def test_react_self_correction_and_sse_execute_endpoint(tmp_path: Path) -> None:
    respx.get("https://api.duckduckgo.com/").mock(
        return_value=httpx.Response(
            200,
            json={
                "Heading": "AI News",
                "AbstractText": "OpenMausBot + 9Router launch update.",
                "AbstractURL": "https://example.com/news",
                "RelatedTopics": [],
            },
        )
    )
    respx.post("https://hooks.example.com/notify").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    # 1. Planner returns 3-step DAG, where step 2 intentionally has invalid action="save"
    #    so ExecutorAgent's ReAct loop triggers self-correction -> action="write"!
    planner_resp = _make_completion(
        model="agent-planner",
        content=json.dumps({
            "status": "ready",
            "goal_summary": "Search AI news, save to file, and dispatch webhook",
            "clarification_questions": [],
            "steps": [
                {
                    "step_id": 1,
                    "title": "Search AI news",
                    "description": "Search DuckDuckGo",
                    "tool_name": "web_search",
                    "arguments": {"query": "AI News", "max_results": 3},
                    "depends_on": [],
                },
                {
                    "step_id": 2,
                    "title": "Write summary file",
                    "description": "Save summary to workspace",
                    "tool_name": "file_operator",
                    "arguments": {"action": "save", "path": "news.txt", "content": "AI News summary"},
                    "depends_on": [1],
                },
                {
                    "step_id": 3,
                    "title": "Send webhook",
                    "description": "Notify external webhook",
                    "tool_name": "webhook_dispatcher",
                    "arguments": {
                        "url": "https://hooks.example.com/notify",
                        "payload": {"status": "done"},
                    },
                    "depends_on": [2],
                },
            ],
        }),
    )

    # 2. Executor self-correction response for step 2
    self_correction_resp = _make_completion(
        model="agent-executor",
        content=json.dumps({
            "thought": "Action 'save' is invalid; valid enum value is 'write'.",
            "arguments": {
                "action": "write",
                "path": "news.txt",
                "content": "AI News summary",
            },
        }),
    )

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[planner_resp, self_correction_resp]
    )

    settings = Settings(nine_router_base_url="http://9router:20128/v1")
    llm_client = NineRouterClient(settings=settings, client=mock_openai)
    registry = StrictToolRegistry(
        tools=[
            WebSearchTool(trust_env=False),
            FileOperatorTool(workspace_dir=tmp_path),
            WebhookDispatcherTool(trust_env=False),
        ]
    )

    app.state.settings = settings
    app.state.llm_client = llm_client
    app.state.instruction_registry = registry

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/v1/execute",
            json={
                "task_id": "sse-exec-001",
                "instruction": "Cari berita AI, simpan ke news.txt, lalu kirim ke https://hooks.example.com/notify",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text

    assert "event: status" in body
    assert "event: plan_ready" in body
    assert "event: retry" in body
    assert "event: completed" in body
    assert (tmp_path / "news.txt").read_text(encoding="utf-8") == "AI News summary"
