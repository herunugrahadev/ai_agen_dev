"""Tests for 9Router configuration schema and NineRouterClient 429 retry/backoff."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yaml
from openai import RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from agent_core.config import Settings
from agent_core.llm_client import NineRouterClient


def test_9router_config_tiers_and_fallbacks() -> None:
    config_path = Path(__file__).resolve().parent.parent / "9router" / "9router-config.yaml"
    assert config_path.exists()

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    combos = {combo["name"]: combo for combo in data["combos"]}

    assert {"agent-planner", "agent-executor", "agent-reviewer"} <= set(combos.keys())

    for tier_name in ("agent-planner", "agent-executor", "agent-reviewer"):
        models = combos[tier_name]["models"]
        assert len(models) >= 3, f"{tier_name} must define fallback models"
        # Ensure fallback models handle HTTP 429
        fallback_models = models[1:]
        assert any(429 in m.get("fallback_on", []) for m in fallback_models)

    assert 429 in data["resilience"]["retry_on_status"]
    assert data["resilience"]["rate_limit_rotation"]["enabled"] is True


@pytest.mark.asyncio
async def test_nine_router_client_retries_on_429() -> None:
    mock_openai = MagicMock()
    request = httpx.Request("POST", "http://9router:20128/v1/chat/completions")
    response_429 = httpx.Response(429, request=request)
    rate_limit_exc = RateLimitError(
        message="Rate limit exceeded",
        response=response_429,
        body={"error": "rate_limit"},
    )

    success_completion = ChatCompletion(
        id="chatcmpl-test",
        created=1700000000,
        model="agent-executor",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content='{"ok": true}'),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    mock_create = AsyncMock(side_effect=[rate_limit_exc, success_completion])
    mock_openai.chat.completions.create = mock_create

    settings = Settings(
        nine_router_base_url="http://9router:20128/v1",
        nine_router_max_retries=2,
        nine_router_retry_base_delay=0.01,
    )
    client = NineRouterClient(settings=settings, client=mock_openai)

    completion, usage = await client.chat_completion(
        model_alias="agent-executor",
        messages=[{"role": "user", "content": "ping"}],
        agent_id="test-agent",
        task_id="task-429-test",
    )

    assert mock_create.await_count == 2
    assert completion.choices[0].message.content == '{"ok": true}'
    assert usage.total_tokens == 15
