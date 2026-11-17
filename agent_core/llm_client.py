"""Decoupled LLM client routing all OpenMausBot inference through 9Router."""

import asyncio
import random
import time
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

from agent_core.config import Settings, get_settings
from agent_core.logging_utils import get_logger
from agent_core.models import TokenUsage

logger = get_logger("openmausbot.llm_client")

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class NineRouterClient:
    """Async OpenAI-compatible client configured exclusively for 9Router."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or AsyncOpenAI(
            base_url=self.settings.nine_router_base_url,
            api_key=self.settings.nine_router_api_key,
            timeout=self.settings.nine_router_timeout_seconds,
            http_client=httpx.AsyncClient(
                timeout=self.settings.nine_router_timeout_seconds,
                trust_env=False,
            ),
            # Retries are managed explicitly with jitter & observability logging
            max_retries=0,
        )

    async def chat_completion(
        self,
        *,
        model_alias: str,
        messages: list[dict[str, Any]],
        agent_id: str,
        task_id: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[ChatCompletion, TokenUsage]:
        """Execute a chat completion via 9Router with exponential backoff & observability."""
        max_attempts = self.settings.nine_router_max_retries + 1
        base_delay = self.settings.nine_router_retry_base_delay

        extra_headers = {
            "X-Agent-ID": agent_id,
            "X-Task-ID": task_id,
            "X-Agent-Tier": model_alias,
        }

        request_kwargs: dict[str, Any] = {
            "model": model_alias,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": extra_headers,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = tool_choice or "auto"
        if response_format:
            request_kwargs["response_format"] = response_format

        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            start_time = time.perf_counter()
            try:
                response: ChatCompletion = await self._client.chat.completions.create(
                    **request_kwargs
                )
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                )
                routed_model = response.model or model_alias

                logger.info(
                    "9Router inference succeeded",
                    extra={
                        "event": "llm_inference_success",
                        "agent_id": agent_id,
                        "task_id": task_id,
                        "routed_model": routed_model,
                        "latency_ms": latency_ms,
                        "token_usage": usage.model_dump(),
                    },
                )
                return response, usage

            except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                status_code = getattr(exc, "status_code", None)
                is_retryable = (
                    isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError))
                    or (status_code in RETRYABLE_STATUS_CODES)
                )
                last_exc = exc

                logger.warning(
                    "9Router inference error (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    str(exc),
                    extra={
                        "event": "llm_inference_retry" if is_retryable else "llm_inference_error",
                        "agent_id": agent_id,
                        "task_id": task_id,
                        "routed_model": model_alias,
                        "latency_ms": latency_ms,
                        "status": status_code or "network_error",
                    },
                )

                if not is_retryable or attempt >= max_attempts:
                    raise

                # Exponential backoff with full jitter
                delay = min(base_delay * (2 ** (attempt - 1)), 15.0)
                jittered_delay = delay * random.uniform(0.5, 1.0)
                await asyncio.sleep(jittered_delay)

        assert last_exc is not None
        raise last_exc

    async def close(self) -> None:
        """Close underlying HTTP client."""
        await self._client.close()
