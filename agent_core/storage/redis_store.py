"""Redis adapter for task queueing and short-term agent working memory."""

import json
from typing import Any

import redis.asyncio as aioredis

from agent_core.config import Settings, get_settings
from agent_core.models import TaskRequest


class RedisStore:
    """Manages task queue (FIFO) and short-term working memory in Redis."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: aioredis.Redis | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or aioredis.from_url(
            self.settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def ping(self) -> bool:
        """Verify Redis connectivity."""
        return bool(await self._client.ping())

    async def enqueue_task(self, task: TaskRequest) -> int:
        """Push a TaskRequest onto the Redis task queue."""
        payload = task.model_dump_json()
        return int(await self._client.lpush(self.settings.redis_task_queue_key, payload))

    async def dequeue_task(self, timeout_seconds: int | None = None) -> TaskRequest | None:
        """Blocking pop of the next TaskRequest from the queue."""
        timeout = timeout_seconds or self.settings.worker_poll_timeout_seconds
        result = await self._client.brpop(self.settings.redis_task_queue_key, timeout=timeout)
        if not result:
            return None
        _, raw_payload = result
        return TaskRequest.model_validate_json(raw_payload)

    async def set_short_term_memory(
        self,
        task_id: str,
        key: str,
        value: dict[str, Any] | list[Any] | str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store ephemeral agent working memory for a task."""
        redis_key = f"openmausbot:memory:{task_id}:{key}"
        ttl = ttl_seconds or self.settings.redis_memory_ttl_seconds
        serialized = json.dumps(value, default=str)
        await self._client.setex(redis_key, ttl, serialized)

    async def get_short_term_memory(
        self,
        task_id: str,
        key: str,
    ) -> dict[str, Any] | list[Any] | str | None:
        """Retrieve ephemeral agent working memory for a task."""
        redis_key = f"openmausbot:memory:{task_id}:{key}"
        raw = await self._client.get(redis_key)
        if raw is None:
            return None
        return json.loads(raw)

    async def save_pending_approval_state(
        self,
        approval_id: str,
        state_payload: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Safely persist suspended agent execution state in Redis while awaiting HITL approval."""
        redis_key = f"openmausbot:approval:{approval_id}"
        ttl = ttl_seconds or self.settings.redis_approval_ttl_seconds
        await self._client.setex(redis_key, ttl, json.dumps(state_payload, default=str))

    async def get_pending_approval_state(
        self,
        approval_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve suspended agent state from Redis by approval_id."""
        redis_key = f"openmausbot:approval:{approval_id}"
        raw = await self._client.get(redis_key)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete_pending_approval_state(self, approval_id: str) -> None:
        """Remove suspended approval state from Redis after resolution."""
        redis_key = f"openmausbot:approval:{approval_id}"
        await self._client.delete(redis_key)

    async def close(self) -> None:
        """Close the Redis connection pool."""
        await self._client.aclose()
