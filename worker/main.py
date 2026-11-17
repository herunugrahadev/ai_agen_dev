"""OpenMausBot Worker: consumes tasks from Redis and executes multi-agent workflows."""

import asyncio
import signal

from agent_core.config import get_settings
from agent_core.llm_client import NineRouterClient
from agent_core.logging_utils import get_logger, setup_logging
from agent_core.storage.postgres_store import PostgresStore
from agent_core.storage.redis_store import RedisStore
from agent_core.workflow.orchestrator import WorkflowOrchestrator

logger = get_logger("openmausbot.worker")


async def run_worker() -> None:
    """Main worker loop with graceful shutdown on SIGINT/SIGTERM."""
    settings = get_settings()
    setup_logging(settings.log_level)

    postgres_store = PostgresStore(settings=settings)
    await postgres_store.init_db()
    redis_store = RedisStore(settings=settings)
    llm_client = NineRouterClient(settings=settings)
    orchestrator = WorkflowOrchestrator(
        llm_client=llm_client,
        postgres_store=postgres_store,
        redis_store=redis_store,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal() -> None:
        logger.info("Shutdown signal received", extra={"event": "worker_stopping"})
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    logger.info(
        "OpenMausBot Worker started (9Router=%s)",
        settings.nine_router_base_url,
        extra={"event": "worker_started"},
    )

    try:
        while not stop_event.is_set():
            task = await redis_store.dequeue_task(
                timeout_seconds=settings.worker_poll_timeout_seconds
            )
            if task is None:
                continue

            logger.info(
                "Dequeued task %s",
                task.task_id,
                extra={"event": "worker_task_dequeued", "task_id": task.task_id},
            )
            await orchestrator.run(task)
    finally:
        await llm_client.close()
        await redis_store.close()
        await postgres_store.close()
        logger.info("OpenMausBot Worker stopped cleanly", extra={"event": "worker_stopped"})


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
