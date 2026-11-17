"""Destructive custom tool: database_deleter (requires HITL approval)."""

from typing import Any

from agent_core.storage.postgres_store import PostgresStore
from agent_core.tools.base import BaseTool


class DatabaseDeleterTool(BaseTool):
    """Deletes extracted records from PostgreSQL. Requires human approval before execution."""

    name = "database_deleter"
    requires_approval = True
    description = (
        "DESTRUCTIVE: Delete extracted records from PostgreSQL by target_task_id or record_id. "
        "Requires human-in-the-loop approval before execution."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target_task_id": {
                "type": "string",
                "description": "Task ID whose extracted records should be deleted.",
            },
            "record_id": {
                "type": "integer",
                "description": "Optional specific record ID to delete.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    async def execute(
        self,
        *,
        task_id: str,
        target_task_id: str | None = None,
        record_id: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        deleted_count = await self._store.delete_extracted_records(
            task_id=target_task_id,
            record_id=record_id,
        )
        return {
            "status": "deleted",
            "deleted_count": deleted_count,
            "target_task_id": target_task_id,
            "record_id": record_id,
            "requested_by_task": task_id,
        }
