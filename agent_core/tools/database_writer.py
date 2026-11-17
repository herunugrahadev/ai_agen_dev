"""Custom tool: database_writer for persisting extracted structured data to PostgreSQL."""

from typing import Any

from agent_core.storage.postgres_store import PostgresStore
from agent_core.tools.base import BaseTool


class DatabaseWriterTool(BaseTool):
    """Persists structured findings and extracted entities into PostgreSQL."""

    name = "database_writer"
    description = (
        "Persist structured extracted data or analysis results into the PostgreSQL "
        "extracted_records table for durable storage and downstream auditing."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short descriptive title for the stored record.",
            },
            "category": {
                "type": "string",
                "description": "Category label (e.g., 'web_summary', 'pricing', 'news').",
                "default": "general",
            },
            "source_url": {
                "type": "string",
                "description": "Optional source URL where the data originated.",
            },
            "payload": {
                "type": "object",
                "description": "Structured JSON object containing the extracted fields.",
            },
        },
        "required": ["title", "payload"],
        "additionalProperties": False,
    }

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    async def execute(
        self,
        *,
        task_id: str,
        title: str,
        payload: dict[str, Any],
        category: str = "general",
        source_url: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        record_id = await self._store.insert_extracted_record(
            task_id=task_id,
            title=title,
            payload=payload,
            category=category,
            source_url=source_url,
        )
        return {
            "status": "persisted",
            "record_id": record_id,
            "task_id": task_id,
            "title": title,
            "category": category,
        }
