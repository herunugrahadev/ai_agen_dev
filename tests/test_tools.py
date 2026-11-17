"""Unit tests for custom tools: web_scraper and database_writer."""

import httpx
import pytest
import respx
from sqlalchemy.ext.asyncio import create_async_engine

from agent_core.storage.postgres_store import PostgresStore
from agent_core.tools.database_writer import DatabaseWriterTool
from agent_core.tools.web_scraper import WebScraperTool


@pytest.mark.asyncio
@respx.mock
async def test_web_scraper_extracts_html_content() -> None:
    html = """
    <html>
      <head>
        <title>OpenMausBot Docs</title>
        <meta name="description" content="Autonomous AI Agent Platform" />
      </head>
      <body>
        <h1>Architecture</h1>
        <h2>9Router Gateway</h2>
        <p>Routes LLM calls seamlessly.</p>
        <script>console.log("ignore me")</script>
      </body>
    </html>
    """
    respx.get("https://example.com/docs").mock(
        return_value=httpx.Response(200, text=html)
    )

    tool = WebScraperTool(timeout_seconds=5.0, trust_env=False)
    result = await tool.execute(task_id="task-1", url="https://example.com/docs")

    assert result["title"] == "OpenMausBot Docs"
    assert result["meta_description"] == "Autonomous AI Agent Platform"
    assert "Architecture" in result["headings"]
    assert "9Router Gateway" in result["headings"]
    assert "Routes LLM calls seamlessly." in result["content_excerpt"]
    assert "ignore me" not in result["content_excerpt"]


@pytest.mark.asyncio
async def test_web_scraper_rejects_invalid_scheme() -> None:
    tool = WebScraperTool()
    with pytest.raises(ValueError, match="only http/https URLs are supported"):
        await tool.execute(task_id="task-1", url="file:///etc/passwd")


@pytest.mark.asyncio
async def test_database_writer_persists_record() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    store = PostgresStore(engine=engine)
    await store.init_db()

    tool = DatabaseWriterTool(store=store)
    res = await tool.execute(
        task_id="task-db-1",
        title="Example Summary",
        category="web_summary",
        source_url="https://example.com/docs",
        payload={"key_points": ["9Router", "OpenMausBot"]},
    )

    assert res["status"] == "persisted"
    assert res["record_id"] == 1

    records = await store.get_extracted_records("task-db-1")
    assert len(records) == 1
    assert records[0]["title"] == "Example Summary"
    assert records[0]["payload"] == {"key_points": ["9Router", "OpenMausBot"]}

    await store.close()
