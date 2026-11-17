"""Modular ToolRegistry with strict Pydantic validation and built-in tools."""

import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from agent_core.models import ToolExecutionRecord
from agent_core.tools.base import BaseTool, ToolRegistry


class ValidatedTool(BaseTool):
    """BaseTool subclass that enforces strict Pydantic input validation."""

    input_model: type[BaseModel]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> BaseModel:
        return self.input_model.model_validate(arguments)


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=2, description="Search query string")
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchTool(ValidatedTool):
    """Built-in tool: web_search for fetching internet search results."""

    name = "web_search"
    description = "Search the internet for up-to-date information, news, or facts."
    input_model = WebSearchInput
    parameters_schema = WebSearchInput.model_json_schema()

    def __init__(self, timeout_seconds: float = 10.0, trust_env: bool = False) -> None:
        self.timeout_seconds = timeout_seconds
        self.trust_env = trust_env

    async def execute(self, *, task_id: str, **kwargs: Any) -> dict[str, Any]:
        validated = WebSearchInput.model_validate(kwargs)
        results: list[dict[str, str]] = []

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            trust_env=self.trust_env,
        ) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": validated.query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading") or validated.query,
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL") or "",
                })

            for topic in data.get("RelatedTopics", [])[: validated.max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL") or "",
                    })

            # Fallback to DuckDuckGo HTML lite if Instant Answer returned no results
            if not results:
                html_resp = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": validated.query},
                    headers={"User-Agent": "OpenMausBot/0.1"},
                )
                if html_resp.status_code == 200:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(html_resp.text, "html.parser")
                    for item in soup.select(".result")[: validated.max_results]:
                        title_el = item.select_one(".result__a")
                        snippet_el = item.select_one(".result__snippet")
                        if title_el:
                            results.append({
                                "title": title_el.get_text(" ", strip=True),
                                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
                                "url": str(title_el.get("href") or ""),
                            })

        return {
            "task_id": task_id,
            "query": validated.query,
            "result_count": len(results[: validated.max_results]),
            "results": results[: validated.max_results],
        }


class FileOperatorInput(BaseModel):
    action: Literal["read", "write", "list"]
    path: str = Field(default=".", description="Relative path inside the workspace directory")
    content: str | None = Field(default=None, description="Content to write when action='write'")


class FileOperatorTool(ValidatedTool):
    """Built-in tool: file_operator for safe read/write/list operations in workspace."""

    name = "file_operator"
    description = "Read, write, or list files inside the isolated OpenMausBot workspace."
    input_model = FileOperatorInput
    parameters_schema = FileOperatorInput.model_json_schema()

    def __init__(self, workspace_dir: str | Path = "./workspace") -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, rel_path: str) -> Path:
        candidate = (self.workspace_dir / rel_path).resolve()
        if candidate != self.workspace_dir and self.workspace_dir not in candidate.parents:
            raise ValueError(f"Path '{rel_path}' escapes workspace boundary.")
        return candidate

    async def execute(self, *, task_id: str, **kwargs: Any) -> dict[str, Any]:
        validated = FileOperatorInput.model_validate(kwargs)
        target = self._resolve_safe_path(validated.path)

        if validated.action == "list":
            if not target.exists():
                return {"task_id": task_id, "action": "list", "entries": []}
            entries = sorted(p.name for p in target.iterdir())
            return {"task_id": task_id, "action": "list", "entries": entries}

        if validated.action == "read":
            if not target.is_file():
                raise FileNotFoundError(f"File '{validated.path}' does not exist.")
            text = target.read_text(encoding="utf-8")
            return {
                "task_id": task_id,
                "action": "read",
                "path": validated.path,
                "content": text,
            }

        # action == "write"
        if validated.content is None:
            raise ValueError("'content' is required when action='write'.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(validated.content, encoding="utf-8")
        return {
            "task_id": task_id,
            "action": "write",
            "path": validated.path,
            "bytes_written": len(validated.content.encode("utf-8")),
        }


class WebhookDispatcherInput(BaseModel):
    url: HttpUrl
    payload: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class WebhookDispatcherTool(ValidatedTool):
    """Built-in tool: webhook_dispatcher for sending results to external APIs/webhooks."""

    name = "webhook_dispatcher"
    description = "Dispatch a JSON payload via HTTP POST to an external webhook or notification API."
    input_model = WebhookDispatcherInput
    parameters_schema = WebhookDispatcherInput.model_json_schema()

    def __init__(self, timeout_seconds: float = 10.0, trust_env: bool = True) -> None:
        self.timeout_seconds = timeout_seconds
        self.trust_env = trust_env

    async def execute(self, *, task_id: str, **kwargs: Any) -> dict[str, Any]:
        validated = WebhookDispatcherInput.model_validate(kwargs)
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            trust_env=self.trust_env,
        ) as client:
            resp = await client.post(
                str(validated.url),
                json={"task_id": task_id, **validated.payload},
                headers=validated.headers,
            )
            resp.raise_for_status()
        return {
            "task_id": task_id,
            "url": str(validated.url),
            "status_code": resp.status_code,
            "dispatched": True,
        }


class StrictToolRegistry(ToolRegistry):
    """ToolRegistry that performs strict Pydantic validation before invoking tools."""

    async def execute_tool(
        self,
        *,
        task_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionRecord:
        start_time = time.perf_counter()
        tool = self.get(tool_name)
        if tool is None:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                output={"error": f"Unknown tool '{tool_name}'"},
                latency_ms=latency_ms,
            )

        try:
            if isinstance(tool, ValidatedTool):
                validated = tool.validate_arguments(arguments)
                result = await tool.execute(task_id=task_id, **validated.model_dump())
            else:
                result = await tool.execute(task_id=task_id, **arguments)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=True,
                output=result,
                latency_ms=latency_ms,
            )
        except ValidationError as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                output={"error": f"Pydantic validation failed: {exc.errors()}"},
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                output={"error": str(exc)},
                latency_ms=latency_ms,
            )


def build_default_instruction_registry(
    *,
    workspace_dir: str | Path = "./workspace",
    trust_env: bool = False,
) -> StrictToolRegistry:
    """Create a StrictToolRegistry pre-loaded with web_search, file_operator, and webhook_dispatcher."""
    return StrictToolRegistry(
        tools=[
            WebSearchTool(trust_env=trust_env),
            FileOperatorTool(workspace_dir=workspace_dir),
            WebhookDispatcherTool(trust_env=trust_env),
        ]
    )
