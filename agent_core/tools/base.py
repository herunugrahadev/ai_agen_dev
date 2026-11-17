"""Base tool abstraction and OpenAI-compatible ToolRegistry."""

import time
from abc import ABC, abstractmethod
from typing import Any

from agent_core.models import ToolExecutionRecord


class BaseTool(ABC):
    """Abstract base class for OpenMausBot custom tools."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    requires_approval: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert tool definition to OpenAI Chat Completions tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    @abstractmethod
    async def execute(self, *, task_id: str, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool and return a JSON-serializable dictionary."""


class ToolRegistry:
    """Registry that manages tools and dispatches tool calls safely."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance by name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def is_approval_required(
        self,
        name: str,
        destructive_names: set[str] | None = None,
    ) -> bool:
        """Return True if the tool is marked destructive or requires HITL approval."""
        tool = self.get(name)
        if tool is not None and tool.requires_approval:
            return True
        return bool(destructive_names and name in destructive_names)

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI tool schemas for all registered tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute_tool(
        self,
        *,
        task_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionRecord:
        """Invoke a registered tool and capture latency, success status, and output."""
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
            result = await tool.execute(task_id=task_id, **arguments)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolExecutionRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=True,
                output=result,
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
