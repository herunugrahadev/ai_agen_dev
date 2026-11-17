"""Custom tools for OpenMausBot ExecutorAgent."""

from agent_core.tools.base import BaseTool, ToolRegistry
from agent_core.tools.database_deleter import DatabaseDeleterTool
from agent_core.tools.database_writer import DatabaseWriterTool
from agent_core.tools.email_sender import EmailSenderTool
from agent_core.tools.web_scraper import WebScraperTool

__all__ = [
    "BaseTool",
    "DatabaseDeleterTool",
    "DatabaseWriterTool",
    "EmailSenderTool",
    "ToolRegistry",
    "WebScraperTool",
]
