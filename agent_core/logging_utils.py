"""Structured JSON logging utilities for OpenMausBot observability."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON with standard observability fields."""

    STANDARD_ATTRS = frozenset({
        "agent_id",
        "task_id",
        "routed_model",
        "latency_ms",
        "token_usage",
        "event",
        "status",
        "extra_data",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for attr in self.STANDARD_ATTRS:
            if hasattr(record, attr):
                value = getattr(record, attr)
                if value is not None:
                    payload[attr] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with structured JSON output to stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers when called multiple times
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
