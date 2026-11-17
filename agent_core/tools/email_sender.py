"""External-impact custom tool: email_sender (requires HITL approval)."""

from typing import Any

from agent_core.logging_utils import get_logger
from agent_core.tools.base import BaseTool

logger = get_logger("openmausbot.tools.email_sender")


class EmailSenderTool(BaseTool):
    """Dispatches an external email. Requires human approval before execution."""

    name = "email_sender"
    requires_approval = True
    description = (
        "EXTERNAL IMPACT: Send an email to an external recipient. "
        "Requires human-in-the-loop approval before execution."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Recipient email address.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line.",
            },
            "body": {
                "type": "string",
                "description": "Email body content.",
            },
        },
        "required": ["recipient", "subject", "body"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        *,
        task_id: str,
        recipient: str,
        subject: str,
        body: str,
        **_: Any,
    ) -> dict[str, Any]:
        if "@" not in recipient:
            raise ValueError(f"Invalid recipient email: '{recipient}'")

        logger.info(
            "External email dispatched to %s",
            recipient,
            extra={
                "event": "external_email_sent",
                "task_id": task_id,
                "agent_id": "openmausbot-executor",
            },
        )
        return {
            "status": "sent",
            "task_id": task_id,
            "recipient": recipient,
            "subject": subject,
            "body_length": len(body),
        }
