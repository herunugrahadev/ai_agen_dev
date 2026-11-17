"""Base agent abstraction for OpenMausBot."""

from abc import ABC

from agent_core.llm_client import NineRouterClient


class BaseAgent(ABC):
    """Base class providing shared 9Router client and identity metadata."""

    def __init__(
        self,
        *,
        agent_id: str,
        model_alias: str,
        llm_client: NineRouterClient,
    ) -> None:
        self.agent_id = agent_id
        self.model_alias = model_alias
        self.llm_client = llm_client
