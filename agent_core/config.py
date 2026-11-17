"""Centralized configuration for OpenMausBot using Pydantic Settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="openmausbot-platform")
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # 9Router Unified AI Gateway (Decoupled Inference — no direct provider keys here)
    nine_router_base_url: str = Field(default="http://9router:20128/v1")
    nine_router_api_key: str = Field(default="9router-local-gateway-key")
    nine_router_timeout_seconds: float = Field(default=60.0, gt=0)
    nine_router_max_retries: int = Field(default=3, ge=0, le=10)
    nine_router_retry_base_delay: float = Field(default=0.5, gt=0)

    # Virtual Model Aliases routed by 9Router
    model_alias_planner: str = Field(default="agent-planner")
    model_alias_executor: str = Field(default="agent-executor")
    model_alias_reviewer: str = Field(default="agent-reviewer")

    # Redis (Task Queue & Short-Term Memory)
    redis_enabled: bool = Field(default=True)
    redis_url: str = Field(default="redis://redis:6379/0")
    redis_task_queue_key: str = Field(default="openmausbot:tasks:queue")
    redis_memory_ttl_seconds: int = Field(default=3600, ge=60)
    redis_approval_ttl_seconds: int = Field(default=86400, ge=60)

    # Human-in-the-Loop (HITL)
    destructive_tools: list[str] = Field(
        default_factory=lambda: ["database_deleter", "email_sender"]
    )

    # PostgreSQL (Persistent State & Execution History)
    database_url: str = Field(
        default="postgresql+asyncpg://openmausbot:openmausbot_secret_change_me@postgres:5432/openmausbot_db"
    )

    # API & Webhook
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    webhook_secret: str = Field(default="replace-with-hmac-webhook-secret")

    # Worker & Tools
    worker_poll_timeout_seconds: int = Field(default=5, ge=1)
    executor_max_tool_iterations: int = Field(default=6, ge=1, le=20)
    scraper_timeout_seconds: float = Field(default=15.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
