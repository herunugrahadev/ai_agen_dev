"""PostgreSQL async persistence for workflow state, execution history, and extracted data."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Integer, JSON, String, Text, delete, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from agent_core.config import Settings, get_settings
from agent_core.models import (
    ExecutorOutput,
    PendingApproval,
    PlanOutput,
    ReviewOutput,
    TaskStatus,
    TokenUsage,
    WorkflowResult,
)


class Base(DeclarativeBase):
    pass


class TaskExecutionModel(Base):
    """Stores complete lifecycle state and history for each OpenMausBot task."""

    __tablename__ = "task_executions"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    plan_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    execution_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    review_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    pending_approval_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    token_usage_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExtractedRecordModel(Base):
    """Stores structured records written by the database_writer tool."""

    __tablename__ = "extracted_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False, default="general")
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class PostgresStore:
    """Async repository for task execution history and extracted tool records."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._engine = engine or create_async_engine(
            self.settings.database_url,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Create database tables and apply lightweight schema migrations if needed."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            dialect = conn.dialect.name
            if dialect == "sqlite":
                rows = (
                    await conn.execute(text("PRAGMA table_info(task_executions)"))
                ).fetchall()
                existing_cols = {row[1] for row in rows}
                if "pending_approval_json" not in existing_cols:
                    await conn.execute(
                        text("ALTER TABLE task_executions ADD COLUMN pending_approval_json JSON")
                    )
            else:
                await conn.execute(
                    text(
                        "ALTER TABLE task_executions "
                        "ADD COLUMN IF NOT EXISTS pending_approval_json JSON"
                    )
                )

    async def ping(self) -> bool:
        """Verify database connectivity."""
        async with self._session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True

    async def save_task_result(self, result: WorkflowResult) -> None:
        """Insert or update a WorkflowResult record."""
        async with self._session_factory() as session:
            existing = await session.get(TaskExecutionModel, result.task_id)
            pending_json = (
                result.pending_approval.model_dump(mode="json")
                if result.pending_approval
                else None
            )
            if existing is None:
                record = TaskExecutionModel(
                    task_id=result.task_id,
                    status=result.status.value,
                    objective=result.objective,
                    plan_json=result.plan.model_dump(mode="json") if result.plan else None,
                    execution_json=(
                        result.execution.model_dump(mode="json") if result.execution else None
                    ),
                    review_json=result.review.model_dump(mode="json") if result.review else None,
                    pending_approval_json=pending_json,
                    token_usage_json=result.token_usage.model_dump(mode="json"),
                    error=result.error,
                    started_at=result.started_at,
                    completed_at=result.completed_at,
                )
                session.add(record)
            else:
                existing.status = result.status.value
                existing.plan_json = result.plan.model_dump(mode="json") if result.plan else None
                existing.execution_json = (
                    result.execution.model_dump(mode="json") if result.execution else None
                )
                existing.review_json = (
                    result.review.model_dump(mode="json") if result.review else None
                )
                existing.pending_approval_json = pending_json
                existing.token_usage_json = result.token_usage.model_dump(mode="json")
                existing.error = result.error
                existing.completed_at = result.completed_at

            await session.commit()

    async def get_task_result(self, task_id: str) -> WorkflowResult | None:
        """Fetch a WorkflowResult by task_id."""
        async with self._session_factory() as session:
            row = await session.get(TaskExecutionModel, task_id)
            if row is None:
                return None
            return WorkflowResult(
                task_id=row.task_id,
                status=TaskStatus(row.status),
                objective=row.objective,
                plan=PlanOutput.model_validate(row.plan_json) if row.plan_json else None,
                execution=(
                    ExecutorOutput.model_validate(row.execution_json)
                    if row.execution_json
                    else None
                ),
                review=ReviewOutput.model_validate(row.review_json) if row.review_json else None,
                pending_approval=(
                    PendingApproval.model_validate(row.pending_approval_json)
                    if row.pending_approval_json
                    else None
                ),
                token_usage=TokenUsage.model_validate(row.token_usage_json or {}),
                error=row.error,
                started_at=row.started_at,
                completed_at=row.completed_at,
            )

    async def list_recent_tasks(self, limit: int = 20) -> list[WorkflowResult]:
        """List most recent task executions."""
        async with self._session_factory() as session:
            stmt = (
                select(TaskExecutionModel)
                .order_by(TaskExecutionModel.started_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                WorkflowResult(
                    task_id=row.task_id,
                    status=TaskStatus(row.status),
                    objective=row.objective,
                    plan=PlanOutput.model_validate(row.plan_json) if row.plan_json else None,
                    execution=(
                        ExecutorOutput.model_validate(row.execution_json)
                        if row.execution_json
                        else None
                    ),
                    review=(
                        ReviewOutput.model_validate(row.review_json) if row.review_json else None
                    ),
                    pending_approval=(
                        PendingApproval.model_validate(row.pending_approval_json)
                        if row.pending_approval_json
                        else None
                    ),
                    token_usage=TokenUsage.model_validate(row.token_usage_json or {}),
                    error=row.error,
                    started_at=row.started_at,
                    completed_at=row.completed_at,
                )
                for row in rows
            ]

    async def insert_extracted_record(
        self,
        *,
        task_id: str,
        title: str,
        payload: dict[str, Any],
        category: str = "general",
        source_url: str | None = None,
    ) -> int:
        """Persist a structured record produced by database_writer tool."""
        async with self._session_factory() as session:
            record = ExtractedRecordModel(
                task_id=task_id,
                source_url=source_url,
                category=category,
                title=title,
                payload=payload,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record.id

    async def get_extracted_records(self, task_id: str) -> list[dict[str, Any]]:
        """Retrieve all extracted records associated with a task_id."""
        async with self._session_factory() as session:
            stmt = select(ExtractedRecordModel).where(ExtractedRecordModel.task_id == task_id)
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": row.id,
                    "task_id": row.task_id,
                    "source_url": row.source_url,
                    "category": row.category,
                    "title": row.title,
                    "payload": row.payload,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    async def delete_extracted_records(
        self,
        *,
        task_id: str | None = None,
        record_id: int | None = None,
    ) -> int:
        """Delete extracted records by record_id or task_id (used by database_deleter)."""
        if task_id is None and record_id is None:
            raise ValueError("Provide at least one of 'task_id' or 'record_id'.")
        async with self._session_factory() as session:
            stmt = delete(ExtractedRecordModel)
            if record_id is not None:
                stmt = stmt.where(ExtractedRecordModel.id == record_id)
            if task_id is not None:
                stmt = stmt.where(ExtractedRecordModel.task_id == task_id)
            res = await session.execute(stmt)
            await session.commit()
            return int(res.rowcount or 0)

    async def close(self) -> None:
        """Dispose database engine."""
        await self._engine.dispose()
