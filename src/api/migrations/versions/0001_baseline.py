"""Baseline: the job database as of v0.8.0.

Before migrations, the API and the worker made the schema with create_all
and added new columns one by one (add_missing_columns), so an old database
may lack later tables and columns. This revision creates what is missing
instead of failing on what exists: a new database and every old one end at
the same schema. The definitions are frozen here on purpose: later changes
are new revisions, never edits of this one or imports of the live models.

Revision ID: 0001
Revises:
Create Date: 2026-10-02
"""

from typing import Callable, Dict, List, Sequence, Tuple, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _jobs() -> List:
    return [
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", name="jobstatus"),
            nullable=False,
        ),
        sa.Column("job_type", sa.Enum("CLIPS", "CHAPTERS", name="jobtype"), nullable=False),
        sa.Column("input_source", sa.String(length=500), nullable=False),
        sa.Column("config_overrides", sa.Text(), nullable=True),
        sa.Column("work_dir", sa.String(length=500), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("current_stage", sa.String(length=64), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("process_group", sa.Integer(), nullable=True),
        sa.Column("sources_removed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    ]


def _worker_status() -> List:
    return [
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("backend", sa.String(length=32), nullable=True),
        sa.Column("version", sa.String(length=32), nullable=True),
        sa.Column("gpu_check", sa.Text(), nullable=True),
        sa.Column("gpu_check_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gpu_check_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("shutdown_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("shutdown_mode", sa.String(length=16), nullable=True),
        sa.Column("shutdown_reason", sa.String(length=16), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    ]


def _clip_reviews() -> List:
    return [
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("clip_id", sa.String(length=64), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=True),
        sa.Column("end_time", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "clip_id", name="uq_clip_review"),
    ]


def _job_events() -> List:
    return [
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("data", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    ]


# Table -> (definition, indexes as (name, columns)), in creation order.
TABLES: Dict[str, Tuple[Callable[[], List], List[Tuple[str, List[str]]]]] = {
    "jobs": (_jobs, [("ix_jobs_id", ["id"]), ("ix_jobs_user_id", ["user_id"])]),
    "worker_status": (_worker_status, []),
    "clip_reviews": (_clip_reviews, [("ix_clip_reviews_job_id", ["job_id"])]),
    "job_events": (_job_events, [("ix_job_events_job_id", ["job_id"])]),
}


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = set(inspector.get_table_names())
    for table, (definition, indexes) in TABLES.items():
        items = definition()
        if table not in existing:
            op.create_table(table, *items)
            have_indexes = set()
        else:
            # A database from before migrations: add the columns it lacks. They
            # are all nullable or have a server default (the rule of
            # add_missing_columns), so SQLite adds them without a rebuild.
            have = {column["name"] for column in inspector.get_columns(table)}
            for item in items:
                if isinstance(item, sa.Column) and item.name not in have:
                    op.add_column(table, item)
            have_indexes = {index["name"] for index in inspector.get_indexes(table)}
        for name, columns in indexes:
            if name not in have_indexes:
                op.create_index(name, table, columns, unique=False)


def downgrade() -> None:
    for table in reversed(list(TABLES)):
        op.drop_table(table)
