import enum

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.sql import expression, func

from src.api.db_base import Base


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    CLIPS = "clips"
    CHAPTERS = "chapters"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    status = Column(SQLEnum(JobStatus), nullable=False, default=JobStatus.QUEUED)
    job_type = Column(SQLEnum(JobType), nullable=False, default=JobType.CLIPS)
    input_source = Column(String(500), nullable=False)
    config_overrides = Column(Text, nullable=True)
    work_dir = Column(String(500), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    # Written by the job worker (src/worker.py). A schema change needs a
    # migration: just db-revision "message" (src/api/migrations).
    current_stage = Column(String(64), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    cancel_requested = Column(Boolean, nullable=False, server_default=expression.false())
    attempts = Column(Integer, nullable=False, server_default="0")
    # Process group of the child running the job, so a restarted worker can
    # stop a child left behind by a worker that was killed.
    process_group = Column(Integer, nullable=True)
    # When the cleanup removed the source video (src/api/services/cleanup.py):
    # clips stay, the job can no longer be resumed.
    sources_removed_at = Column(DateTime(timezone=True), nullable=True)


class JobEvent(Base):
    """Progress event of a job, written by the worker and streamed over SSE.

    Stored in the database rather than in API memory, so the worker (another
    process) can publish them and a page opened later still gets the history.
    """

    __tablename__ = "job_events"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(32), nullable=False)
    data = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
