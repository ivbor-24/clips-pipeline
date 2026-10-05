"""State the job worker publishes for the Diagnostics page.

Only the worker sees the GPU (in Docker only its container gets the device),
so it runs the GPU self-check (scripts/check_gpu.py --json) and stores the
result here; the API reads it. A single row, id 1.
"""

from sqlalchemy import Column, DateTime, Integer, String, Text

from src.api.db_base import Base

WORKER_STATUS_ID = 1


class WorkerStatus(Base):
    __tablename__ = "worker_status"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    # Updated every few seconds while the worker runs, between and during jobs.
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    backend = Column(String(32), nullable=True)
    version = Column(String(32), nullable=True)
    # JSON printed by scripts/check_gpu.py --json, and when.
    gpu_check = Column(Text, nullable=True)
    gpu_check_at = Column(DateTime(timezone=True), nullable=True)
    # Set by the API ("check again"); the worker runs the check between jobs.
    gpu_check_requested_at = Column(DateTime(timezone=True), nullable=True)
    # Shutdown from the web UI or after idle time. A worker acts
    # only on a request newer than its own start, so `just up` is not undone.
    shutdown_requested_at = Column(DateTime(timezone=True), nullable=True)
    # wait (for the running job) | cancel (it) | requeue (it, resumes next start)
    shutdown_mode = Column(String(16), nullable=True)
    # user | idle
    shutdown_reason = Column(String(16), nullable=True)
    # Written by the worker when it exits on its own (shutdown, SIGTERM).
    stopped_at = Column(DateTime(timezone=True), nullable=True)
