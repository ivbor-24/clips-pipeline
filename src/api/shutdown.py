"""Shutting the system down from the web UI or after idle time.

The API records the request in worker_status; the job worker finishes the way
the user chose (wait for the running job, cancel it, or put it back in the
queue), removes its temporary files and exits; then the API exits with code 0
(src/api/serve.py). In Docker the web UI container sees the health endpoint
say "shutting_down", then the API gone, and stops too. Nothing restarts them
(restart: on-failure); `just up` starts everything again.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.models.job import Job, JobStatus
from src.api.models.worker import WORKER_STATUS_ID, WorkerStatus

logger = structlog.get_logger("api.shutdown")

MODES = ("wait", "cancel", "requeue")
# After the worker has stopped: time for open pages to show the final state
# and for the web UI container to notice, before the API exits.
EXIT_DELAY_SEC = 6.0
# A worker that sent no heartbeat for this long is not running.
WORKER_GONE_SEC = 45.0
IDLE_CHECK_SEC = 60.0


class _State:
    """This API process's side: when a shutdown began, when a page last called."""

    def __init__(self) -> None:
        self.requested_at: Optional[datetime] = None
        self.last_activity = time.monotonic()
        # The task that waits for the worker and then stops the API. Kept here:
        # the event loop holds only a weak reference to a task.
        self.finish_task: Optional[asyncio.Task] = None


state = _State()


def note_activity() -> None:
    """A web page called the API: the system is in use."""
    state.last_activity = time.monotonic()


def idle_minutes() -> float:
    return (time.monotonic() - state.last_activity) / 60


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is not None and value.tzinfo is None:  # SQLite: naive UTC
        value = value.replace(tzinfo=timezone.utc)
    return value


def worker_stopped(status: Optional[WorkerStatus], since: datetime) -> bool:
    """The worker exited after ``since``, or is not running at all."""
    if status is None or status.heartbeat_at is None:
        return True
    stopped = _aware(status.stopped_at)
    if stopped is not None and stopped >= since:
        return True
    age = (datetime.now(timezone.utc) - _aware(status.heartbeat_at)).total_seconds()
    return age > WORKER_GONE_SEC


async def request_shutdown(
    db: AsyncSession, session_factory: async_sessionmaker, mode: str, reason: str
) -> datetime:
    """Record the request for the worker and start waiting for it to stop."""
    if state.requested_at is not None:
        return state.requested_at
    now = datetime.now(timezone.utc)
    status = await db.get(WorkerStatus, WORKER_STATUS_ID)
    if status is None:
        status = WorkerStatus(id=WORKER_STATUS_ID)
        db.add(status)
    status.shutdown_requested_at = now
    status.shutdown_mode = mode
    status.shutdown_reason = reason
    await db.commit()
    state.requested_at = now
    logger.info("shutdown_requested", mode=mode, reason=reason)
    state.finish_task = asyncio.get_running_loop().create_task(_finish(session_factory, now))
    return now


async def _finish(session_factory: async_sessionmaker, since: datetime) -> None:
    """Wait for the worker to stop (a "wait" may take a whole job), then exit."""
    from src.api import serve

    while True:
        async with session_factory() as db:
            status = await db.get(WorkerStatus, WORKER_STATUS_ID)
            if worker_stopped(status, since):
                break
        await asyncio.sleep(1.0)
    logger.info("shutdown_worker_stopped")
    await asyncio.sleep(EXIT_DELAY_SEC)
    if serve.request_exit():
        logger.info("shutdown_api_exiting")
    else:
        # uvicorn started directly (dev mode): dev.sh stops the API together
        # with the worker; otherwise it keeps serving the final status.
        logger.info("shutdown_api_left_running", reason="not started through src.api.serve")


async def idle_watchdog(session_factory: async_sessionmaker, minutes: int) -> None:
    """Shut down after ``minutes`` without page requests and without jobs."""
    if minutes <= 0:
        return
    while True:
        await asyncio.sleep(IDLE_CHECK_SEC)
        if state.requested_at is not None or idle_minutes() < minutes:
            continue
        async with session_factory() as db:
            busy = await db.scalar(
                select(func.count())
                .select_from(Job)
                .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
            )
            if busy:
                continue
            logger.info("idle_shutdown", idle_minutes=round(idle_minutes()), limit=minutes)
            await request_shutdown(db, session_factory, "wait", "idle")
