"""Shutdown from the web UI and the state the shutdown page follows."""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api import shutdown
from src.api.config import settings
from src.api.database import async_session, get_db
from src.api.models.job import Job, JobStatus
from src.api.models.worker import WORKER_STATUS_ID, WorkerStatus
from src.api.routes.auth import get_current_user
from src.api.services.auth import LocalUser
from src.api.services.diagnostics import worker_state

router = APIRouter()


class ShutdownRequest(BaseModel):
    # What happens to a running job: wait for it, cancel it, or put it back in
    # the queue (it continues from its last stage on the next start).
    mode: Literal["wait", "cancel", "requeue"] = "requeue"


@router.get("/status")
async def system_status(
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    status = await db.get(WorkerStatus, WORKER_STATUS_ID)
    running = (
        await db.execute(select(Job).where(Job.status == JobStatus.RUNNING).limit(1))
    ).scalar_one_or_none()
    queued = await db.scalar(
        select(func.count()).select_from(Job).where(Job.status == JobStatus.QUEUED)
    )
    running_job = {"id": running.id, "stage": running.current_stage} if running else None
    since = shutdown.state.requested_at
    return {
        "state": "shutting_down" if since else "running",
        "shutdown": (
            {
                "requested_at": since.isoformat(),
                "mode": status.shutdown_mode if status else None,
                "reason": status.shutdown_reason if status else None,
                "worker_stopped": shutdown.worker_stopped(status, since),
            }
            if since
            else None
        ),
        "worker": worker_state(status, running_job),
        "jobs": {"running": running_job, "queued": queued or 0},
        "auto_shutdown_min": settings.idle_shutdown_min,
        "idle_min": round(shutdown.idle_minutes(), 1),
    }


@router.post("/shutdown", status_code=202)
async def request_shutdown(
    request: ShutdownRequest,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Stop the worker (as chosen for a running job), then the API and the web UI."""
    since = await shutdown.request_shutdown(db, async_session, request.mode, "user")
    return {"requested_at": since.isoformat(), "mode": request.mode}
