"""Diagnostics page: GET the state, POST to run the GPU check again."""

import asyncio
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import get_db
from src.api.models.job import Job, JobStatus
from src.api.models.worker import WORKER_STATUS_ID, WorkerStatus
from src.api.routes.auth import get_current_user
from src.api.services import diagnostics
from src.api.services.auth import LocalUser

logger = structlog.get_logger("api.diagnostics")

router = APIRouter()


def _local_checks() -> dict:
    """Files, ffmpeg and disks: blocking calls, run off the event loop."""
    try:
        models = diagnostics.models()
        models_error = None
    except Exception as e:  # e.g. an invalid config.yaml
        logger.warning("diagnostics_models_failed", error=str(e))
        models, models_error = [], str(e)
    return {
        "models": models,
        "models_error": models_error,
        "ffmpeg": diagnostics.ffmpeg(),
        "disk": diagnostics.disks(),
    }


@router.get("/")
async def get_diagnostics(
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    status = await db.get(WorkerStatus, WORKER_STATUS_ID)
    running = (
        await db.execute(select(Job).where(Job.status == JobStatus.RUNNING).limit(1))
    ).scalar_one_or_none()
    running_job = {"id": running.id, "stage": running.current_stage} if running else None
    local = await asyncio.to_thread(_local_checks)
    return {
        "system": diagnostics.system(),
        "worker": diagnostics.worker_state(status, running_job),
        "gpu_check": diagnostics.gpu_check(status),
        **local,
    }


@router.post("/gpu-check", status_code=202)
async def request_gpu_check(
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Ask the worker to run the GPU self-check again (after the current job, if any)."""
    status = await db.get(WorkerStatus, WORKER_STATUS_ID)
    if status is None:
        status = WorkerStatus(id=WORKER_STATUS_ID)
        db.add(status)
    status.gpu_check_requested_at = datetime.now(timezone.utc)
    await db.commit()
    return {"requested_at": status.gpu_check_requested_at.isoformat()}
