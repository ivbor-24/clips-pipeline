"""Job service of the API: puts jobs in the queue, cancels and retries them.

The jobs themselves run in the job worker (src/worker.py), a separate process
that takes queued jobs from the database one at a time.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.job_config import build_job_config
from src.api.models.job import Job, JobEvent, JobStatus, JobType
from src.api.schemas.job import JobCreate

__all__ = [
    "build_job_config",
    "cancel_job",
    "create_job",
    "get_job",
    "list_jobs",
    "retry_job",
]

logger = structlog.get_logger("job_service")

# Work directories of web jobs, relative to the project root. In Docker it is
# user data mounted from the host; tests point it at a temporary directory.
JOBS_DIR = Path("jobs")


async def create_job(user_id: int, job_data: JobCreate, db: AsyncSession) -> Job:
    """Create a job in the queue; the worker picks it up."""
    job_id_temp = str(uuid.uuid4())[:8]
    work_dir = JOBS_DIR / f"job_{job_id_temp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    job_type = JobType.CHAPTERS if job_data.job_type == "chapters" else JobType.CLIPS

    job = Job(
        user_id=user_id,
        status=JobStatus.QUEUED,
        job_type=job_type,
        input_source=job_data.input_source,
        config_overrides=(
            json.dumps(job_data.config_overrides) if job_data.config_overrides else None
        ),
        work_dir=str(work_dir),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    logger.info("job_queued", job_id=job.id, job_type=job_type.value)
    return job


async def retry_job(job_id: int, user_id: int, db: AsyncSession) -> bool:
    """Put a finished job back in the queue; it resumes from its artifacts."""
    result = await db.execute(select(Job).where(Job.id == job_id, Job.user_id == user_id))
    job = result.scalar_one_or_none()
    if not job:
        return False

    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        return False

    job.status = JobStatus.QUEUED
    job.started_at = None
    job.completed_at = None
    job.error_message = None
    job.cancel_requested = False
    db.add(JobEvent(job_id=job.id, event_type="job_requeued", data='{"reason": "retry"}'))
    await db.commit()
    logger.info("job_requeued", job_id=job.id, reason="retry")
    return True


async def cancel_job(job_id: int, user_id: int, db: AsyncSession) -> bool:
    """Cancel a job.

    A queued job is cancelled at once. For a running job the worker is asked to
    stop it: it kills the job's processes within seconds and then marks the job
    cancelled, so the status stays ``running`` until the GPU is actually free.
    """
    result = await db.execute(select(Job).where(Job.id == job_id, Job.user_id == user_id))
    job = result.scalar_one_or_none()
    if not job:
        return False

    if job.status == JobStatus.QUEUED:
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        db.add(JobEvent(job_id=job.id, event_type="job_cancelled", data="{}"))
    elif job.status == JobStatus.RUNNING:
        job.cancel_requested = True
    else:
        return False

    await db.commit()
    logger.info("job_cancel_requested", job_id=job.id, status=job.status.value)
    return True


async def get_job(job_id: int, user_id: int, db: AsyncSession) -> Job:
    result = await db.execute(select(Job).where(Job.id == job_id, Job.user_id == user_id))
    return result.scalar_one_or_none()


async def list_jobs(
    user_id: int, db: AsyncSession, limit: int = 50, offset: int = 0
) -> tuple[list[Job], int]:
    result = await db.execute(
        select(Job)
        .where(Job.user_id == user_id)
        .order_by(Job.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    jobs = result.scalars().all()

    count_result = await db.execute(
        select(func.count()).select_from(Job).where(Job.user_id == user_id)
    )
    total = count_result.scalar()

    return jobs, total
