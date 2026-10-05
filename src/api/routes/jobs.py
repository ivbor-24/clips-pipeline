import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import get_db
from src.api.models.job import Job, JobEvent, JobStatus
from src.api.routes.auth import get_current_user
from src.api.schemas.job import JobCreate, JobListResponse, JobResponse
from src.api.services.auth import LocalUser
from src.api.services.job import cancel_job, create_job, get_job, list_jobs, retry_job

router = APIRouter()

# The worker writes job events to the database; the progress stream polls them.
EVENT_POLL_SEC = 0.5
KEEPALIVE_SEC = 30.0
TERMINAL_EVENTS = {
    JobStatus.COMPLETED: "job_completed",
    JobStatus.FAILED: "job_failed",
    JobStatus.CANCELLED: "job_cancelled",
}


@router.post("/", response_model=JobResponse, status_code=201)
async def create_job_endpoint(
    job_data: JobCreate,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await create_job(user.id, job_data, db)


@router.get("/", response_model=JobListResponse)
async def list_jobs_endpoint(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    jobs, total = await list_jobs(user.id, db, limit, offset)
    return JobListResponse(jobs=jobs, total=total)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_endpoint(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/{job_id}", status_code=204)
async def cancel_job_endpoint(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    success = await cancel_job(job_id, user.id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")


@router.post("/{job_id}/delete", status_code=204)
async def delete_job_endpoint(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a finished job with its clips and files (DELETE /{job_id} cancels)."""
    from src.api.services.cleanup import CleanupError, delete_job

    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        await delete_job(db, job)
    except CleanupError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_job_endpoint(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if job and job.sources_removed_at is not None:
        raise HTTPException(
            status_code=409,
            detail="The source video of this job was removed to free disk space "
            "(cleanup.retention_days): it cannot be resumed. Start a new job instead.",
        )
    success = await retry_job(job_id, user.id, db)
    if not success:
        raise HTTPException(status_code=400, detail="Job not found or cannot be retried")
    job = await get_job(job_id, user.id, db)
    return job


@router.get("/{job_id}/progress")
async def stream_progress(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        # Send initial job status immediately so frontend shows progress
        status = {"id": job.id, "status": job.status.value, "job_type": job.job_type.value}
        yield f"event: job_status\ndata: {json.dumps(status)}\n\n"

        # Replay the current attempt only: events before the last retry or
        # restart would close the stream on an old job_failed.
        last_id = (
            await db.scalar(
                select(func.max(JobEvent.id)).where(
                    JobEvent.job_id == job_id, JobEvent.event_type == "job_requeued"
                )
            )
            or 0
        )
        idle = 0.0
        while True:
            # Plain rows, not ORM objects: the rollback below expires those.
            events = (
                await db.execute(
                    select(JobEvent.id, JobEvent.event_type, JobEvent.data)
                    .where(JobEvent.job_id == job_id, JobEvent.id > last_id)
                    .order_by(JobEvent.id)
                )
            ).all()
            current = await db.scalar(select(Job.status).where(Job.id == job_id))
            # End the read transaction so the next poll sees the worker's writes.
            await db.rollback()

            for event_id, event_type, data in events:
                last_id = event_id
                yield f"event: {event_type}\ndata: {data}\n\n"
                if event_type in TERMINAL_EVENTS.values():
                    return
            if not events and current in TERMINAL_EVENTS:
                # A finished job without a final event (e.g. from before the
                # worker existed): say so, or EventSource would reconnect forever.
                data = json.dumps({"job_id": job_id})
                yield f"event: {TERMINAL_EVENTS[current]}\ndata: {data}\n\n"
                return

            if events:
                idle = 0.0
            elif idle >= KEEPALIVE_SEC:
                yield ": keepalive\n\n"
                idle = 0.0
            await asyncio.sleep(EVENT_POLL_SEC)
            idle += EVENT_POLL_SEC

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
