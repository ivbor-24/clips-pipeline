"""
Disk cleanup: source videos of old jobs, orphan uploads, job deletion.

Policy: ``cleanup.retention_days`` after a job
finished (default 30, 0 = off), its source video is removed: the upload, the
job's copy (artifacts/video) and its audio (artifacts/audio), ~2 GB for a
lecture. Clips, JSON artifacts, reviews and the job card stay; such a job can
no longer be resumed (``jobs.sources_removed_at``). A whole job goes only on
request (the Delete button, ``delete_job``). Uploads no job refers to are
removed after ORPHAN_UPLOAD_HOURS.

The API runs ``cleanup_loop``: when it starts and every CLEANUP_INTERVAL_SEC.

Inputs / Outputs:
- data/api.db: jobs (status, completed_at, input_source, work_dir, sources_removed_at)
- jobs/<id>/artifacts/{video,audio,temp_download}, uploads/
"""

import asyncio
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.models.job import Job, JobEvent, JobStatus
from src.api.models.review import ClipReview
from src.config import CleanupConfig

logger = structlog.get_logger("cleanup")

# Parts of a job directory that only the source video needs: the copy made
# by ingestion, its audio, a half-finished download.
SOURCE_DIRS = ("artifacts/video", "artifacts/audio", "artifacts/temp_download")
TERMINAL_STATUSES = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
# An upload whose job was never created (the page closed between the two).
ORPHAN_UPLOAD_HOURS = 24
CLEANUP_INTERVAL_SEC = 6 * 3600


class CleanupError(Exception):
    """Custom exception for cleanup errors."""

    pass


def _jobs_root() -> Path:
    return Path("jobs").resolve()


def _uploads_root() -> Path:
    return Path("uploads").resolve()


def _inside(path: Path, root: Path) -> bool:
    """True if ``path`` is below ``root`` (not ``root`` itself)."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved != root and resolved.is_relative_to(root)


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


async def _upload_in_use(db: AsyncSession, input_source: str, job_id: int) -> bool:
    """Does another job still need this upload (its sources are not removed)?"""
    count = await db.scalar(
        select(func.count())
        .select_from(Job)
        .where(
            Job.input_source == input_source,
            Job.id != job_id,
            Job.sources_removed_at.is_(None),
        )
    )
    return bool(count)


def _remove_upload(input_source: str) -> int:
    """Delete an uploaded file; URLs and paths outside uploads/ are left alone."""
    path = Path(input_source)
    if not _inside(path, _uploads_root()) or not path.is_file():
        return 0
    size = path.stat().st_size
    path.unlink()
    logger.info("upload_removed", path=input_source, bytes=size)
    return size


def _remove_tree(path: Path) -> int:
    size = get_directory_size(path)
    shutil.rmtree(path, ignore_errors=True)
    return size


async def remove_job_sources(db: AsyncSession, job: Job) -> int:
    """Delete a finished job's source video, keeping its clips and card.

    Returns:
        Bytes freed.
    """
    if job.status not in TERMINAL_STATUSES:
        raise CleanupError(f"Job {job.id} is {job.status.value}, not finished")
    freed = 0
    work_dir = Path(job.work_dir)
    if _inside(work_dir, _jobs_root()):
        for relative in SOURCE_DIRS:
            target = work_dir / relative
            if target.exists():
                freed += _remove_tree(target)
    else:
        logger.warning("skipping_unsafe_work_dir", job_id=job.id, path=job.work_dir)
    if not await _upload_in_use(db, job.input_source, job.id):
        freed += _remove_upload(job.input_source)
    job.sources_removed_at = datetime.now(timezone.utc)
    await db.commit()
    logger.info("job_sources_removed", job_id=job.id, bytes_freed=freed)
    return freed


async def delete_job(db: AsyncSession, job: Job) -> int:
    """Delete a finished job: its directory, its upload if unused, its rows.

    Returns:
        Bytes freed.
    """
    if job.status not in TERMINAL_STATUSES:
        raise CleanupError(f"Job {job.id} is {job.status.value}: cancel it first")
    freed = 0
    work_dir = Path(job.work_dir)
    if _inside(work_dir, _jobs_root()):
        if work_dir.exists():
            freed += _remove_tree(work_dir)
    else:
        logger.warning("skipping_unsafe_work_dir", job_id=job.id, path=job.work_dir)
    if not await _upload_in_use(db, job.input_source, job.id):
        freed += _remove_upload(job.input_source)
    job_id = job.id
    await db.execute(delete(JobEvent).where(JobEvent.job_id == job_id))
    await db.execute(delete(ClipReview).where(ClipReview.job_id == job_id))
    await db.delete(job)
    await db.commit()
    logger.info("job_deleted", job_id=job_id, bytes_freed=freed)
    return freed


async def remove_orphan_uploads(
    db: AsyncSession, older_than_hours: float = ORPHAN_UPLOAD_HOURS
) -> Dict[str, int]:
    """Delete uploads no job refers to, older than ``older_than_hours``."""
    root = _uploads_root()
    if not root.is_dir():
        return {"files_removed": 0, "bytes_freed": 0}
    sources = (await db.execute(select(Job.input_source))).scalars().all()
    referenced: Set[Path] = set()
    for source in sources:
        try:
            referenced.add(Path(source).resolve())
        except OSError:
            continue
    cutoff = datetime.now(timezone.utc).timestamp() - older_than_hours * 3600
    removed = freed = 0
    for path in root.rglob("*"):
        # *.part: uploads in progress; the API removes stale ones at start.
        if not path.is_file() or path.name.endswith(".part") or path.resolve() in referenced:
            continue
        stat = path.stat()
        if stat.st_mtime >= cutoff:
            continue
        path.unlink()
        removed += 1
        freed += stat.st_size
        logger.info("orphan_upload_removed", path=str(path), bytes=stat.st_size)
    for directory in sorted(root.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return {"files_removed": removed, "bytes_freed": freed}


async def run_cleanup(
    db: AsyncSession, config: CleanupConfig, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """Remove the sources of jobs older than ``retention_days``, then orphan uploads."""
    now = now or datetime.now(timezone.utc)
    jobs_cleaned = 0
    sources_freed = 0
    cutoff = None
    if config.retention_days > 0:
        cutoff = now - timedelta(days=config.retention_days)
        finished = func.coalesce(Job.completed_at, Job.created_at)
        old_jobs = (
            (
                await db.execute(
                    select(Job).where(
                        Job.status.in_(TERMINAL_STATUSES),
                        Job.sources_removed_at.is_(None),
                        finished < cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )
        for job in old_jobs:
            sources_freed += await remove_job_sources(db, job)
            jobs_cleaned += 1
    orphans = await remove_orphan_uploads(db)

    uploads_gb = get_directory_size(_uploads_root()) / 1024**3
    if uploads_gb > config.max_upload_size_gb:
        logger.warning(
            "uploads_size_exceeded",
            current_gb=round(uploads_gb, 2),
            max_gb=config.max_upload_size_gb,
        )
    total = sources_freed + orphans["bytes_freed"]
    result = {
        "retention_days": config.retention_days,
        "cutoff_date": cutoff.isoformat() if cutoff else None,
        "jobs_cleaned": jobs_cleaned,
        "orphan_uploads_removed": orphans["files_removed"],
        "total_bytes_freed": total,
        "total_mb_freed": round(total / 1024**2, 2),
        "uploads_current_size_gb": round(uploads_gb, 2),
    }
    logger.info("cleanup_completed", **result)
    return result


async def cleanup_loop(
    session_factory: async_sessionmaker,
    config_path: str = "config/config.yaml",
    interval_sec: float = CLEANUP_INTERVAL_SEC,
) -> None:
    """Run the cleanup now and every ``interval_sec``; the config is read each time."""
    from src.user_settings import load_app_config

    while True:
        try:
            config = load_app_config(config_path)
            async with session_factory() as db:
                await run_cleanup(db, config.cleanup)
        except Exception as e:  # a bad config or a locked file must not stop the API
            logger.warning("cleanup_failed", error=str(e))
        await asyncio.sleep(interval_sec)
