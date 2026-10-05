import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import get_db
from src.api.models.review import ClipReview
from src.api.routes.auth import get_current_user
from src.api.schemas.review import (
    ClipListResponse,
    ClipResponse,
    ClipUpdate,
    ReviewStatus,
)
from src.api.services.auth import LocalUser
from src.api.services.job import get_job

router = APIRouter()
logger = structlog.get_logger("review")

# Clip boundaries are rounded to 0.01 s in the clip metadata.
SAME_CLIP_TOLERANCE_SEC = 0.05

CLIP_ID_RE = re.compile(r"^clip_[A-Za-z0-9_-]+$")


def _validate_clip_id(clip_id: str) -> None:
    """Reject clip ids that could traverse the filesystem."""
    if not CLIP_ID_RE.match(clip_id):
        raise HTTPException(status_code=400, detail="Invalid clip id")


def _clip_range(meta: Dict[str, Any]) -> tuple:
    return meta.get("start_time"), meta.get("end_time")


def _review_matches(review: ClipReview, meta: Dict[str, Any]) -> bool:
    """Whether a stored review still describes the clip rendered under its id."""
    start, end = _clip_range(meta)
    if None in (review.start_time, review.end_time, start, end):
        return True
    return (
        abs(review.start_time - start) <= SAME_CLIP_TOLERANCE_SEC
        and abs(review.end_time - end) <= SAME_CLIP_TOLERANCE_SEC
    )


def _read_clip_metas(job) -> Dict[str, Dict[str, Any]]:
    clips_dir = Path(job.work_dir) / "output" / "clips"
    if not clips_dir.exists():
        return {}
    metas = {}
    for meta_file in sorted(clips_dir.glob("clip_*.meta.json")):
        with open(meta_file) as f:
            metas[meta_file.name.removesuffix(".meta.json")] = json.load(f)
    return metas


async def _import_legacy_reviews(
    job, metas: Dict[str, Dict[str, Any]], reviews: Dict[str, ClipReview], db: AsyncSession
) -> None:
    """Move decisions older versions wrote into ``<clip>.meta.json`` to the database.

    Runs once per clip: after that the database row wins, even when rendering
    rewrites the metadata file without them.
    """
    imported = []
    for clip_id, meta in metas.items():
        status = meta.get("review_status", ReviewStatus.PENDING.value)
        if clip_id in reviews or status == ReviewStatus.PENDING.value:
            continue
        start, end = _clip_range(meta)
        review = ClipReview(
            job_id=job.id,
            clip_id=clip_id,
            start_time=start,
            end_time=end,
            status=ReviewStatus(status).value,
            notes=meta.get("review_notes"),
        )
        db.add(review)
        reviews[clip_id] = review
        imported.append(clip_id)
    if not imported:
        return
    try:
        await db.commit()
    except IntegrityError:
        # A parallel request imported them first; use its rows.
        await db.rollback()
        rows = await db.scalars(select(ClipReview).where(ClipReview.job_id == job.id))
        reviews.clear()
        reviews.update({r.clip_id: r for r in rows})
        return
    logger.info("legacy_reviews_imported", job_id=job.id, clips=imported)


async def load_clips(job, db: AsyncSession) -> List[ClipResponse]:
    metas = _read_clip_metas(job)
    rows = await db.scalars(select(ClipReview).where(ClipReview.job_id == job.id))
    reviews = {r.clip_id: r for r in rows}
    await _import_legacy_reviews(job, metas, reviews, db)

    clips_dir = Path(job.work_dir) / "output" / "clips"
    clips = []
    for clip_id, meta in metas.items():
        review: Optional[ClipReview] = reviews.get(clip_id)
        if review is not None and not _review_matches(review, meta):
            review = None  # other content under the same id: review it again
        clips.append(
            ClipResponse(
                id=clip_id,
                video_path=f"output/clips/{clip_id}.mp4",
                srt_path=(
                    f"output/clips/{clip_id}.srt"
                    if (clips_dir / f"{clip_id}.srt").exists()
                    else None
                ),
                score=meta.get("score", 0.0),
                duration=meta.get("duration", meta.get("duration_sec", 0.0)),
                tags=meta.get("tags", []),
                review_status=ReviewStatus(review.status) if review else ReviewStatus.PENDING,
                review_notes=review.notes if review else None,
                metadata=meta,
            )
        )
    return clips


@router.get("/{job_id}/clips", response_model=ClipListResponse)
async def list_clips(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clips = await load_clips(job, db)
    return ClipListResponse(clips=clips, total=len(clips))


@router.put("/{job_id}/clips/{clip_id}", response_model=ClipResponse)
async def update_clip(
    job_id: int,
    clip_id: str,
    update: ClipUpdate,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    _validate_clip_id(clip_id)
    meta_file = Path(job.work_dir) / "output" / "clips" / f"{clip_id}.meta.json"
    if not meta_file.resolve().is_relative_to(Path(job.work_dir).resolve()):
        raise HTTPException(status_code=400, detail="Invalid clip path")
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    with open(meta_file) as f:
        meta = json.load(f)
    start, end = _clip_range(meta)

    review = await db.scalar(
        select(ClipReview).where(ClipReview.job_id == job.id, ClipReview.clip_id == clip_id)
    )
    if review is None:
        # A notes-only update of a clip without a decision keeps it pending.
        review = ClipReview(job_id=job.id, clip_id=clip_id, status=ReviewStatus.PENDING.value)
        db.add(review)
    elif not _review_matches(review, meta):
        review.notes = None  # notes were about the clip that used to have this id
    review.start_time, review.end_time = start, end
    if update.review_status is not None:
        review.status = update.review_status.value
    if update.notes is not None:
        review.notes = update.notes or None
    await db.commit()

    clips = await load_clips(job, db)
    clip = next((c for c in clips if c.id == clip_id), None)
    if not clip:
        raise HTTPException(status_code=500, detail="Failed to reload clip")

    return clip


@router.get("/{job_id}/clips/{clip_id}/video")
async def stream_video(
    job_id: int,
    clip_id: str,
    download: bool = False,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The clip's MP4; ``?download=1`` asks the browser to save it, not play it."""
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    _validate_clip_id(clip_id)
    video_path = Path(job.work_dir) / "output" / "clips" / f"{clip_id}.mp4"
    if not video_path.resolve().is_relative_to(Path(job.work_dir).resolve()):
        raise HTTPException(status_code=400, detail="Invalid clip path")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    def iter_file():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(video_path.stat().st_size),
    }
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{clip_id}.mp4"'
    return StreamingResponse(iter_file(), media_type="video/mp4", headers=headers)


@router.get("/{job_id}/clips/{clip_id}/subtitles")
async def download_subtitles(
    job_id: int,
    clip_id: str,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The clip's SRT subtitles, as a download (the Review page's button)."""
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    _validate_clip_id(clip_id)
    srt_path = Path(job.work_dir) / "output" / "clips" / f"{clip_id}.srt"
    if not srt_path.resolve().is_relative_to(Path(job.work_dir).resolve()):
        raise HTTPException(status_code=400, detail="Invalid clip path")
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Subtitles file not found")

    return FileResponse(srt_path, media_type="application/x-subrip", filename=f"{clip_id}.srt")
