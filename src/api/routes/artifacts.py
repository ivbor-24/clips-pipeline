import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import get_db
from src.api.models.job import Job
from src.api.routes.auth import get_current_user
from src.api.schemas.artifacts import (
    BrollResponse,
    BrollTextResponse,
    ChaptersResponse,
    ChaptersTextResponse,
    CropParamsResponse,
    JobLogLine,
    JobLogResponse,
    ManifestResponse,
    NoticesResponse,
    ScoredSegmentsResponse,
    TranscriptResponse,
)
from src.api.services.auth import LocalUser
from src.api.services.job import get_job
from src.notices import read_notices

router = APIRouter()


def load_artifact(job: Job, filename: str) -> dict:
    artifact_path = Path(job.work_dir) / "artifacts" / filename
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact {filename} not found")

    with open(artifact_path) as f:
        return json.load(f)


@router.get("/{job_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    data = load_artifact(job, "transcript.json")
    return TranscriptResponse(segments=data)


@router.get("/{job_id}/scored_segments", response_model=ScoredSegmentsResponse)
async def get_scored_segments(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    data = load_artifact(job, "scored_segments.json")
    return ScoredSegmentsResponse(segments=data)


@router.get("/{job_id}/crop_params", response_model=CropParamsResponse)
async def get_crop_params(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    data = load_artifact(job, "crop_params.json")
    return CropParamsResponse(crops=data)


@router.get("/{job_id}/manifest", response_model=ManifestResponse)
async def get_manifest(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    manifest_path = Path(job.work_dir) / "output" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")

    with open(manifest_path) as f:
        data = json.load(f)

    return ManifestResponse(**data)


@router.get("/{job_id}/chapters", response_model=ChaptersResponse)
async def get_chapters(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    data = load_artifact(job, "chapters.json")
    return ChaptersResponse(chapters=data.get("chapters", []), source=data.get("source", ""))


@router.get("/{job_id}/chapters/text", response_model=ChaptersTextResponse)
async def get_chapters_text(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    chapters_txt_path = Path(job.work_dir) / "artifacts" / "chapters.txt"
    if not chapters_txt_path.exists():
        raise HTTPException(status_code=404, detail="Chapters text file not found")

    with open(chapters_txt_path) as f:
        text = f.read()

    return ChaptersTextResponse(text=text, format="youtube")


@router.get("/{job_id}/broll", response_model=BrollResponse)
async def get_broll(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    data = load_artifact(job, "broll_suggestions.json")
    return BrollResponse(suggestions=data.get("suggestions", []), source=data.get("source", ""))


@router.get("/{job_id}/broll/text", response_model=BrollTextResponse)
async def get_broll_text(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    broll_txt_path = Path(job.work_dir) / "artifacts" / "broll_suggestions.txt"
    if not broll_txt_path.exists():
        raise HTTPException(status_code=404, detail="B-roll suggestions text file not found")

    with open(broll_txt_path) as f:
        text = f.read()

    return BrollTextResponse(text=text)


@router.get("/{job_id}/notices", response_model=NoticesResponse)
async def get_notices(
    job_id: int,
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fallbacks that changed the job's result, e.g. clips picked without the LLM."""
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return NoticesResponse(notices=read_notices(job.work_dir))


# Enough for any job (a 25-minute lecture writes ~50 KB); a runaway log cannot
# make one response huge.
MAX_LOG_CHUNK_BYTES = 2 * 1024 * 1024


def read_job_log(path: Path, offset: int) -> JobLogResponse:
    """The JSON lines of job.log after byte ``offset``, up to the last full line."""
    if not path.exists():
        return JobLogResponse(lines=[], offset=0)
    size = path.stat().st_size
    if offset > size:  # the log was replaced (job retried from scratch)
        offset = 0
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read(MAX_LOG_CHUNK_BYTES)
    end = chunk.rfind(b"\n") + 1  # a line still being written waits for the next call
    lines = []
    for raw in chunk[:end].splitlines():
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            record = {"event": raw.decode("utf-8", errors="replace")}
        if not isinstance(record, dict):
            record = {"event": str(record)}
        lines.append(
            JobLogLine(
                timestamp=record.pop("timestamp", None),
                level=str(record.pop("level", "info")),
                event=str(record.pop("event", "")),
                fields={k: v for k, v in record.items() if k != "job_id"},
            )
        )
    return JobLogResponse(lines=lines, offset=offset + end)


@router.get("/{job_id}/log", response_model=JobLogResponse)
async def get_job_log(
    job_id: int,
    offset: int = Query(0, ge=0),
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The job's log (artifacts/logs/job.log), also after the job has ended."""
    job = await get_job(job_id, user.id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return read_job_log(Path(job.work_dir) / "artifacts" / "logs" / "job.log", offset)
