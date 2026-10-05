"""Video upload from the web UI.

The request body is the file itself (``application/octet-stream``), streamed
straight into ``uploads/``: nothing is buffered in memory or in ``/tmp``, which
is RAM on many distributions. The size limit and the free disk space are
checked before the first byte is stored. The file is written as
``<name>.part`` and renamed when complete, so an interrupted upload never looks
like a finished one.
"""

import shutil
import time
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.requests import ClientDisconnect

from src.api.config import settings
from src.api.routes.auth import get_current_user
from src.api.schemas.upload import UploadResponse
from src.api.services.auth import LocalUser

router = APIRouter()
logger = structlog.get_logger("upload")

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".ts"}
PART_SUFFIX = ".part"

# The job copies the video into its own directory (remux), so an upload needs
# room for itself and one copy, plus a margin for audio and clips.
SPACE_PER_UPLOADED_BYTE = 2
DISK_RESERVE_BYTES = 1024**3

UPLOAD_DIR = Path("uploads")


def remove_stale_partial_uploads(max_age_sec: float = 3600) -> int:
    """Remove ``.part`` files of uploads that stopped (API killed mid-upload).

    Only files untouched for ``max_age_sec``: an upload in progress keeps
    writing, and another API process may be receiving one right now.

    Returns:
        Number of files removed.
    """
    if not UPLOAD_DIR.exists():
        return 0
    cutoff = time.time() - max_age_sec
    removed = 0
    for part in UPLOAD_DIR.rglob(f"*{PART_SUFFIX}"):
        try:
            if part.is_file() and part.stat().st_mtime < cutoff:
                part.unlink()
                removed += 1
        except OSError as e:
            logger.warning("partial_upload_not_removed", path=str(part), error=str(e))
    if removed:
        logger.info("partial_uploads_removed", files=removed)
    return removed


def _gb(n: int) -> str:
    """Human size: "60 MB", "10.04 GB" (two decimals, so near-limit sizes differ)."""
    if n < 1e9:
        return f"{n / 1e6:.0f} MB"
    return f"{n / 1e9:.2f} GB"


def max_upload_bytes() -> int:
    return int(settings.max_upload_gb * 1e9)


def check_room(size: int) -> None:
    """Refuse an upload that is over the limit or would fill the disk.

    Raises:
        HTTPException: 413 over the size limit, 507 not enough free space.
    """
    if size > max_upload_bytes():
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is {_gb(size)}, the limit is {_gb(max_upload_bytes())} "
                "(API_MAX_UPLOAD_GB in .env)"
            ),
        )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    needed = size * SPACE_PER_UPLOADED_BYTE + DISK_RESERVE_BYTES
    free = shutil.disk_usage(UPLOAD_DIR).free
    if needed > free:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Not enough disk space: processing a {_gb(size)} video needs about "
                f"{_gb(needed)}, {_gb(free)} is free. Delete old jobs or free up space."
            ),
        )


def _safe_filename(filename: str) -> str:
    # Strip any directory components to prevent path traversal
    safe_filename = Path(filename).name
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {ext}. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )
    return safe_filename


@router.get("/check", status_code=204)
async def check_upload(
    filename: str = Query(..., min_length=1, max_length=255),
    size: int = Query(..., ge=0),
    user: LocalUser = Depends(get_current_user),
):
    """Answer before the upload starts whether the file will be accepted.

    The browser would otherwise send gigabytes only to be refused at the end.
    """
    _safe_filename(filename)
    check_room(size)


@router.post("/", response_model=UploadResponse, status_code=201)
async def upload_file(
    request: Request,
    filename: str = Query(..., min_length=1, max_length=255),
    user: LocalUser = Depends(get_current_user),
):
    safe_filename = _safe_filename(filename)

    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_size = int(declared)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
        check_room(declared_size)

    user_dir = UPLOAD_DIR / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex[:12]}_{safe_filename}"
    file_path = user_dir / unique_name
    if not file_path.resolve().is_relative_to(user_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    part_path = file_path.with_name(file_path.name + PART_SUFFIX)

    logger.info("upload_started", filename=filename, declared_size=declared)

    limit = max_upload_bytes()
    total_size = 0
    completed = False
    try:
        with open(part_path, "wb") as f:
            async for chunk in request.stream():
                total_size += len(chunk)
                if total_size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File is over the limit of {_gb(limit)} (API_MAX_UPLOAD_GB in .env)"
                        ),
                    )
                f.write(chunk)
        if total_size == 0:
            raise HTTPException(status_code=400, detail="Empty upload")
        part_path.rename(file_path)
        completed = True
    except ClientDisconnect:
        logger.warning("upload_interrupted", filename=filename, received=total_size)
        raise HTTPException(status_code=400, detail="Upload interrupted")
    finally:
        if not completed:
            part_path.unlink(missing_ok=True)

    logger.info("upload_completed", filename=filename, path=str(file_path), size=total_size)

    return UploadResponse(
        path=str(file_path),
        filename=safe_filename,
        size=total_size,
    )
