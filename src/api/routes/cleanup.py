"""
Cleanup API route: run the cleanup now instead of waiting for the
API's next pass (it runs at start and every 6 hours).
"""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import get_db
from src.api.routes.auth import get_current_user
from src.api.services.auth import LocalUser
from src.api.services.cleanup import run_cleanup
from src.user_settings import load_app_config

router = APIRouter()


class CleanupResponse(BaseModel):
    retention_days: int
    cutoff_date: Optional[str]
    jobs_cleaned: int
    orphan_uploads_removed: int
    total_bytes_freed: int
    total_mb_freed: float
    uploads_current_size_gb: float


@router.delete("/", response_model=CleanupResponse)
async def cleanup_endpoint(
    user: LocalUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove the source videos of jobs older than retention_days and orphan uploads.

    Clips and job cards stay (src/api/services/cleanup.py).
    """
    config = load_app_config()
    result = await run_cleanup(db, config.cleanup)
    return result
