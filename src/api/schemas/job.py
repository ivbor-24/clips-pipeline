from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict

from src.api.models.job import JobStatus, JobType


class JobCreate(BaseModel):
    input_source: str
    job_type: Literal["clips", "chapters"] = "clips"
    config_overrides: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    status: JobStatus
    job_type: JobType
    input_source: str
    work_dir: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    # Set while the worker is stopping a running job (the status turns
    # "cancelled" once its processes have exited).
    cancel_requested: bool = False
    # The cleanup removed the source video after cleanup.retention_days: the
    # clips are kept, Resume is no longer possible.
    sources_removed_at: Optional[datetime] = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
