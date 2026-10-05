from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ReviewStatus(str, Enum):
    PENDING = "pending"
    KEEP = "keep"
    REJECT = "reject"
    EDIT = "edit"


class ClipResponse(BaseModel):
    id: str
    video_path: str
    srt_path: Optional[str]
    score: float
    duration: float
    tags: List[str]
    review_status: ReviewStatus
    review_notes: Optional[str] = None
    metadata: Dict[str, Any]


class ClipListResponse(BaseModel):
    clips: List[ClipResponse]
    total: int


class ClipUpdate(BaseModel):
    review_status: Optional[ReviewStatus] = None
    notes: Optional[str] = None
