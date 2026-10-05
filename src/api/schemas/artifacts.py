from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class TranscriptResponse(BaseModel):
    segments: List[Dict[str, Any]]


class ScoredSegmentsResponse(BaseModel):
    segments: List[Dict[str, Any]]


class CropParamsResponse(BaseModel):
    crops: List[Dict[str, Any]]


class ManifestResponse(BaseModel):
    clips: List[Dict[str, Any]]
    total_clips: int


class ChaptersResponse(BaseModel):
    chapters: List[Dict[str, Any]]
    source: str = ""


class ChaptersTextResponse(BaseModel):
    text: str
    format: str = "youtube"


class BrollResponse(BaseModel):
    suggestions: List[Dict[str, Any]]
    source: str = ""


class BrollTextResponse(BaseModel):
    text: str


class Notice(BaseModel):
    """A problem that did not stop the job but changed its result (src/notices.py)."""

    model_config = ConfigDict(extra="allow")

    stage: str
    code: str
    message: str


class NoticesResponse(BaseModel):
    notices: List[Notice]


class JobLogLine(BaseModel):
    timestamp: Optional[str] = None
    level: str = "info"
    event: str = ""
    # The other fields of the structlog record.
    fields: Dict[str, Any] = {}


class JobLogResponse(BaseModel):
    lines: List[JobLogLine]
    # Pass it back as ?offset= to get only the lines written since.
    offset: int
