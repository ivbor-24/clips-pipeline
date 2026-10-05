from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    # The effective config (config.yaml + hardware profile + user settings).
    config: Dict[str, Any]
    # The same without the user settings: what "changed" is measured against.
    defaults: Dict[str, Any]
    # The user settings file (data/settings.yaml): only the changed keys.
    overrides: Dict[str, Any]
    # Null, or why the user settings file was ignored (the page shows it).
    error: Optional[str] = None


class ConfigUpdate(BaseModel):
    updates: Dict[str, Any]


class ConfigResetRequest(BaseModel):
    # Dotted keys to remove from the user settings, e.g.
    # ["scoring.max_clips_per_video"]; an empty list resets everything.
    keys: List[str] = Field(default_factory=list)


class ProfileListResponse(BaseModel):
    profiles: list[str]


class ValidateRequest(BaseModel):
    config: Dict[str, Any]
    input_source: str = "/dev/null"


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = []
