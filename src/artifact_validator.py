"""
TASK-E2E-02: Artifact Validation Module

Validates pipeline artifacts at multiple levels:
- L1: File exists, non-empty, valid JSON
- L2: JSON has expected structure (keys, types, non-empty lists)
- L3: Semantic validation (e.g., transcript duration matches meta)

Used by the pipeline orchestrator for strict resume decisions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger("artifact_validator")


class ArtifactValidationError(Exception):
    """Custom exception for artifact validation errors."""

    pass


def is_valid_json_file(path: Path) -> bool:
    """L1: Check that file exists, is non-empty, and contains valid JSON."""
    if not path.exists():
        logger.debug("artifact_not_found", path=str(path))
        return False

    if path.stat().st_size == 0:
        logger.debug("artifact_empty", path=str(path))
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug("artifact_invalid_json", path=str(path), error=str(e))
        return False


def _load_json(path: Path) -> Any:
    """Load JSON file. Assumes L1 validation already passed."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_meta(path: Path) -> bool:
    """L1 + L2: Validate meta.json structure."""
    if not is_valid_json_file(path):
        return False

    data = _load_json(path)

    if not isinstance(data, dict):
        logger.debug("meta_not_dict", path=str(path))
        return False

    required_keys = {"source", "duration_sec", "resolution", "fps"}
    missing = required_keys - set(data.keys())
    if missing:
        logger.debug("meta_missing_keys", path=str(path), missing=list(missing))
        return False

    if not isinstance(data["duration_sec"], (int, float)) or data["duration_sec"] <= 0:
        logger.debug("meta_invalid_duration", path=str(path), duration=data["duration_sec"])
        return False

    if not isinstance(data["resolution"], list) or len(data["resolution"]) != 2:
        logger.debug("meta_invalid_resolution", path=str(path))
        return False

    logger.info("meta_validated", path=str(path))
    return True


def is_valid_transcript(
    path: Path, expected_duration: float = 0, tolerance_percent: float = 2.0
) -> bool:
    """L1 + L2 + L3: Validate transcript.json with optional duration check."""
    if not is_valid_json_file(path):
        return False

    data = _load_json(path)

    if not isinstance(data, list) or len(data) == 0:
        logger.debug("transcript_not_list_or_empty", path=str(path))
        return False

    # L2: Check structure of first segment
    first = data[0]
    if not isinstance(first, dict):
        logger.debug("transcript_segment_not_dict", path=str(path))
        return False

    required_segment_keys = {"start", "end", "text"}
    if not required_segment_keys.issubset(set(first.keys())):
        logger.debug("transcript_missing_segment_keys", path=str(path))
        return False

    # L3: Duration validation if expected_duration provided
    if expected_duration > 0:
        if not transcript_reaches_end(data, expected_duration, tolerance_percent):
            logger.debug("transcript_duration_mismatch", path=str(path), expected=expected_duration)
            return False

    logger.info("transcript_validated", path=str(path), segments=len(data))
    return True


def transcript_reaches_end(
    segments: List[Dict[str, Any]], expected_duration: float, tolerance_percent: float = 2.0
) -> bool:
    """L3: Check that the transcript covers the video up to its end.

    Compares the end of the last segment with the video duration. A truncated
    transcript (a failed chunk, an interrupted run) stops early; pauses inside
    the talk do not matter. Measuring the share of time covered by speech, as
    before, failed on every real lecture: pauses alone take more than 2%.

    Allowed silence at the end: 10% of the video, at least 10 s, at most 60 s.
    Segments ending past the video (plus tolerance_percent) are invalid too.
    """
    if not segments or expected_duration <= 0:
        return False

    last_end = max(float(seg["end"]) for seg in segments)
    allowed_tail = min(60.0, max(10.0, expected_duration * 0.10))
    overrun_limit = expected_duration * (1 + tolerance_percent / 100) + 1.0
    return expected_duration - allowed_tail <= last_end <= overrun_limit


def is_valid_scored_segments(path: Path) -> bool:
    """L1 + L2: Validate scored_segments.json structure."""
    if not is_valid_json_file(path):
        return False

    data = _load_json(path)

    if not isinstance(data, list) or len(data) == 0:
        logger.debug("scored_segments_not_list_or_empty", path=str(path))
        return False

    # L2: Check structure of first segment
    first = data[0]
    if not isinstance(first, dict):
        logger.debug("scored_segment_not_dict", path=str(path))
        return False

    required_keys = {"id", "start", "end", "score"}
    missing = required_keys - set(first.keys())
    if missing:
        logger.debug("scored_segments_missing_keys", path=str(path), missing=list(missing))
        return False

    if not isinstance(first["score"], (int, float)):
        logger.debug("scored_segments_invalid_score", path=str(path))
        return False

    logger.info("scored_segments_validated", path=str(path), count=len(data))
    return True


def is_valid_crop_params(path: Path) -> bool:
    """L1 + L2: Validate crop_params.json structure."""
    if not is_valid_json_file(path):
        return False

    data = _load_json(path)

    if not isinstance(data, list) or len(data) == 0:
        logger.debug("crop_params_not_list_or_empty", path=str(path))
        return False

    # L2: Check structure of first entry
    first = data[0]
    if not isinstance(first, dict):
        logger.debug("crop_params_entry_not_dict", path=str(path))
        return False

    required_keys = {"clip_id", "start", "end", "frames", "fallback"}
    missing = required_keys - set(first.keys())
    if missing:
        logger.debug("crop_params_missing_keys", path=str(path), missing=list(missing))
        return False

    if not isinstance(first["frames"], list):
        logger.debug("crop_params_frames_not_list", path=str(path))
        return False

    logger.info("crop_params_validated", path=str(path), count=len(data))
    return True


def is_valid_manifest(path: Path) -> bool:
    """L1 + L2: Validate manifest.json structure."""
    if not is_valid_json_file(path):
        return False

    data = _load_json(path)

    if not isinstance(data, dict):
        logger.debug("manifest_not_dict", path=str(path))
        return False

    required_keys = {"total_clips", "clips"}
    missing = required_keys - set(data.keys())
    if missing:
        logger.debug("manifest_missing_keys", path=str(path), missing=list(missing))
        return False

    if not isinstance(data["total_clips"], int) or data["total_clips"] < 0:
        logger.debug("manifest_invalid_total_clips", path=str(path))
        return False

    if not isinstance(data["clips"], list):
        logger.debug("manifest_clips_not_list", path=str(path))
        return False

    logger.info("manifest_validated", path=str(path), total_clips=data["total_clips"])
    return True


def get_expected_duration_from_meta(meta_path: Path) -> float:
    """Extract expected duration from meta.json for L3 transcript validation."""
    if not is_valid_meta(meta_path):
        return 0.0

    data = _load_json(meta_path)
    return float(data.get("duration_sec", 0))
