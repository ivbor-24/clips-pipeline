"""
Pipeline Orchestrator Module

Framework-agnostic pipeline execution logic extracted from cli.py.
No Typer or Rich imports — suitable for CLI and web UI backends.

Inputs:
- Config object
- input_source (path or URL)

Outputs:
- Artifacts in artifacts/ and output/ directories
"""

import gc
import time
from pathlib import Path
from typing import Callable, Optional

import structlog

from src.config import Config
from src.gpu_utils import GPUConfig, get_auto_gpu_config

logger = structlog.get_logger("pipeline")


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    pass


class IngestionError(PipelineError):
    """Raised when ingestion stage fails."""

    pass


class TranscriptionError(PipelineError):
    """Raised when transcription stage fails."""

    pass


class ScoringError(PipelineError):
    """Raised when scoring stage fails."""

    pass


class FaceCroppingError(PipelineError):
    """Raised when face cropping stage fails."""

    pass


class RenderingError(PipelineError):
    """Raised when rendering stage fails."""

    pass


class ChaptersError(PipelineError):
    """Raised when chapters stage fails."""

    pass


STAGE_EXCEPTIONS = {
    "ingestion": IngestionError,
    "transcription": TranscriptionError,
    "scoring": ScoringError,
    "face_cropping": FaceCroppingError,
    "rendering": RenderingError,
    "chapters": ChaptersError,
}


def _cleanup_vram(gpu_config: GPUConfig) -> None:
    gc.collect()
    gpu_config.cleanup_fn()
    logger.info("VRAM cleared", cuda_cache_cleared=True)


def _noop_callback(stage_name: str, status: str, progress_percent: Optional[float] = None) -> None:
    pass


def run_pipeline(
    config: Config,
    input_source: str,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
    progress_callback: Optional[Callable[[str, str, Optional[float]], None]] = None,
) -> None:
    """
    Execute all pipeline stages sequentially with resume/force support.

    Args:
        config: Pipeline configuration.
        input_source: Path to video file or YouTube URL.
        dry_run: If True, pass dry_run=True to each stage.
        resume: If True, skip stages with valid existing artifacts.
        force: If True, recompute all stages regardless of existing artifacts.
        progress_callback: Optional callback receiving (stage_name, status, progress_percent).

    Raises:
        IngestionError: On ingestion failure.
        TranscriptionError: On transcription failure.
        ScoringError: On scoring failure.
        FaceCroppingError: On face cropping failure.
        RenderingError: On rendering failure.
    """
    if progress_callback is None:
        progress_callback = _noop_callback

    gpu_config = get_auto_gpu_config(config.gpu)

    from src.artifact_validator import (
        get_expected_duration_from_meta,
        is_valid_crop_params,
        is_valid_manifest,
        is_valid_meta,
        is_valid_scored_segments,
        is_valid_transcript,
    )
    from src.face_cropping import FaceCroppingError as _ModuleFaceCroppingError
    from src.face_cropping import detect_faces
    from src.ingestion import IngestionError as _ModuleIngestionError
    from src.ingestion import ingest
    from src.rendering import RenderingError as _ModuleRenderingError
    from src.rendering import render_and_export
    from src.scoring import ScoringError as _ModuleScoringError
    from src.scoring import score_transcript
    from src.transcription import TranscriptionError as _ModuleTranscriptionError
    from src.transcription import transcribe

    paths = {
        "meta": config.work_dir / "artifacts" / "meta.json",
        "transcript": config.work_dir / "artifacts" / "transcript.json",
        "scored_segments": config.work_dir / "artifacts" / "scored_segments.json",
        "crop_params": config.work_dir / "artifacts" / "crop_params.json",
        "manifest": config.work_dir / config.output.manifest_file,
    }

    stages = [
        (
            "ingestion",
            lambda: ingest(input_source, config, dry_run=dry_run),
            paths["meta"],
            is_valid_meta,
        ),
        (
            "transcription",
            lambda: transcribe(config, dry_run=dry_run),
            paths["transcript"],
            lambda p: is_valid_transcript(
                p,
                expected_duration=get_expected_duration_from_meta(paths["meta"]),
            ),
        ),
        (
            "scoring",
            lambda: score_transcript(config, dry_run=dry_run),
            paths["scored_segments"],
            is_valid_scored_segments,
        ),
        (
            "face_cropping",
            lambda: detect_faces(config, dry_run=dry_run),
            paths["crop_params"],
            is_valid_crop_params,
        ),
        (
            "rendering",
            lambda: render_and_export(config, dry_run=dry_run, reuse=not force),
            paths["manifest"],
            is_valid_manifest,
        ),
    ]

    for stage_name, stage_fn, artifact_path, validator in stages:
        if not force and resume and artifact_path.exists() and validator(artifact_path):
            logger.info("skipping_stage", stage=stage_name, reason="valid_artifact_exists")
            progress_callback(stage_name, "skipped", None)
            continue

        if resume and artifact_path.exists() and not validator(artifact_path):
            logger.warning(
                "invalid_artifact_recomputing",
                stage=stage_name,
                path=str(artifact_path),
            )

        progress_callback(stage_name, "started", None)
        logger.info("starting_stage", stage=stage_name)
        start = time.time()

        exc_class = STAGE_EXCEPTIONS[stage_name]

        try:
            stage_fn()
        except (
            _ModuleIngestionError,
            _ModuleTranscriptionError,
            _ModuleScoringError,
            _ModuleFaceCroppingError,
            _ModuleRenderingError,
        ) as e:
            elapsed = time.time() - start
            logger.error(
                "stage_failed",
                stage=stage_name,
                error=str(e),
                duration_sec=round(elapsed, 2),
            )
            progress_callback(stage_name, "failed", None)
            raise exc_class(str(e)) from e
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "stage_failed_unexpected",
                stage=stage_name,
                error=str(e),
                duration_sec=round(elapsed, 2),
            )
            progress_callback(stage_name, "failed", None)
            raise exc_class(f"Unexpected error: {e}") from e

        elapsed = time.time() - start
        logger.info("stage_complete", stage=stage_name, duration_sec=round(elapsed, 2))
        progress_callback(stage_name, "completed", 100.0)

        if stage_name in ("transcription", "face_cropping"):
            _cleanup_vram(gpu_config)

    logger.info("starting_stage", stage="review")
    progress_callback("review", "started", None)
    review_start = time.time()

    try:
        from src.review import review_and_export

        review_and_export(
            clips_dir=str(config.work_dir / config.output.clips_dir),
            review_output=str(config.work_dir / config.output.review_file),
            auto_mode=True,
            work_dir=config.work_dir,
        )
        review_elapsed = time.time() - review_start
        logger.info("stage_complete", stage="review", duration_sec=round(review_elapsed, 2))
        progress_callback("review", "completed", 100.0)
    except Exception as e:
        review_elapsed = time.time() - review_start
        logger.warning("review_failed", error=str(e), duration_sec=round(review_elapsed, 2))
        progress_callback("review", "failed", None)

    logger.info("pipeline_complete")


def run_chapters_pipeline(
    config: Config,
    input_source: str,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
    progress_callback: Optional[Callable[[str, str, Optional[float]], None]] = None,
) -> None:
    """
    Execute the draft/pre-production pipeline: ingestion → transcription →
    chapters → (optional) broll suggestions.

    This is the cheap, clip-less path for planning a video before committing
    to the expensive scoring/face-cropping/rendering stages of run_pipeline:
    it only ever needs the transcript, so both chapters and broll suggestions
    live here rather than in the clips pipeline.

    Args:
        config: Pipeline configuration.
        input_source: Path to video file or YouTube URL.
        dry_run: If True, pass dry_run=True to each stage.
        resume: If True, skip stages with valid existing artifacts.
        force: If True, recompute all stages regardless of existing artifacts.
        progress_callback: Optional callback receiving (stage_name, status, progress_percent).

    Raises:
        IngestionError: On ingestion failure.
        TranscriptionError: On transcription failure.
        ChaptersError: On chapters generation failure.

    Note:
        A failed broll stage (when config.broll.enabled) only logs a warning
        and does not raise — it is a supplementary, best-effort artifact.
    """
    if progress_callback is None:
        progress_callback = _noop_callback

    gpu_config = get_auto_gpu_config(config.gpu)

    from src.artifact_validator import (
        get_expected_duration_from_meta,
        is_valid_meta,
        is_valid_transcript,
    )
    from src.chapters import ChaptersError as _ModuleChaptersError
    from src.chapters import generate_chapters
    from src.ingestion import IngestionError as _ModuleIngestionError
    from src.ingestion import ingest
    from src.transcription import TranscriptionError as _ModuleTranscriptionError
    from src.transcription import transcribe

    paths = {
        "meta": config.work_dir / "artifacts" / "meta.json",
        "transcript": config.work_dir / "artifacts" / "transcript.json",
        "chapters": config.work_dir / "artifacts" / "chapters.json",
    }

    def is_valid_chapters(p: Path) -> bool:
        if not p.exists():
            return False
        try:
            import json

            with open(p) as f:
                data = json.load(f)
            return "chapters" in data and len(data["chapters"]) > 0
        except Exception:
            return False

    stages = [
        (
            "ingestion",
            lambda: ingest(input_source, config, dry_run=dry_run),
            paths["meta"],
            is_valid_meta,
        ),
        (
            "transcription",
            lambda: transcribe(config, dry_run=dry_run),
            paths["transcript"],
            lambda p: is_valid_transcript(
                p,
                expected_duration=get_expected_duration_from_meta(paths["meta"]),
            ),
        ),
        (
            "chapters",
            lambda: generate_chapters(config, dry_run=dry_run),
            paths["chapters"],
            is_valid_chapters,
        ),
    ]

    for stage_name, stage_fn, artifact_path, validator in stages:
        if not force and resume and artifact_path.exists() and validator(artifact_path):
            logger.info("skipping_stage", stage=stage_name, reason="valid_artifact_exists")
            progress_callback(stage_name, "skipped", None)
            continue

        if resume and artifact_path.exists() and not validator(artifact_path):
            logger.warning(
                "invalid_artifact_recomputing",
                stage=stage_name,
                path=str(artifact_path),
            )

        progress_callback(stage_name, "started", None)
        logger.info("starting_stage", stage=stage_name)
        start = time.time()

        exc_class = STAGE_EXCEPTIONS[stage_name]

        try:
            stage_fn()
        except (
            _ModuleIngestionError,
            _ModuleTranscriptionError,
            _ModuleChaptersError,
        ) as e:
            elapsed = time.time() - start
            logger.error(
                "stage_failed",
                stage=stage_name,
                error=str(e),
                duration_sec=round(elapsed, 2),
            )
            progress_callback(stage_name, "failed", None)
            raise exc_class(str(e)) from e
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "stage_failed_unexpected",
                stage=stage_name,
                error=str(e),
                duration_sec=round(elapsed, 2),
            )
            progress_callback(stage_name, "failed", None)
            raise exc_class(f"Unexpected error: {e}") from e

        elapsed = time.time() - start
        logger.info("stage_complete", stage=stage_name, duration_sec=round(elapsed, 2))
        progress_callback(stage_name, "completed", 100.0)

        if stage_name == "transcription":
            _cleanup_vram(gpu_config)

    if config.broll.enabled:
        logger.info("starting_stage", stage="broll")
        progress_callback("broll", "started", None)
        broll_start = time.time()

        try:
            from src.broll import generate_broll_suggestions

            generate_broll_suggestions(config, dry_run=dry_run)
            broll_elapsed = time.time() - broll_start
            logger.info("stage_complete", stage="broll", duration_sec=round(broll_elapsed, 2))
            progress_callback("broll", "completed", 100.0)
        except Exception as e:
            broll_elapsed = time.time() - broll_start
            logger.warning("broll_failed", error=str(e), duration_sec=round(broll_elapsed, 2))
            progress_callback("broll", "failed", None)

    logger.info("chapters_pipeline_complete")
