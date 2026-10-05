"""
P2-13: Chapters Mode Module

Generates video chapters/outline with timestamps using LLM analysis.

Inputs:
- artifacts/transcript.json
- artifacts/meta.json

Outputs:
- artifacts/chapters.json
- artifacts/chapters.txt (YouTube/VK format)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.config import ChaptersConfig, Config, LLMConfig
from src.notices import add_notice, clear_notices

logger = structlog.get_logger("chapters")


class ChaptersError(Exception):
    """Custom exception for chapters module errors."""

    pass


def format_timestamp_mmss(seconds: float) -> str:
    """
    Convert seconds to MM:SS format for YouTube chapters.

    Args:
        seconds: Time in seconds

    Returns:
        MM:SS formatted timestamp string
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_chapters_youtube(chapters: List[Dict[str, Any]]) -> str:
    """
    Format chapters for YouTube/VK description.

    Args:
        chapters: List of chapter dicts with 'timestamp' and 'title'

    Returns:
        Formatted string with timestamps and titles
    """
    lines = []
    for ch in chapters:
        timestamp = ch.get("timestamp", "00:00")
        title = ch.get("title", "Untitled")
        lines.append(f"{timestamp} {title}")
    return "\n".join(lines)


def validate_chapters(chapters: List[Dict[str, Any]], duration_sec: float) -> bool:
    """
    Validate chapters against video duration and rules.

    Args:
        chapters: List of chapter dicts
        duration_sec: Total video duration in seconds

    Returns:
        True if valid, False otherwise
    """
    if not chapters:
        logger.warning("No chapters to validate")
        return False

    first_start = chapters[0].get("start_sec", -1)
    if first_start != 0:
        logger.warning("First chapter does not start at 00:00", start=first_start)
        return False

    for i, ch in enumerate(chapters):
        start = ch.get("start_sec", -1)
        if start < 0 or start > duration_sec:
            logger.warning("Chapter out of bounds", index=i, start=start, duration=duration_sec)
            return False

    for i in range(len(chapters) - 1):
        curr_end = chapters[i + 1].get("start_sec", 0)
        curr_start = chapters[i].get("start_sec", 0)
        if curr_end <= curr_start:
            logger.warning("Chapters overlap or out of order", index=i)
            return False

    return True


def _build_chapters_prompt(transcript: List[Dict[str, Any]], config: ChaptersConfig) -> str:
    """
    Build the chapters detection prompt with transcript data.

    Args:
        transcript: List of transcript segments
        config: Chapters configuration

    Returns:
        Formatted prompt string
    """
    prompt_path = Path(config.prompt_file)
    if not prompt_path.exists():
        raise ChaptersError(f"Prompt file not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")

    transcript_text = "\n".join(
        f"[{format_timestamp_mmss(seg.get('start', 0))}] {seg.get('text', '').strip()}"
        for seg in transcript
    )

    if config.target_count:
        count_instruction = (
            f"Create exactly {config.target_count} chapters, "
            "spaced naturally by topic across the whole video"
        )
    else:
        count_instruction = (
            f"Each chapter should be {config.min_minutes} to {config.max_minutes} minutes long"
        )

    prompt = template.replace("{{CHAPTER_COUNT_INSTRUCTION}}", count_instruction)
    prompt = prompt.replace("{{MIN_MINUTES}}", str(config.min_minutes))
    prompt = prompt.replace("{{MAX_MINUTES}}", str(config.max_minutes))
    prompt = prompt.replace("{{TRANSCRIPT}}", transcript_text)

    return prompt


def _parse_chapters_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse LLM response into structured chapters.

    Args:
        response_text: Raw LLM response text

    Returns:
        List of chapter dicts
    """
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        result = json.loads(text)

        if isinstance(result, dict) and "chapters" in result:
            return result["chapters"]
        if isinstance(result, list):
            return result

        raise ChaptersError("Unexpected response format")
    except json.JSONDecodeError as e:
        logger.error("chapters_response_parse_failed", error=str(e))
        raise ChaptersError(f"Failed to parse LLM response as JSON: {e}") from e


def _generate_chapters_fallback(
    transcript: List[Dict[str, Any]], duration_sec: float, config: ChaptersConfig
) -> List[Dict[str, Any]]:
    """
    Generate simple time-based chapters as fallback when LLM is unavailable.

    Args:
        transcript: List of transcript segments
        duration_sec: Total video duration
        config: Chapters configuration

    Returns:
        List of chapter dicts
    """
    if config.target_count:
        num_chapters = max(1, config.target_count)
    else:
        target_duration_sec = (config.min_minutes + config.max_minutes) * 60 / 2
        num_chapters = max(1, int(duration_sec / target_duration_sec))
    chapter_duration = duration_sec / num_chapters

    chapters = []
    for i in range(num_chapters):
        start_sec = i * chapter_duration
        chapters.append(
            {
                "timestamp": format_timestamp_mmss(start_sec),
                "start_sec": round(start_sec, 2),
                "title": f"Part {i + 1}",
                "summary": f"Segment {i + 1} of {num_chapters}",
            }
        )

    return chapters


def generate_chapters(config: Config, dry_run: bool = False) -> Optional[List[Dict[str, Any]]]:
    """
    Main chapters generation function.

    Args:
        config: Pipeline configuration
        dry_run: If True, validate without executing

    Returns:
        List of chapter dicts on success, None on failure

    Raises:
        ChaptersError: On chapters generation failure
    """
    logger.info("Starting chapters generation", dry_run=dry_run)

    artifacts_dir = config.work_dir / "artifacts"
    transcript_path = artifacts_dir / "transcript.json"
    meta_path = artifacts_dir / "meta.json"

    if not transcript_path.exists():
        raise ChaptersError(f"Transcript not found: {transcript_path}")
    if not meta_path.exists():
        raise ChaptersError(f"Meta not found: {meta_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    duration_sec = meta.get("duration_sec", 0)
    if duration_sec <= 0:
        raise ChaptersError("Invalid video duration in meta.json")

    if dry_run:
        logger.info("Dry run - skipping chapters generation")
        return [{"status": "dry_run_passed"}]

    clear_notices(config.work_dir, "chapters")
    chapters_config = config.chapters

    if config.scoring.llm.enabled:
        logger.info("Using LLM for chapters generation")
        adapter = None
        try:
            from src.llm_adapter import LLMAdapterError, create_llm_adapter

            llm_config = LLMConfig(
                enabled=True,
                provider=config.scoring.llm.provider,
                model=config.scoring.llm.model,
                api_key=config.scoring.llm.api_key,
                api_base=config.scoring.llm.api_base,
                temperature=config.scoring.llm.temperature,
                max_tokens=config.scoring.llm.max_tokens,
                prompt_file=chapters_config.prompt_file,
            )

            adapter = create_llm_adapter(llm_config, config.work_dir, "chapters")
            prompt = _build_chapters_prompt(transcript, chapters_config)

            response = adapter.analyze_segments(transcript, prompt=prompt)
            if isinstance(response, list):
                chapters = response
            else:
                chapters = _parse_chapters_response(str(response))

        except (LLMAdapterError, ChaptersError) as e:
            add_notice(
                config.work_dir,
                "chapters",
                "llm_failed_fallback_chapters",
                "The LLM failed, so chapters were split evenly by time instead of by topic. "
                "Check the model and the GPU (just check-gpu), then run the job again.",
                error=str(e),
            )
            chapters = _generate_chapters_fallback(transcript, duration_sec, chapters_config)
        finally:
            if adapter is not None:
                adapter.unload()
    else:
        logger.info("LLM disabled, using fallback chapters generation")
        chapters = _generate_chapters_fallback(transcript, duration_sec, chapters_config)

    if not validate_chapters(chapters, duration_sec):
        logger.warning("Chapters validation failed, adjusting")
        if chapters and chapters[0].get("start_sec", -1) != 0:
            chapters[0]["start_sec"] = 0
            chapters[0]["timestamp"] = "00:00"

    chapters_json_path = artifacts_dir / "chapters.json"
    with open(chapters_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"chapters": chapters, "source": meta.get("source", "")},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Chapters JSON saved", path=str(chapters_json_path))

    if chapters_config.output_format in ("youtube", "both"):
        youtube_content = format_chapters_youtube(chapters)
        chapters_txt_path = artifacts_dir / "chapters.txt"
        with open(chapters_txt_path, "w", encoding="utf-8") as f:
            f.write(youtube_content)
        logger.info("Chapters YouTube format saved", path=str(chapters_txt_path))

    logger.info("Chapters generation complete", total_chapters=len(chapters))
    return chapters
