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
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.config import ChaptersConfig, Config, LLMConfig
from src.notices import add_notice, clear_notices

logger = structlog.get_logger("chapters")

# Conservative chars-per-token for sizing transcript blocks: Cyrillic text plus
# the digit-heavy [MM:SS] labels tokenize to roughly 2-3 chars per token.
_CHARS_PER_TOKEN = 2.0
# Context reserved for the prompt template, system message and chat markup.
_PROMPT_OVERHEAD_TOKENS = 600
# Context reserved for the JSON answer of one block (a handful of chapters).
_RESPONSE_RESERVE_TOKENS = 1500


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


def format_timestamp(seconds: float, with_hours: bool = False) -> str:
    """
    Convert seconds to a chapter timestamp.

    Args:
        seconds: Time in seconds
        with_hours: Use H:MM:SS (required by YouTube/VK for videos of an hour or more)

    Returns:
        MM:SS or H:MM:SS formatted timestamp string
    """
    if not with_hours:
        return format_timestamp_mmss(seconds)
    total = int(seconds)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def parse_timestamp(value: Any) -> Optional[float]:
    """
    Parse "MM:SS" or "H:MM:SS" (minutes may exceed 59) into seconds.

    Args:
        value: Timestamp string from the LLM response

    Returns:
        Seconds, or None if the value is not a timestamp
    """
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*\[?(?:(\d+):)?(\d+):(\d{1,2})\]?\s*", value)
    if not match:
        return None
    hours, minutes, secs = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(secs)


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


def _build_chapters_prompt(
    transcript: List[Dict[str, Any]],
    config: ChaptersConfig,
    is_excerpt: bool = False,
) -> str:
    """
    Build the chapters detection prompt with transcript data.

    Args:
        transcript: List of transcript segments (the whole video or one block)
        config: Chapters configuration
        is_excerpt: True when the segments are a block of a longer video that
            continues after it

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

    first_timestamp = (
        format_timestamp_mmss(transcript[0].get("start", 0)) if transcript else "00:00"
    )
    excerpt_note = (
        "The transcript below is an excerpt of a longer video and may stop mid-topic. "
        "Only mark breaks you can see in the excerpt."
        if is_excerpt
        else ""
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
        # Local models follow an explicit count far better than a duration
        span_min = (
            (
                transcript[-1].get("end", transcript[-1].get("start", 0))
                - transcript[0].get("start", 0)
            )
            / 60
            if transcript
            else 0
        )
        if span_min >= config.min_minutes:
            low = max(1, int(span_min // config.max_minutes))
            high = max(low, math.ceil(span_min / config.min_minutes))
            count_instruction += (
                f". The transcript covers about {round(span_min)} minutes, "
                f"so return {low} to {high} chapters, no more"
            )

    prompt = template.replace("{{CHAPTER_COUNT_INSTRUCTION}}", count_instruction)
    prompt = prompt.replace("{{MIN_MINUTES}}", str(config.min_minutes))
    prompt = prompt.replace("{{MAX_MINUTES}}", str(config.max_minutes))
    prompt = prompt.replace("{{FIRST_TIMESTAMP}}", first_timestamp)
    prompt = prompt.replace("{{EXCERPT_NOTE}}", excerpt_note)
    prompt = prompt.replace("{{TRANSCRIPT}}", transcript_text)

    return prompt


def resolve_block_max_chars(llm_config: LLMConfig, override: Optional[int] = None) -> Optional[int]:
    """
    Transcript characters that fit into one LLM request.

    Args:
        llm_config: LLM configuration (provider and context size)
        override: Explicit budget from the module config (max_block_chars)

    Returns:
        Character budget per block, or None to send the transcript whole
    """
    if override is not None:
        return override
    if llm_config.provider != "llama_cpp":
        return None
    response_tokens = min(llm_config.max_tokens, _RESPONSE_RESERVE_TOKENS)
    prompt_tokens = llm_config.n_ctx - response_tokens - _PROMPT_OVERHEAD_TOKENS
    return max(1000, int(prompt_tokens * _CHARS_PER_TOKEN))


def block_end(transcript: List[Dict[str, Any]], start_idx: int, max_chars: Optional[int]) -> int:
    """Exclusive end index of the block starting at start_idx within the char budget."""
    if max_chars is None:
        return len(transcript)
    used = 0
    idx = start_idx
    while idx < len(transcript):
        # +9 for the "[MM:SS] " label and the newline
        used += len(str(transcript[idx].get("text", "")).strip()) + 9
        if used > max_chars and idx > start_idx:
            break
        idx += 1
    return idx


def _normalize_block_chapters(
    raw: List[Dict[str, Any]], block: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Clean the LLM chapters of one block.

    Start times are taken from the timestamp label (models miscompute
    start_sec), snapped to a segment start so a break never lands mid-sentence,
    and the first chapter is pinned to the block start.

    Args:
        raw: Chapter dicts as returned by the LLM
        block: Transcript segments the chapters were detected in

    Returns:
        Chapter dicts with start_sec, title and summary, in ascending order
    """
    starts = [float(seg.get("start", 0)) for seg in block]
    chapters: List[Dict[str, Any]] = []
    for ch in raw:
        if not isinstance(ch, dict) or not str(ch.get("title", "")).strip():
            continue
        start = parse_timestamp(ch.get("timestamp"))
        if start is None:
            try:
                start = float(ch.get("start_sec"))
            except (TypeError, ValueError):
                continue
        start = min(starts, key=lambda s: abs(s - start))
        if not chapters:
            start = starts[0]
        elif start <= chapters[-1]["start_sec"]:
            continue
        chapters.append(
            {
                "start_sec": start,
                "title": str(ch["title"]).strip(),
                "summary": str(ch.get("summary", "")).strip(),
            }
        )
    return chapters


def _generate_chapters_llm(
    adapter: Any,
    transcript: List[Dict[str, Any]],
    config: ChaptersConfig,
    max_block_chars: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Detect chapters with the LLM, block by block when the transcript does not
    fit into the context.

    The last chapter of a non-final block may continue past the block, so the
    next block starts from that chapter's beginning and detects it again.

    Args:
        adapter: LLM adapter
        transcript: List of transcript segments
        config: Chapters configuration
        max_block_chars: Character budget per request; None sends everything at once

    Returns:
        List of chapter dicts with start_sec, title and summary

    Raises:
        ChaptersError: If the LLM returns no usable chapters for a block
    """
    chapters: List[Dict[str, Any]] = []
    idx = 0
    block_no = 0
    while idx < len(transcript):
        end = block_end(transcript, idx, max_block_chars)
        block = transcript[idx:end]
        is_last = end >= len(transcript)
        block_no += 1

        # An exact chapter count only makes sense for the transcript as a whole
        block_config = (
            config if idx == 0 and is_last else config.model_copy(update={"target_count": None})
        )
        prompt = _build_chapters_prompt(block, block_config, is_excerpt=not is_last)
        response = adapter.analyze_segments(block, prompt=prompt)
        raw = response if isinstance(response, list) else _parse_chapters_response(str(response))
        block_chapters = _normalize_block_chapters(raw, block)
        if not block_chapters:
            raise ChaptersError(f"LLM returned no chapters for block {block_no}")

        logger.info(
            "chapters_block_done",
            block=block_no,
            block_start_sec=block[0].get("start", 0),
            block_end_sec=block[-1].get("end", block[-1].get("start", 0)),
            chapters=len(block_chapters),
        )

        next_idx = end
        if not is_last and len(block_chapters) > 1:
            carry_start = block_chapters[-1]["start_sec"]
            carry_idx = next(
                i for i in range(idx, end) if float(transcript[i].get("start", 0)) >= carry_start
            )
            if carry_idx > idx:
                block_chapters = block_chapters[:-1]
                next_idx = carry_idx

        chapters.extend(block_chapters)
        idx = next_idx

    return chapters


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

            # Copy the scoring LLM settings whole: the local llama.cpp model
            # needs model_path/model_repo, n_ctx, seed and the rest of them.
            llm_config = config.scoring.llm.model_copy(
                update={"enabled": True, "prompt_file": chapters_config.prompt_file}
            )
            max_block_chars = resolve_block_max_chars(llm_config, chapters_config.max_block_chars)
            logger.info("chapters_llm_blocks", max_block_chars=max_block_chars)

            adapter = create_llm_adapter(llm_config, config.work_dir, "chapters")
            chapters = _generate_chapters_llm(adapter, transcript, chapters_config, max_block_chars)

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

    with_hours = duration_sec >= 3600
    for ch in chapters:
        ch["start_sec"] = round(float(ch.get("start_sec", 0)), 2)
    if chapters:
        chapters[0]["start_sec"] = 0.0
    chapters = [
        {
            "timestamp": format_timestamp(ch["start_sec"], with_hours),
            "start_sec": ch["start_sec"],
            "title": ch.get("title", "Untitled"),
            "summary": ch.get("summary", ""),
        }
        for ch in chapters
    ]

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
