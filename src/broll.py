"""
B-Roll Suggestions Module

Generates advisory B-roll footage suggestions with timestamps using LLM
analysis of the transcript. This module only proposes ideas for an editor
to act on manually — it does not source, download, or insert any footage.

Inputs:
- artifacts/transcript.json
- artifacts/meta.json

Outputs:
- artifacts/broll_suggestions.json
- artifacts/broll_suggestions.txt
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.chapters import format_timestamp_mmss
from src.config import BrollConfig, Config, LLMConfig
from src.notices import add_notice, clear_notices

logger = structlog.get_logger("broll")

# Used only by the non-LLM fallback, which needs a concrete slice size even
# when the user leaves max_suggestions unset (model's discretion).
_DEFAULT_FALLBACK_COUNT = 15


class BrollError(Exception):
    """Custom exception for b-roll suggestion module errors."""

    pass


def format_broll_text(suggestions: List[Dict[str, Any]]) -> str:
    """
    Format b-roll suggestions as a plain-text artifact (one line per suggestion).

    Args:
        suggestions: List of suggestion dicts with 'timestamp', 'suggestion',
            and optional 'keywords'.

    Returns:
        Formatted string, one suggestion per line.
    """
    lines = []
    for s in suggestions:
        timestamp = s.get("timestamp", "00:00")
        text = s.get("suggestion", "")
        keywords = s.get("keywords") or []
        line = f"{timestamp} {text}"
        if keywords:
            line += f" [{', '.join(keywords)}]"
        lines.append(line)
    return "\n".join(lines)


def validate_broll_suggestions(
    suggestions: List[Dict[str, Any]],
    duration_sec: float,
) -> bool:
    """
    Validate b-roll suggestions against video duration.

    Args:
        suggestions: List of suggestion dicts.
        duration_sec: Total video duration in seconds.

    Returns:
        True if every suggestion is well-formed and in bounds, False otherwise.
    """
    if not suggestions:
        logger.warning("No b-roll suggestions to validate")
        return False

    for i, s in enumerate(suggestions):
        start = s.get("start_sec", -1)
        if not isinstance(start, (int, float)) or start < 0 or start > duration_sec:
            logger.warning("Suggestion out of bounds", index=i, start=start, duration=duration_sec)
            return False
        if not s.get("suggestion"):
            logger.warning("Suggestion missing text", index=i)
            return False

    return True


def _build_broll_prompt(
    transcript: List[Dict[str, Any]],
    config: BrollConfig,
) -> str:
    """
    Build the b-roll suggestion prompt with transcript data.

    Args:
        transcript: List of transcript segments.
        config: B-roll configuration.

    Returns:
        Formatted prompt string.
    """
    prompt_path = Path(config.prompt_file)
    if not prompt_path.exists():
        raise BrollError(f"Prompt file not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")

    transcript_text = "\n".join(
        f"[{format_timestamp_mmss(seg.get('start', 0))}] {seg.get('text', '').strip()}"
        for seg in transcript
    )

    if config.max_suggestions:
        count_instruction = (
            f"Suggest at most {config.max_suggestions} moments, "
            "spread across the whole video (not clustered together)"
        )
    else:
        count_instruction = (
            "Suggest as many moments as genuinely warrant a visual, spread across the whole "
            "video (not clustered together) — use your judgment on how many is appropriate "
            "for this video's length and content density"
        )

    prompt = template.replace("{{COUNT_INSTRUCTION}}", count_instruction)
    prompt = prompt.replace("{{TRANSCRIPT}}", transcript_text)

    return prompt


def _parse_broll_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse LLM response into structured b-roll suggestions.

    Args:
        response_text: Raw LLM response text.

    Returns:
        List of suggestion dicts.
    """
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        result = json.loads(text)

        if isinstance(result, dict) and "suggestions" in result:
            return result["suggestions"]
        if isinstance(result, list):
            return result

        raise BrollError("Unexpected response format")
    except json.JSONDecodeError as e:
        logger.error("broll_response_parse_failed", error=str(e))
        raise BrollError(f"Failed to parse LLM response as JSON: {e}") from e


def _generate_broll_fallback(
    transcript: List[Dict[str, Any]],
    config: BrollConfig,
) -> List[Dict[str, Any]]:
    """
    Generate simple keyword-based b-roll suggestions when LLM is unavailable.

    Picks up to `max_suggestions` transcript segments spread evenly across
    the video and turns each segment's leading words into a generic suggestion.

    Args:
        transcript: List of transcript segments.
        config: B-roll configuration.

    Returns:
        List of suggestion dicts.
    """
    if not transcript:
        return []

    max_suggestions = config.max_suggestions or _DEFAULT_FALLBACK_COUNT
    step = max(1, len(transcript) // max_suggestions)
    picked = transcript[::step][:max_suggestions]

    suggestions = []
    for seg in picked:
        start_sec = seg.get("start", 0)
        text = seg.get("text", "").strip()
        words = text.split()
        snippet = " ".join(words[:6])
        suggestions.append(
            {
                "timestamp": format_timestamp_mmss(start_sec),
                "start_sec": round(start_sec, 2),
                "suggestion": (
                    f'Illustrative footage for: "{snippet}"' if snippet else "Illustrative footage"
                ),
                "keywords": words[:3],
            }
        )

    return suggestions


def generate_broll_suggestions(
    config: Config,
    dry_run: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Main b-roll suggestion generation function.

    Args:
        config: Pipeline configuration.
        dry_run: If True, skip generation and return immediately.

    Returns:
        List of suggestion dicts on success, None on failure.

    Raises:
        BrollError: On b-roll suggestion generation failure.
    """
    logger.info("Starting b-roll suggestion generation", dry_run=dry_run)

    if dry_run:
        logger.info("Dry run - skipping b-roll suggestion generation")
        return [{"status": "dry_run_passed"}]

    clear_notices(config.work_dir, "broll")
    artifacts_dir = config.work_dir / "artifacts"
    transcript_path = artifacts_dir / "transcript.json"
    meta_path = artifacts_dir / "meta.json"

    if not transcript_path.exists():
        raise BrollError(f"Transcript not found: {transcript_path}")
    if not meta_path.exists():
        raise BrollError(f"Meta not found: {meta_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    duration_sec = meta.get("duration_sec", 0)
    if duration_sec <= 0:
        raise BrollError("Invalid video duration in meta.json")

    broll_config = config.broll

    if config.scoring.llm.enabled:
        logger.info("Using LLM for b-roll suggestions")
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
                prompt_file=broll_config.prompt_file,
            )

            adapter = create_llm_adapter(llm_config, config.work_dir, "broll")
            prompt = _build_broll_prompt(transcript, broll_config)

            response = adapter.analyze_segments(transcript, prompt=prompt)
            if isinstance(response, list):
                suggestions = response
            else:
                suggestions = _parse_broll_response(str(response))

        except (LLMAdapterError, BrollError) as e:
            add_notice(
                config.work_dir,
                "broll",
                "llm_failed_fallback_broll",
                "The LLM failed, so b-roll suggestions are generic keyword hints. "
                "Check the model and the GPU (just check-gpu), then run the job again.",
                error=str(e),
            )
            suggestions = _generate_broll_fallback(transcript, broll_config)
        finally:
            if adapter is not None:
                adapter.unload()
    else:
        logger.info("LLM disabled, using fallback b-roll suggestions")
        suggestions = _generate_broll_fallback(transcript, broll_config)

    if not validate_broll_suggestions(suggestions, duration_sec):
        logger.warning("Some b-roll suggestions are invalid, filtering them out")
        suggestions = [
            s
            for s in suggestions
            if isinstance(s.get("start_sec"), (int, float))
            and 0 <= s["start_sec"] <= duration_sec
            and s.get("suggestion")
        ]

    suggestions.sort(key=lambda s: s.get("start_sec", 0))

    broll_json_path = artifacts_dir / "broll_suggestions.json"
    with open(broll_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"suggestions": suggestions, "source": meta.get("source", "")},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("B-roll suggestions JSON saved", path=str(broll_json_path))

    broll_txt_path = artifacts_dir / "broll_suggestions.txt"
    with open(broll_txt_path, "w", encoding="utf-8") as f:
        f.write(format_broll_text(suggestions))
    logger.info("B-roll suggestions text saved", path=str(broll_txt_path))

    logger.info("B-roll suggestion generation complete", total_suggestions=len(suggestions))
    return suggestions
