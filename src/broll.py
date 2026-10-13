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
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.chapters import (
    block_end,
    format_timestamp,
    format_timestamp_mmss,
    parse_timestamp,
    resolve_block_max_chars,
)
from src.config import BrollConfig, Config
from src.notices import add_notice, clear_notices

logger = structlog.get_logger("broll")

# Whisper language codes named in the prompt; the model drifts into the
# language of the prompt's English example unless told the language outright.
_LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
    "uk": "Ukrainian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

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
    excerpt_limit: Optional[int] = None,
) -> str:
    """
    Build the b-roll suggestion prompt with transcript data.

    Args:
        transcript: List of transcript segments (the whole video or one block).
        config: B-roll configuration.
        excerpt_limit: Set when the segments are one block of a longer video:
            the most suggestions this block may return.

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

    excerpt_note = ""
    if excerpt_limit is not None:
        excerpt_note = "The transcript below is an excerpt of a longer video."
        count_instruction = (
            f"Suggest at most {excerpt_limit} moments in this excerpt, only where a visual "
            "is genuinely warranted, spread across the excerpt (not clustered together)"
        )
    elif config.max_suggestions:
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
    prompt = prompt.replace("{{EXCERPT_NOTE}}", excerpt_note)
    language_code = transcript[0].get("language") if transcript else None
    language = _LANGUAGE_NAMES.get(language_code, "the language of the transcript")
    prompt = prompt.replace("{{LANGUAGE}}", language)
    prompt = prompt.replace("{{TRANSCRIPT}}", transcript_text)

    return prompt


def _normalize_block_suggestions(
    raw: List[Dict[str, Any]], block: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Clean the LLM suggestions of one block.

    Start times are taken from the timestamp label (models miscompute
    start_sec) and snapped to a segment start of the block.

    Args:
        raw: Suggestion dicts as returned by the LLM.
        block: Transcript segments the suggestions were made for.

    Returns:
        Suggestion dicts with start_sec, suggestion and keywords.
    """
    starts = [float(seg.get("start", 0)) for seg in block]
    suggestions: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not str(item.get("suggestion", "")).strip():
            continue
        start = parse_timestamp(item.get("timestamp"))
        if start is None:
            try:
                start = float(item.get("start_sec"))
            except (TypeError, ValueError):
                continue
        keywords = item.get("keywords")
        suggestions.append(
            {
                "start_sec": min(starts, key=lambda s: abs(s - start)),
                "suggestion": str(item["suggestion"]).strip(),
                "keywords": [str(k) for k in keywords] if isinstance(keywords, list) else [],
            }
        )
    return suggestions


def _generate_broll_llm(
    adapter: Any,
    transcript: List[Dict[str, Any]],
    config: BrollConfig,
    max_block_chars: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Collect b-roll suggestions from the LLM, block by block when the
    transcript does not fit into the context.

    A failed block is skipped; the run fails only if the model is unavailable
    or more than half of the blocks failed.

    Args:
        adapter: LLM adapter.
        transcript: List of transcript segments.
        config: B-roll configuration.
        max_block_chars: Character budget per request; None sends everything at once.

    Returns:
        List of suggestion dicts with start_sec, suggestion and keywords.

    Raises:
        BrollError: If more than half of the blocks failed.
    """
    from src.llm_adapter import LLMAdapterError, LLMUnavailableError

    blocks: List[List[Dict[str, Any]]] = []
    idx = 0
    while idx < len(transcript):
        end = block_end(transcript, idx, max_block_chars)
        blocks.append(transcript[idx:end])
        idx = end

    suggestions: List[Dict[str, Any]] = []
    failed = 0
    for block_no, block in enumerate(blocks, start=1):
        excerpt_limit = None
        if len(blocks) > 1:
            if config.max_suggestions:
                share = len(block) / len(transcript)
                excerpt_limit = max(1, math.ceil(config.max_suggestions * share))
            else:
                # about one per minute keeps the JSON answer inside the context
                span_sec = block[-1].get("end", block[-1].get("start", 0)) - block[0].get(
                    "start", 0
                )
                excerpt_limit = max(1, round(span_sec / 60))

        try:
            prompt = _build_broll_prompt(block, config, excerpt_limit=excerpt_limit)
            response = adapter.analyze_segments(block, prompt=prompt)
            raw = response if isinstance(response, list) else _parse_broll_response(str(response))
        except LLMUnavailableError:
            raise
        except (LLMAdapterError, BrollError) as e:
            failed += 1
            logger.warning("broll_block_failed", block=block_no, error=str(e))
            continue

        block_suggestions = _normalize_block_suggestions(raw, block)
        logger.info(
            "broll_block_done",
            block=block_no,
            blocks=len(blocks),
            block_start_sec=block[0].get("start", 0),
            suggestions=len(block_suggestions),
        )
        suggestions.extend(block_suggestions)

    if failed * 2 > len(blocks):
        raise BrollError(f"LLM failed on {failed} of {len(blocks)} blocks")

    suggestions.sort(key=lambda s: s["start_sec"])
    if config.max_suggestions and len(suggestions) > config.max_suggestions:
        # keep the cap while staying spread across the whole video
        step = len(suggestions) / config.max_suggestions
        suggestions = [suggestions[int(i * step)] for i in range(config.max_suggestions)]

    return suggestions


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

            # Copy the scoring LLM settings whole: the local llama.cpp model
            # needs model_path/model_repo, n_ctx, seed and the rest of them.
            llm_config = config.scoring.llm.model_copy(
                update={"enabled": True, "prompt_file": broll_config.prompt_file}
            )
            max_block_chars = resolve_block_max_chars(llm_config, broll_config.max_block_chars)
            logger.info("broll_llm_blocks", max_block_chars=max_block_chars)

            adapter = create_llm_adapter(llm_config, config.work_dir, "broll")
            suggestions = _generate_broll_llm(adapter, transcript, broll_config, max_block_chars)

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

    with_hours = duration_sec >= 3600
    for s in suggestions:
        s["start_sec"] = round(float(s["start_sec"]), 2)
        s["timestamp"] = format_timestamp(s["start_sec"], with_hours)

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
