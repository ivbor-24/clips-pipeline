"""
Scoring strategy "llm_windows": clip candidates chosen by the LLM.

Instead of scoring single sentences and gluing them into fixed tiles, the
transcript is cut into overlapping windows (~100 s, step ~50 s). For every
window the LLM picks the best clip inside it (a start line that works as a
hook, an end line that completes the thought) and rates it. Candidates are
then trimmed to the duration limits and the best non-overlapping ones are
selected.

Inputs:
- transcript segments (artifacts/transcript.json)

Outputs (written by src.scoring.score_transcript):
- artifacts/scored_segments.json
- artifacts/llm_analysis.json (debug: windows, raw LLM answers)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import structlog

from src.config import ScoringConfig

logger = structlog.get_logger("window_scoring")

# Weights of the LLM sub-scores in the clip score: the user's first
# criterion is a hook, the second a clean start and a finished thought.
SCORE_WEIGHTS = {"hook": 0.4, "complete": 0.35, "value": 0.25}

# Margins around the chosen lines (s), limited by the neighbouring lines.
PAD_START_SEC = 0.25
PAD_END_SEC = 0.4

# How far (in lines) a boundary may move to reach a sentence start/end.
MAX_SNAP_LINES = 4
_SENTENCE_END = (".", "!", "?", "…", "»", '"')

# Openings that continue a previous thought: a clip starting with them
# depends on context the viewer has not seen.
CONTINUATION_WORDS = {
    "и",
    "а",
    "но",
    "поэтому",
    "причем",
    "причём",
    "соответственно",
    "тоже",
    "также",
    "он",
    "она",
    "оно",
    "они",
    "его",
    "её",
    "ее",
    "их",
    "нет",
    "да",
    "and",
    "but",
    "so",
    "no",
    "yes",
}
CONTINUATION_PHRASES = (
    "то есть",
    "потому что",
    "так что",
    "при этом",
    "кроме того",
    "тем самым",
    "вот это",
    "в том числе",
    "that is",
    "because",
)
CONTINUATION_PENALTY = 0.25  # off both hook and complete

# A trailing question may be cut even if the clip drops to this share of
# min_duration.
QUESTION_TRIM_MIN_SHARE = 0.8


class WindowScoringError(Exception):
    """Raised when the LLM results cannot produce any clip."""

    pass


@dataclass
class Window:
    """A run of consecutive transcript segments shown to the LLM."""

    id: str
    segments: List[Dict[str, Any]]

    @property
    def start(self) -> float:
        return float(self.segments[0]["start"])

    @property
    def end(self) -> float:
        return float(self.segments[-1]["end"])


@dataclass
class ClipCandidate:
    """A clip proposed for one window, in absolute transcript indices."""

    window_id: str
    first: int
    last: int
    start: float
    end: float
    score: float
    hook: float
    complete: float
    value: float
    title: str = ""
    reason: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_windows(
    transcript: Sequence[Dict[str, Any]],
    window_sec: float,
    step_sec: float,
) -> List[Window]:
    """
    Cut the transcript into overlapping windows aligned to segment boundaries.

    Args:
        transcript: Segments ordered by time.
        window_sec: Target window length; a window takes whole segments while
            they end within window_sec of its first segment.
        step_sec: Distance between window starts.

    Returns:
        Windows w_001, w_002, ...; the last one always reaches the end.
    """
    segments = sorted(transcript, key=lambda s: s["start"])
    if not segments:
        return []

    windows: List[Window] = []
    i = 0
    while True:
        first_start = segments[i]["start"]
        j = i
        while j + 1 < len(segments) and segments[j + 1]["end"] - first_start <= window_sec:
            j += 1
        windows.append(Window(id=f"w_{len(windows) + 1:03d}", segments=segments[i : j + 1]))
        if j == len(segments) - 1:
            break
        next_start = first_start + step_sec
        k = i + 1
        while k < len(segments) and segments[k]["start"] < next_start:
            k += 1
        # Never skip past the end of this window (a single very long segment).
        i = min(k, j + 1)
    return windows


def format_windows(windows: Sequence[Window]) -> str:
    """Render windows as numbered lines with offsets for the prompt."""
    blocks = []
    for w in windows:
        lines = [f"### Window {w.id}"]
        for n, seg in enumerate(w.segments):
            offset = int(round(seg["start"] - w.start))
            lines.append(f"[{n} | {offset}s] {seg['text'].strip()}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_window_prompt(
    template: str, windows: Sequence[Window], min_duration: float, max_duration: float
) -> str:
    """Fill the v2 prompt template for a batch of windows."""
    return (
        template.replace("{{WINDOWS}}", format_windows(windows))
        .replace("{{MIN_DURATION}}", str(int(min_duration)))
        .replace("{{MAX_DURATION}}", str(int(max_duration)))
    )


def weighted_score(hook: float, complete: float, value: float) -> float:
    """Clip score from the LLM sub-scores (SCORE_WEIGHTS)."""
    return round(
        SCORE_WEIGHTS["hook"] * hook
        + SCORE_WEIGHTS["complete"] * complete
        + SCORE_WEIGHTS["value"] * value,
        3,
    )


def _unit(value: Any) -> float:
    """Coerce an LLM rating to [0, 1]; missing or invalid -> 0."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def candidate_from_result(
    window: Window,
    result: Dict[str, Any],
    index_of: Dict[str, int],
) -> Optional[ClipCandidate]:
    """
    Turn one LLM answer into a candidate with absolute segment indices.

    Args:
        window: The window the answer refers to.
        result: Parsed answer (start_line, end_line, hook, complete, value...).
        index_of: Segment id -> index in the time-ordered transcript.

    Returns:
        Candidate, or None if the line numbers are unusable.
    """
    try:
        first_line = int(result.get("start_line"))
        last_line = int(result.get("end_line"))
    except (TypeError, ValueError):
        return None
    n = len(window.segments)
    first_line = min(max(first_line, 0), n - 1)
    last_line = min(max(last_line, 0), n - 1)
    if last_line < first_line:
        return None

    hook = _unit(result.get("hook"))
    complete = _unit(result.get("complete"))
    value = _unit(result.get("value"))
    score = weighted_score(hook, complete, value)
    first_seg = window.segments[first_line]
    last_seg = window.segments[last_line]
    return ClipCandidate(
        window_id=window.id,
        first=index_of[first_seg["id"]],
        last=index_of[last_seg["id"]],
        start=float(first_seg["start"]),
        end=float(last_seg["end"]),
        score=score,
        hook=hook,
        complete=complete,
        value=value,
        title=str(result.get("title") or "")[:120],
        reason=str(result.get("reason") or "")[:300],
    )


def _words(text: str) -> List[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def starts_sentence(text: str) -> bool:
    """True unless the line continues a sentence (first letter is lowercase)."""
    match = re.search(r"[^\W\d_]", text)
    return match is None or not match.group(0).islower()


def ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(_SENTENCE_END)


def is_continuation(text: str) -> bool:
    """True if a line opens with a word that refers back to earlier speech."""
    words = _words(text)
    if not words:
        return False
    if words[0] in CONTINUATION_WORDS:
        return True
    return " ".join(words[:3]).startswith(CONTINUATION_PHRASES)


def fit_boundaries(
    candidate: ClipCandidate,
    segments: Sequence[Dict[str, Any]],
    min_duration: float,
    max_duration: float,
) -> ClipCandidate:
    """
    Align a candidate to whole sentences and bring it within the limits.

    whisper segments are sentence fragments (a third of them continue the
    previous line), and the LLM often picks clips shorter than
    min_duration; extending line by line then cut clips mid-sentence.
    Now the start moves back to the first line of its sentence, the end
    forward to a sentence end, and duration fixes stop on sentence ends.
    The hook line itself is always kept. A clip that opens with a
    continuation word ("то есть", "и", "он"...) or still mid-sentence loses
    CONTINUATION_PENALTY of both its "hook" and "complete" ratings.

    Returns:
        The same candidate, updated (tags record every adjustment).
    """
    first, last = candidate.first, candidate.last
    tags = list(candidate.tags)

    def text(i: int) -> str:
        return str(segments[i].get("text", ""))

    def span(a: int, b: int) -> float:
        return float(segments[b]["end"]) - float(segments[a]["start"])

    moved = 0
    while first > 0 and not starts_sentence(text(first)) and moved < MAX_SNAP_LINES:
        first -= 1
        moved += 1
    if moved:
        tags.append("start_to_sentence")

    def sentence_end_after(i: int, limit: int) -> Optional[int]:
        for j in range(i, min(len(segments), i + limit + 1)):
            if span(first, j) > max_duration:
                return None
            if ends_sentence(text(j)):
                return j
        return None

    if span(first, last) > max_duration:
        fits = [j for j in range(first, last) if span(first, j) <= max_duration]
        ends = [j for j in fits if ends_sentence(text(j))]
        last = (ends or fits or [first])[-1]
        tags.append("trimmed_to_max")
    elif span(first, last) < min_duration:
        j = last
        while j + 1 < len(segments) and span(first, j) < min_duration:
            if span(first, j + 1) > max_duration:
                break
            j += 1
        end = sentence_end_after(j, MAX_SNAP_LINES)
        if end is None:
            ends = [k for k in range(last, j + 1) if ends_sentence(text(k))]
            end = ends[-1] if ends else j
        if end != last:
            last = end
            tags.append("extended_to_min")
    elif not ends_sentence(text(last)):
        end = sentence_end_after(last, MAX_SNAP_LINES)
        if end is not None and end != last:
            last = end
            tags.append("end_to_sentence")

    # A clip that ends on a question stops on the next topic (in interviews
    # usually the host's next question): end on the previous sentence, even
    # a little below min_duration - a shorter clip beats a cliffhanger.
    while (
        last > first
        and text(last).rstrip().endswith("?")
        and span(first, last - 1) >= QUESTION_TRIM_MIN_SHARE * min_duration
        and ends_sentence(text(last - 1))
    ):
        last -= 1
        if "question_end_trimmed" not in tags:
            tags.append("question_end_trimmed")

    if not starts_sentence(text(first)) or is_continuation(text(first)):
        # The user's main complaint: a clip that "starts from nothing".
        candidate.hook = max(0.0, candidate.hook - CONTINUATION_PENALTY)
        candidate.complete = max(0.0, candidate.complete - CONTINUATION_PENALTY)
        candidate.score = weighted_score(candidate.hook, candidate.complete, candidate.value)
        tags.append("continuation_start")

    candidate.first, candidate.last = first, last
    candidate.start = float(segments[first]["start"])
    candidate.end = float(segments[last]["end"])
    candidate.tags = tags
    return candidate


def pad_bounds(
    candidate: ClipCandidate,
    segments: Sequence[Dict[str, Any]],
    pad_start: float = PAD_START_SEC,
    pad_end: float = PAD_END_SEC,
) -> ClipCandidate:
    """Widen a clip slightly without reaching into the neighbouring lines.

    whisper timestamps are approximate; a hard cut at the segment start can
    swallow the first syllable of the hook.
    """
    prev_end = float(segments[candidate.first - 1]["end"]) if candidate.first > 0 else 0.0
    start = max(candidate.start - pad_start, prev_end, 0.0)
    end = candidate.end + pad_end
    if candidate.last + 1 < len(segments):
        end = min(end, float(segments[candidate.last + 1]["start"]))
    candidate.start = round(min(start, candidate.start), 3)
    candidate.end = round(max(end, candidate.end), 3)
    return candidate


def select_clips(
    candidates: Sequence[ClipCandidate],
    max_clips: int,
    min_score: float,
    max_overlap_sec: float = 2.0,
) -> List[ClipCandidate]:
    """
    Pick the best candidates that do not overlap each other.

    Args:
        candidates: All candidates (any order).
        max_clips: How many to keep.
        min_score: Candidates below this are dropped.
        max_overlap_sec: Tolerated overlap between two selected clips.

    Returns:
        Selected candidates, best first.
    """
    selected: List[ClipCandidate] = []
    for cand in sorted(candidates, key=lambda c: (-c.score, c.start)):
        if cand.score < min_score:
            break
        overlaps = any(
            min(cand.end, s.end) - max(cand.start, s.start) > max_overlap_sec for s in selected
        )
        if not overlaps:
            selected.append(cand)
        if len(selected) >= max_clips:
            break
    return selected


def candidates_from_results(
    windows: Sequence[Window],
    results: Sequence[Dict[str, Any]],
    segments: Sequence[Dict[str, Any]],
    scoring: ScoringConfig,
) -> List[ClipCandidate]:
    """
    Turn raw LLM answers into boundary-fitted candidates.

    Also used to re-run the post-processing on saved answers
    (llm_analysis.json "results") without calling the LLM again.

    Args:
        windows: Windows the answers refer to (by id).
        results: Raw LLM answers.
        segments: Time-ordered transcript.
        scoring: Duration limits.

    Returns:
        Candidates, one per usable answer.
    """
    index_of = {seg["id"]: i for i, seg in enumerate(segments)}
    by_id = {w.id: w for w in windows}
    candidates: List[ClipCandidate] = []
    for result in results:
        window = by_id.get(str(result.get("id")))
        if window is None:
            continue
        cand = candidate_from_result(window, result, index_of)
        if cand is None:
            logger.warning("window_result_unusable", window_id=window.id)
            continue
        cand = fit_boundaries(cand, segments, scoring.min_duration, scoring.max_duration)
        candidates.append(pad_bounds(cand, segments))
    return candidates


def score_windows(
    transcript: Sequence[Dict[str, Any]],
    scoring: ScoringConfig,
    analyze: Callable[[List[Dict[str, Any]], str], List[Dict[str, Any]]],
    run_batches: Callable[..., Tuple[List[Dict[str, Any]], int, int]],
    prompt_template: str,
) -> Tuple[List[ClipCandidate], List[ClipCandidate], Dict[str, Any]]:
    """
    Run the llm_windows strategy.

    Args:
        transcript: Transcript segments with ids.
        scoring: Scoring configuration (window sizes, limits, max clips).
        analyze: adapter.analyze_segments-like callable (payload, prompt).
        run_batches: Batch runner with the per-batch failure policy
            (src.scoring.run_llm_batches).
        prompt_template: The v2 prompt template.

    Returns:
        (selected clips, all candidates, debug payload for llm_analysis.json)

    Raises:
        WindowScoringError: If no usable candidate came back.
    """
    segments = sorted(transcript, key=lambda s: s["start"])
    windows = build_windows(segments, scoring.window_sec, scoring.window_step_sec)
    per_call = max(1, scoring.windows_per_call)
    batches = [windows[i : i + per_call] for i in range(0, len(windows), per_call)]
    logger.info(
        "window_scoring_start",
        windows=len(windows),
        calls=len(batches),
        window_sec=scoring.window_sec,
        step_sec=scoring.window_step_sec,
    )

    def call(batch: List[Window]) -> List[Dict[str, Any]]:
        prompt = build_window_prompt(
            prompt_template, batch, scoring.min_duration, scoring.max_duration
        )
        payload = [{"id": w.id, "start": w.start, "end": w.end} for w in batch]
        return analyze(payload, prompt)

    results, ok, failed = run_batches(batches, call)

    candidates = candidates_from_results(windows, results, segments, scoring)
    if not candidates:
        raise WindowScoringError("LLM returned no usable clip candidates")

    selected = select_clips(candidates, scoring.max_clips_per_video, scoring.min_score_threshold)
    logger.info(
        "window_scoring_complete",
        candidates=len(candidates),
        selected=len(selected),
        calls_ok=ok,
        calls_failed=failed,
    )
    debug = {
        "strategy": "llm_windows",
        "score_weights": SCORE_WEIGHTS,
        "calls": {"total": len(batches), "failed": failed},
        "windows": [
            {"id": w.id, "start": w.start, "end": w.end, "segments": len(w.segments)}
            for w in windows
        ],
        "results": results,
        "candidates": [_candidate_dict(c) for c in candidates],
    }
    return selected, candidates, debug


def _candidate_dict(c: ClipCandidate) -> Dict[str, Any]:
    return {
        "window_id": c.window_id,
        "start": c.start,
        "end": c.end,
        "duration": round(c.duration, 2),
        "score": c.score,
        "hook": c.hook,
        "complete": c.complete,
        "value": c.value,
        "title": c.title,
        "reason": c.reason,
        "tags": c.tags,
    }


def to_scored_segments(selected: Sequence[ClipCandidate]) -> List[Dict[str, Any]]:
    """Serialize selected clips in the scored_segments.json contract."""
    output = []
    for rank, c in enumerate(selected, start=1):
        output.append(
            {
                "id": f"clip_{rank:03d}",
                "start": c.start,
                "end": c.end,
                "score": c.score,
                "title": c.title,
                "tags": ["llm_window"] + c.tags,
                "self_contained": c.complete >= 0.6,
                "_debug": {
                    "window_id": c.window_id,
                    "hook": c.hook,
                    "complete": c.complete,
                    "value": c.value,
                    "reason": c.reason,
                },
            }
        )
    return output


def load_prompt_template(path: str) -> str:
    """Read the window prompt template."""
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise WindowScoringError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")
