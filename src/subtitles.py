"""
Subtitle cues and styles for burned-in captions.

Splits transcript segments into short cues (a few lines each) and turns a
SubtitleStyle (src/config.py) into ASS style fields. Rendering writes the
cues as SRT/ASS files and burns the ASS one into the clip.

Styles are named presets plus per-field overrides
(rendering.subtitle_style / rendering.subtitle_style_overrides), so a style
picker only has to send a preset name and a few overrides.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import RenderingConfig, SubtitleStyle

# Built-in presets expressed as SubtitleStyle. The legacy presets
# ("default", "modern", "minimal") live in rendering.get_ass_style_preset.
PRESETS: Dict[str, SubtitleStyle] = {
    "typewriter": SubtitleStyle(),
}

# Wrapping for the legacy presets (proportional fonts, bottom-anchored).
LEGACY_MAX_CHARS_PER_LINE = 36
LEGACY_MAX_LINES = 2

_SENTENCE_END = (".", "!", "?", "…")
_ALIGNMENT = {"bottom": 2, "center": 5, "top": 8}


@dataclass
class Cue:
    """One subtitle event, times relative to the clip start."""

    start: float
    end: float
    lines: List[str]


def resolve_subtitle_style(config: RenderingConfig) -> Optional[SubtitleStyle]:
    """
    Return the style for a built-in preset with overrides applied.

    Args:
        config: Rendering configuration.

    Returns:
        SubtitleStyle, or None for a legacy preset (default/modern/minimal).
    """
    base = PRESETS.get(config.subtitle_style)
    if base is None:
        return None
    return SubtitleStyle(**{**base.model_dump(), **config.subtitle_style_overrides})


def ass_color(hex_color: str, opacity: float = 1.0) -> str:
    """
    Convert "#RRGGBB" and an opacity to ASS "&HAABBGGRR" (alpha 00 = opaque).

    Raises:
        ValueError: On a malformed color.
    """
    match = re.fullmatch(r"#?([0-9A-Fa-f]{6})", hex_color.strip())
    if not match:
        raise ValueError(f"Invalid color (expected #RRGGBB): {hex_color}")
    rgb = match.group(1)
    alpha = round(255 * (1.0 - opacity))
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()


def ass_style_fields(style: SubtitleStyle, width: int, height: int) -> Dict[str, Any]:
    """
    Map a SubtitleStyle to ASS "Style:" fields for a width x height frame.

    A box uses libass BorderStyle 4 (one box behind the whole event, no dark
    bands where per-line boxes of BorderStyle 3 overlap); box_padding is the
    border width.
    """
    box_colour = ass_color(style.box_color, style.box_opacity)
    return {
        "fontname": style.font,
        "fontsize": style.font_size,
        "primary_colour": ass_color(style.text_color),
        "secondary_colour": "&H000000FF",
        "outline_colour": box_colour if style.box else ass_color(style.outline_color),
        "back_colour": box_colour if style.box else "&H00000000",
        "bold": -1 if style.bold else 0,
        "italic": -1 if style.italic else 0,
        "underline": 0,
        "strikeout": 0,
        "scale_x": 100,
        "scale_y": 100,
        "spacing": 0,
        "angle": 0,
        "border_style": 4 if style.box else 1,
        "outline": style.box_padding if style.box else style.outline,
        "shadow": 0 if style.box else style.shadow,
        "alignment": _ALIGNMENT[style.anchor],
        "margin_l": round(style.margin_h * width),
        "margin_r": round(style.margin_h * width),
        "margin_v": round(style.margin_v * height),
    }


def wrap_text(text: str, max_chars: int) -> List[str]:
    """Greedy word wrap; explicit newlines stay line breaks."""
    lines: List[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}" if current else word
            if current and len(candidate) > max_chars:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def _segment_words(
    seg: Dict[str, Any], clip_start: float, clip_end: float
) -> List[Tuple[float, float, str]]:
    """Words of a segment with clip-relative times.

    Uses word timestamps when the transcript has them (faster-whisper);
    otherwise spreads the segment time over words by length (whisper.cpp).
    """
    words = seg.get("words") or []
    if words and all("start" in w and "end" in w for w in words):
        timed = [
            (float(w["start"]), float(w["end"]), str(w.get("word", "")).strip()) for w in words
        ]
    else:
        # Explicit line breaks stay as zero-length tokens.
        tokens = re.findall(r"[^\s]+|\n", str(seg.get("text", "")))
        if not any(tok != "\n" for tok in tokens):
            return []
        start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        weights = [0 if tok == "\n" else len(tok) + 1 for tok in tokens]
        total = float(sum(weights))
        timed, t = [], start
        for token, weight in zip(tokens, weights):
            dt = (end - start) * weight / total
            timed.append((t, t + dt, token))
            t += dt
    return [
        (max(0.0, s - clip_start), min(clip_end, e) - clip_start, w)
        for s, e, w in timed
        if w and clip_start <= (s + e) / 2 <= clip_end
    ]


def _chunks(words: List[str], max_chars: int, max_lines: int) -> List[List[int]]:
    """Group word indices into cues of at most max_lines wrapped lines."""
    chunks: List[List[int]] = []
    current: List[int] = []
    for i, word in enumerate(words):
        trial = current + [i]
        if current and len(wrap_text(" ".join(words[j] for j in trial), max_chars)) > max_lines:
            chunks.append(current)
            current = [i]
        else:
            current = trial
        # Start a new sentence on a new cue once the current one is nearly full.
        lines = len(wrap_text(" ".join(words[j] for j in current), max_chars))
        if word.endswith(_SENTENCE_END) and lines >= max(1, max_lines - 1):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    # A short tail flashes on screen: balance it with the previous cue.
    if len(chunks) >= 2:
        prev, last = chunks[-2], chunks[-1]
        capacity = max_chars * max_lines

        def length(idx: List[int]) -> int:
            return len(" ".join(words[j] for j in idx))

        while (
            len(prev) > 1
            and length(last) < capacity * 0.3
            and length(last) + len(words[prev[-1]]) + 1 < length(prev)
            and not words[prev[-1]].endswith(_SENTENCE_END)
        ):
            last.insert(0, prev.pop())
            if len(wrap_text(" ".join(words[j] for j in last), max_chars)) > max_lines:
                prev.append(last.pop(0))
                break
    return chunks


def build_cues(
    transcript: Sequence[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    max_chars_per_line: int,
    max_lines: int,
) -> List[Cue]:
    """
    Split the transcript inside [clip_start, clip_end] into short cues.

    Each segment is cut into cues of at most max_lines lines of at most
    max_chars_per_line characters; cues never span two segments.

    Returns:
        Cues with clip-relative times, in order.
    """
    cues: List[Cue] = []
    for seg in transcript:
        if not (seg.get("end", 0) > clip_start and seg.get("start", 0) < clip_end):
            continue
        timed = _segment_words(seg, clip_start, clip_end)
        if not timed:
            continue
        words = [w for _, _, w in timed]
        seg_cues = []
        for idx in _chunks(words, max_chars_per_line, max_lines):
            text = " ".join(words[j] for j in idx)
            seg_cues.append(
                Cue(
                    start=round(timed[idx[0]][0], 3),
                    end=round(max(timed[idx[-1]][1], timed[idx[0]][0] + 0.3), 3),
                    lines=wrap_text(text, max_chars_per_line),
                )
            )
        # Cues of one segment follow each other without gaps.
        for a, b in zip(seg_cues, seg_cues[1:]):
            a.end = b.start
        cues.extend(seg_cues)
    return cues


def escape_ass_text(text: str) -> str:
    """Neutralise ASS override braces in transcript text."""
    return text.replace("{", "(").replace("}", ")")
