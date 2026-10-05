"""
TASK-06: Rendering & Export Module

Generates final video clips using ffmpeg commands, including:
- Vertical crop 1080×1920 with padding
- SRT subtitle overlay from transcript timestamps
- Audio normalization (loudnorm=I=-14:TP=-1.5:LRA=11)
- Metadata export (render_time, score, terms_found, source_hash)
- Output validation via ffprobe
- Hardware video encoding: Quick Sync / VAAPI / NVENC when a test
  encode works, libx264 otherwise
- Per-clip resume: a clip whose inputs did not change is not rendered again

Inputs:
- artifacts/video/prep.mp4
- artifacts/scored_segments.json
- artifacts/crop_params.json
- artifacts/transcript.json
- artifacts/meta.json

Outputs:
- output/clips/clip_001.mp4
- output/clips/clip_001.srt
- output/clips/clip_001.meta.json
- output/manifest.json
"""

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from src import __version__
from src.config import Config, RenderingConfig
from src.notices import add_notice, clear_notices
from src.subtitles import (
    LEGACY_MAX_CHARS_PER_LINE,
    LEGACY_MAX_LINES,
    ass_style_fields,
    build_cues,
    escape_ass_text,
    resolve_subtitle_style,
)

logger = structlog.get_logger("rendering")

# ASS [V4+ Styles] field order; must stay on one line in the .ass file.
_ASS_STYLE_FORMAT = "Format: " + ", ".join(
    [
        "Name",
        "Fontname",
        "Fontsize",
        "PrimaryColour",
        "SecondaryColour",
        "OutlineColour",
        "BackColour",
        "Bold",
        "Italic",
        "Underline",
        "StrikeOut",
        "ScaleX",
        "ScaleY",
        "Spacing",
        "Angle",
        "BorderStyle",
        "Outline",
        "Shadow",
        "Alignment",
        "MarginL",
        "MarginR",
        "MarginV",
        "Encoding",
    ]
)

# Timeouts for external commands (seconds)
_FFPROBE_TIMEOUT = 60
_DEFAULT_RENDER_CLIP_TIMEOUT = 300


def resolve_video_encoder(codec: str) -> str:
    """Map friendly codec name to ffmpeg video encoder library."""
    mapping = {
        "h264": "libx264",
        "x264": "libx264",
        "avc": "libx264",
        "h265": "libx265",
        "hevc": "libx265",
        "x265": "libx265",
        "av1": "libsvtav1",
        "copy": "copy",
    }
    return mapping.get(codec.lower(), codec)


class RenderingError(Exception):
    """Custom exception for rendering errors."""

    pass


# GPU encoders. Crop, scale and subtitles stay on the CPU; only
# the encoder moves to the GPU. libx264 took most of the rendering time: 0.7 s
# per clip second on 6 Ivy Bridge cores; Quick Sync on Arc renders the same
# clips 2.7-3.5 times faster, VAAPI 2-2.8 times, so Quick Sync is tried first.
_HW_CODEC_FAMILY = {
    "h264": "h264",
    "x264": "h264",
    "avc": "h264",
    "h265": "hevc",
    "hevc": "hevc",
    "x265": "hevc",
    "av1": "av1",
}
_ENCODER_PROBE_TIMEOUT = 30
# Quality of each encoder's constant-quality mode, picked so that clips match
# libx264 -crf 23. Measured on 4 clips of the reference lecture (Arc B580,
# VMAF against a lossless render of the same filters): Quick Sync ICQ 22 is
# within 0.3 VMAF of libx264 at ~10% lower bitrate (23 was 0.4 lower on
# average), VAAPI CQP 23 within 0.5. NVENC is not measured: cq 23 is the
# usual equivalent.
_SOFTWARE_CRF = 23
_QSV_GLOBAL_QUALITY = 22
_VAAPI_QP = 23
_NVENC_CQ = 23
# libx264 preset names Quick Sync does not have.
_QSV_PRESETS = {"ultrafast": "veryfast", "superfast": "veryfast", "placebo": "veryslow"}


def _valid_fps(fps: Optional[str]) -> bool:
    """ffprobe's r_frame_rate ("25/1", "30000/1001"), not "0/0"."""
    try:
        num, _, den = str(fps).partition("/")
        return float(num) > 0 and float(den or 1) > 0
    except ValueError:
        return False


@dataclass(frozen=True)
class VideoEncoder:
    """An ffmpeg video encoder and the arguments it needs around the filters.

    kind: software | qsv | vaapi | nvenc; device: the VAAPI render node.
    """

    name: str
    kind: str = "software"
    device: Optional[str] = None

    def input_args(self) -> List[str]:
        """Arguments before the inputs: the device VAAPI uploads frames to."""
        if self.kind == "vaapi":
            return ["-vaapi_device", str(self.device)]
        return []

    def filter_suffix(self) -> str:
        """Filter appended to the video chain: VAAPI takes frames in GPU memory."""
        return "format=nv12,hwupload" if self.kind == "vaapi" else ""

    def output_args(self, preset: str, fps: Optional[str] = None) -> List[str]:
        """Encoder and quality arguments; ``fps`` is the source frame rate ("25/1")."""
        if self.kind == "qsv":
            args = ["-c:v", self.name, "-preset", _QSV_PRESETS.get(preset, preset)]
            args += ["-global_quality", str(_QSV_GLOBAL_QUALITY)]
            # Quick Sync refuses to open without a frame rate, and the concat
            # of a multi-shot clip leaves none.
            return args + (["-r", fps] if _valid_fps(fps) else [])
        if self.kind == "vaapi":
            return ["-c:v", self.name, "-rc_mode", "CQP", "-qp", str(_VAAPI_QP)]
        if self.kind == "nvenc":
            return [
                "-c:v",
                self.name,
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(_NVENC_CQ),
                "-b:v",
                "0",
            ]
        return ["-c:v", self.name, "-preset", preset, "-crf", str(_SOFTWARE_CRF)]


def software_encoder(config: RenderingConfig) -> VideoEncoder:
    """The CPU encoder for ``video_codec`` (libx264 for h264)."""
    return VideoEncoder(resolve_video_encoder(config.video_codec))


def _render_nodes() -> List[str]:
    """GPU render nodes that VAAPI and Quick Sync open (/dev/dri/renderD128...)."""
    return [str(p) for p in sorted(Path("/dev/dri").glob("renderD*"))]


def _gpu_backend(config: Config) -> str:
    from src.gpu_utils import get_auto_gpu_config

    try:
        return get_auto_gpu_config(config.gpu).backend
    except Exception as e:  # detection is best effort here: the encoder probe decides
        logger.warning("gpu_backend_unknown", error=str(e))
        return "cpu"


def hardware_encoders(kind: str, codec: str) -> List[VideoEncoder]:
    """Encoders of one kind (qsv, vaapi, nvenc) for a codec, one per device for VAAPI."""
    family = _HW_CODEC_FAMILY.get(codec.lower())
    if family is None:
        return []
    name = f"{family}_{kind}"
    if kind == "vaapi":
        return [VideoEncoder(name, kind, node) for node in _render_nodes()]
    return [VideoEncoder(name, kind)]


def auto_encoder_candidates(codec: str, backend: str) -> List[VideoEncoder]:
    """GPU encoders worth a test encode, best first: NVENC on NVIDIA, then Intel/AMD.

    Without /dev/dri (the CPU and CUDA images do not get it) only NVENC is tried.
    """
    candidates = []
    if backend == "cuda":
        candidates += hardware_encoders("nvenc", codec)
    if _render_nodes():
        candidates += hardware_encoders("qsv", codec) + hardware_encoders("vaapi", codec)
    return candidates


def probe_encoder(encoder: VideoEncoder, width: int, height: int) -> Optional[str]:
    """Encode half a second of black frames; None if it works, else ffmpeg's error."""
    video_filter = "format=yuv420p"
    if encoder.filter_suffix():
        video_filter += "," + encoder.filter_suffix()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *encoder.input_args(),
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r=25:d=0.5",
        "-vf",
        video_filter,
        *encoder.output_args("medium"),
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_ENCODER_PROBE_TIMEOUT)
    except FileNotFoundError:
        raise RenderingError("ffmpeg is not installed. Please install it first.")
    except subprocess.TimeoutExpired:
        return f"test encode timed out after {_ENCODER_PROBE_TIMEOUT}s"
    if result.returncode == 0:
        return None
    lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
    return " | ".join(lines[-3:]) or f"ffmpeg exited with {result.returncode}"


_ENCODER_HINT = (
    "Docker: the GPU must be passed to the worker (./setup.sh does it). Native install: "
    "Intel needs intel-media-va-driver and libmfx-gen1.2, NVIDIA a driver with NVENC."
)


def find_gpu_encoder(
    rendering: RenderingConfig, backend: str, width: int, height: int
) -> Tuple[Optional[VideoEncoder], Dict[str, str]]:
    """The first GPU encoder that ``rendering.video_encoder`` allows and that works.

    Returns (encoder or None, test-encode errors by encoder). Also used by
    scripts/check_gpu.py, so Diagnostics shows the encoder jobs will get.
    """
    choice = rendering.video_encoder
    if choice == "software" or rendering.video_codec.lower() not in _HW_CODEC_FAMILY:
        return None, {}
    candidates = (
        auto_encoder_candidates(rendering.video_codec, backend)
        if choice == "auto"
        else hardware_encoders(choice, rendering.video_codec)
    )
    errors = {}
    for encoder in candidates:
        error = probe_encoder(encoder, width, height)
        if error is None:
            return encoder, errors
        errors[f"{encoder.name}{'@' + encoder.device if encoder.device else ''}"] = error
        logger.info(
            "video_encoder_unavailable", encoder=encoder.name, device=encoder.device, error=error
        )
    return None, errors


def select_video_encoder(config: Config) -> VideoEncoder:
    """Pick the encoder for this job's clips (``rendering.video_encoder``).

    auto: the first GPU encoder whose test encode works, else libx264, with a
    notice when the machine has a GPU backend. software: libx264. qsv / vaapi /
    nvenc: that encoder; if it does not work, the job fails with the reason.
    """
    rendering = config.rendering
    choice = rendering.video_encoder
    backend = _gpu_backend(config) if choice == "auto" else ""
    encoder, errors = find_gpu_encoder(
        rendering, backend, config.cropping.output_width, config.cropping.output_height
    )
    if encoder is not None:
        logger.info("video_encoder", encoder=encoder.name, kind=encoder.kind, device=encoder.device)
        return encoder

    software = software_encoder(rendering)
    if choice in ("qsv", "vaapi", "nvenc") and rendering.video_codec.lower() in _HW_CODEC_FAMILY:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items()) or "no GPU render node found"
        raise RenderingError(
            f"Video encoder '{choice}' does not work: {detail}. {_ENCODER_HINT} "
            "Or set rendering.video_encoder to auto or software."
        )
    logger.info("video_encoder", encoder=software.name, kind="software", errors=errors or None)
    if choice == "auto" and backend != "cpu" and rendering.video_codec.lower() in _HW_CODEC_FAMILY:
        add_notice(
            config.work_dir,
            "rendering",
            "video_encoded_on_cpu",
            "Clips were encoded on the CPU: the GPU video encoder did not start, "
            "so rendering took longer. " + _ENCODER_HINT,
            backend=backend,
            errors=errors,
        )
    return software


def load_json_file(file_path: Path) -> Any:
    """Load JSON file and return parsed content."""
    if not file_path.exists():
        raise RenderingError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_timestamp_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm

    Args:
        seconds: Time in seconds

    Returns:
        SRT formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_from_transcript(
    transcript: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    max_chars_per_line: int = LEGACY_MAX_CHARS_PER_LINE,
    max_lines: int = LEGACY_MAX_LINES,
) -> str:
    """
    Generate SRT subtitle content for a specific clip segment.

    Args:
        transcript: Full transcript (word timestamps are used when present)
        clip_start: Start time of the clip in seconds
        clip_end: End time of the clip in seconds
        max_chars_per_line: Line length for wrapping
        max_lines: Lines per cue; longer segments are split into several cues

    Returns:
        SRT formatted subtitle string
    """
    cues = build_cues(transcript, clip_start, clip_end, max_chars_per_line, max_lines)
    if not cues:
        logger.warning("No transcript segments found for clip", start=clip_start, end=clip_end)
        return ""

    srt_lines = []
    for i, cue in enumerate(cues, start=1):
        srt_lines.append(f"{i}")
        srt_lines.append(f"{format_timestamp_srt(cue.start)} --> {format_timestamp_srt(cue.end)}")
        srt_lines.extend(cue.lines)
        srt_lines.append("")  # Empty line between cues

    return "\n".join(srt_lines)


def save_srt_file(srt_content: str, output_path: Path) -> None:
    """Save SRT content to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    logger.info("SRT file saved", path=str(output_path))


def format_timestamp_ass(seconds: float) -> str:
    """
    Convert seconds to ASS timestamp format: H:MM:SS.cc

    Args:
        seconds: Time in seconds

    Returns:
        ASS formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)

    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def get_ass_style_preset(style_name: str, config: RenderingConfig) -> Dict[str, Any]:
    """
    Get ASS style preset configuration.

    Args:
        style_name: Style preset name (default, modern, minimal)
        config: Rendering configuration with custom overrides

    Returns:
        Dictionary with ASS style parameters
    """
    presets = {
        "default": {
            "fontname": config.subtitle_font,
            "fontsize": config.subtitle_fontsize,
            "primary_colour": config.subtitle_primary_color,
            "secondary_colour": "&H000000FF",
            "outline_colour": config.subtitle_outline_color,
            "back_colour": "&H00000000",
            "bold": 0,
            "italic": 0,
            "underline": 0,
            "strikeout": 0,
            "scale_x": 100,
            "scale_y": 100,
            "spacing": 0,
            "angle": 0,
            "border_style": 1,
            "outline": 2,
            "shadow": 1,
            "alignment": 2,
            "margin_l": 10,
            "margin_r": 10,
            "margin_v": 50,
        },
        "modern": {
            "fontname": config.subtitle_font,
            "fontsize": config.subtitle_fontsize + 4,
            "primary_colour": config.subtitle_primary_color,
            "secondary_colour": "&H000000FF",
            "outline_colour": "&H00000000",
            "back_colour": config.subtitle_back_color,
            "bold": 1,
            "italic": 0,
            "underline": 0,
            "strikeout": 0,
            "scale_x": 100,
            "scale_y": 100,
            "spacing": 0,
            "angle": 0,
            "border_style": 3,
            "outline": 0,
            "shadow": 0,
            "alignment": 2,
            "margin_l": 20,
            "margin_r": 20,
            "margin_v": 60,
        },
        "minimal": {
            "fontname": config.subtitle_font,
            "fontsize": config.subtitle_fontsize - 8,
            "primary_colour": config.subtitle_primary_color,
            "secondary_colour": "&H000000FF",
            "outline_colour": config.subtitle_outline_color,
            "back_colour": "&H00000000",
            "bold": 0,
            "italic": 0,
            "underline": 0,
            "strikeout": 0,
            "scale_x": 100,
            "scale_y": 100,
            "spacing": 0,
            "angle": 0,
            "border_style": 1,
            "outline": 1,
            "shadow": 0,
            "alignment": 2,
            "margin_l": 10,
            "margin_r": 10,
            "margin_v": 40,
        },
    }

    return presets.get(style_name, presets["default"])


def subtitle_layout(config: RenderingConfig) -> Tuple[int, int]:
    """(max_chars_per_line, max_lines) for the configured subtitle style."""
    style = resolve_subtitle_style(config)
    if style is None:
        return LEGACY_MAX_CHARS_PER_LINE, LEGACY_MAX_LINES
    return style.max_chars_per_line, style.max_lines


def generate_ass_from_transcript(
    transcript: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    config: RenderingConfig,
    output_width: int = 1080,
    output_height: int = 1920,
) -> str:
    """
    Generate ASS subtitle content for a specific clip segment.

    Segments are split into short cues (see src/subtitles.py) wrapped by us,
    so libass does not re-wrap them (WrapStyle 2).

    Args:
        transcript: Full transcript (word timestamps are used when present)
        clip_start: Start time of the clip in seconds
        clip_end: End time of the clip in seconds
        config: Rendering configuration (subtitle_style and overrides)
        output_width: Output video width
        output_height: Output video height

    Returns:
        ASS formatted subtitle string
    """
    max_chars, max_lines = subtitle_layout(config)
    cues = build_cues(transcript, clip_start, clip_end, max_chars, max_lines)
    if not cues:
        logger.warning("No transcript segments found for ASS clip", start=clip_start, end=clip_end)
        return ""

    style_obj = resolve_subtitle_style(config)
    if style_obj is None:
        style = get_ass_style_preset(config.subtitle_style, config)
    else:
        style = ass_style_fields(style_obj, output_width, output_height)

    script_info = f"""[Script Info]
Title: Generated by AI Video Clips Pipeline
ScriptType: v4.00+
PlayResX: {output_width}
PlayResY: {output_height}
WrapStyle: 2
ScaledBorderAndShadow: yes

"""

    style_line = (
        f"Style: Default,{style['fontname']},{style['fontsize']},"
        f"{style['primary_colour']},{style['secondary_colour']},"
        f"{style['outline_colour']},{style['back_colour']},"
        f"{style['bold']},{style['italic']},{style['underline']},{style['strikeout']},"
        f"{style['scale_x']},{style['scale_y']},{style['spacing']},{style['angle']},"
        f"{style['border_style']},{style['outline']},{style['shadow']},"
        f"{style['alignment']},{style['margin_l']},{style['margin_r']},{style['margin_v']},1"
    )

    styles_section = f"""[V4+ Styles]
{_ASS_STYLE_FORMAT}
{style_line}

"""

    events_lines = [
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for cue in cues:
        text = "\\N".join(escape_ass_text(line) for line in cue.lines)
        events_lines.append(
            f"Dialogue: 0,{format_timestamp_ass(cue.start)},{format_timestamp_ass(cue.end)},"
            f"Default,,0,0,0,,{text}"
        )

    events_section = "\n".join(events_lines) + "\n"

    return script_info + styles_section + events_section


def save_ass_file(ass_content: str, output_path: Path) -> None:
    """Save ASS content to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    logger.info("ASS file saved", path=str(output_path))


def _strip_crop_window(
    video_width: int,
    video_height: int,
    output_width: int,
    output_height: int,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    """
    Compute a full-height (landscape source) / full-width (portrait source)
    crop window matching the target aspect ratio, centered on (cx, cy).

    The window covers the full shorter dimension of the source, so the output
    is filled edge-to-edge — no letterboxing.
    """
    video_aspect = video_width / video_height
    target_aspect = output_width / output_height

    if video_aspect > target_aspect:
        # Landscape source: vertical strip, full height
        w = max(2, int(round(video_height * target_aspect)))
        h = video_height
        if cx is not None:
            x = int(round(cx - w / 2))
        else:
            x = (video_width - w) // 2
        x = max(0, min(x, video_width - w))
        return w, h, x, 0
    else:
        # Portrait source: horizontal strip, full width
        h = max(2, int(round(video_width / target_aspect)))
        w = video_width
        if cy is not None:
            y = int(round(cy - h / 2))
        else:
            y = (video_height - h) // 2
        y = max(0, min(y, video_height - h))
        return w, h, 0, y


def _escape_filter_path(path: Path) -> str:
    """Escape a file path for use inside an ffmpeg filter argument."""
    return str(path).replace(":", "\\:").replace("'", "'\\\\''")


def build_crop_filter(
    crop_params: Dict[str, Any],
    video_width: int,
    video_height: int,
    output_width: int,
    output_height: int,
    clip_start: float = 0.0,
) -> Tuple[str, Optional[str]]:
    """
    Build the video crop/scale filtergraph for vertical output.

    crop_params["segments"] (optional) contains tracking segments with
    absolute video timestamps and per-segment face centers. When multiple
    segments exist, each is trimmed and cropped with its own window and the
    branches are concatenated, so shot/zoom changes in the source are
    followed instead of averaged into one wrong box.

    Returns (filter_str, output_label):
      - single window: linear chain ("crop=...,scale=...,setsar=1"), label None
      - multi-segment: trim/.../concat graph, label "vout"
    """
    segments = crop_params.get("segments") or []
    frames = crop_params.get("frames") or []

    def seg_cx_cy(seg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        if seg.get("has_face"):
            return seg.get("cx"), seg.get("cy")
        return None, None

    if len(segments) > 1:
        branches = []
        labels = []
        for i, seg in enumerate(segments):
            cx, cy = seg_cx_cy(seg)
            w, h, x, y = _strip_crop_window(
                video_width, video_height, output_width, output_height, cx, cy
            )
            start_t = round(seg.get("start", 0.0) - clip_start, 3)
            end_t = round(seg.get("end", clip_start) - clip_start, 3)
            label = f"[v{i}]"
            labels.append(label)
            branches.append(
                f"trim=start={start_t}:end={end_t},setpts=PTS-STARTPTS,"
                f"crop={w}:{h}:{x}:{y},scale={output_width}:{output_height},setsar=1{label}"
            )
        graph = ";".join(f"[0:v]{branch}" for branch in branches)
        graph += f";{''.join(labels)}concat=n={len(branches)}:v=1:a=0[vout]"
        return graph, "vout"

    # Single window: face center from the only segment or from frames.
    cx = cy = None
    if segments:
        cx, cy = seg_cx_cy(segments[0])
    elif frames:
        confident = [f for f in frames if f.get("conf", 0) > 0]
        if confident:
            cx = sum(f["x"] + f["w"] / 2 for f in confident) / len(confident)
            cy = sum(f["y"] + f["h"] / 2 for f in confident) / len(confident)

    w, h, x, y = _strip_crop_window(video_width, video_height, output_width, output_height, cx, cy)
    return (
        f"crop={w}:{h}:{x}:{y},scale={output_width}:{output_height},setsar=1",
        None,
    )


def build_audio_loudnorm_filter(config: RenderingConfig) -> str:
    """
    Build ffmpeg audio loudnorm filter string.

    Args:
        config: Rendering configuration

    Returns:
        ffmpeg audio filter string
    """
    return (
        f"loudnorm=I={config.audio_loudnorm_i}:"
        f"TP={config.audio_loudnorm_tp}:"
        f"LRA={config.audio_loudnorm_lra}"
    )


def build_render_command(
    video_path: Path,
    clip_params: Dict[str, Any],
    subtitle_path: Path,
    output_path: Path,
    config: Config,
    encoder: VideoEncoder,
    video_info: Dict[str, Any],
    subtitle_format: str = "srt",
) -> List[str]:
    """The ffmpeg command that renders one clip to ``output_path``."""
    start = clip_params.get("start", 0)
    duration = clip_params.get("end", 0) - start

    crop_params_for_clip = {
        "frames": clip_params.get("crop_frames", []),
        "segments": clip_params.get("segments", []),
        "fallback": clip_params.get("fallback", "center_crop"),
    }
    video_graph, video_label = build_crop_filter(
        crop_params_for_clip,
        video_info.get("width", 1920),
        video_info.get("height", 1080),
        config.cropping.output_width,
        config.cropping.output_height,
        clip_start=start,
    )

    if subtitle_path.exists():
        subtitle_escaped = _escape_filter_path(subtitle_path)
        if subtitle_format == "ass":
            subtitle_filter = f"ass='{subtitle_escaped}'"
            fonts_dir = config.rendering.subtitle_fonts_dir
            if fonts_dir and Path(fonts_dir).is_dir():
                subtitle_filter += f":fontsdir='{_escape_filter_path(Path(fonts_dir))}'"
        else:
            subtitle_filter = f"subtitles='{subtitle_escaped}'"
        if video_label:
            video_graph += f";[{video_label}]{subtitle_filter}[vout]"
            video_label = "vout"
        else:
            video_graph += f",{subtitle_filter}"

    upload = encoder.filter_suffix()
    if upload:
        if video_label:
            video_graph += f";[{video_label}]{upload}[venc]"
            video_label = "venc"
        else:
            video_graph += f",{upload}"

    audio_filter = build_audio_loudnorm_filter(config.rendering)

    cmd = ["ffmpeg", *encoder.input_args(), "-ss", str(start), "-i", str(video_path)]
    cmd += ["-t", str(duration)]

    if video_label:
        # Multi-segment graph: video branches + concat, audio loudnorm.
        cmd += [
            "-filter_complex",
            video_graph + f";[0:a]{audio_filter}[aout]",
            "-map",
            f"[{video_label}]",
            "-map",
            "[aout]",
        ]
    else:
        cmd += ["-vf", video_graph, "-af", audio_filter]

    cmd += encoder.output_args(config.rendering.video_preset, video_info.get("fps"))
    cmd += [
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    return cmd


def _partial_path(output_path: Path) -> Path:
    """Where ffmpeg writes a clip until it is complete (clip_001.part.mp4)."""
    return output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")


def render_clip(
    video_path: Path,
    clip_params: Dict[str, Any],
    subtitle_path: Path,
    output_path: Path,
    config: Config,
    subtitle_format: str = "srt",
    encoder: Optional[VideoEncoder] = None,
) -> bool:
    """
    Render a single clip using ffmpeg.

    ffmpeg writes ``clip_NNN.part.mp4``, renamed when it finishes, so a clip
    cut short by a stop or a crash never looks complete.

    Args:
        video_path: Path to preprocessed video
        clip_params: Clip parameters (start, end, crop info)
        subtitle_path: Path to subtitle file (SRT or ASS)
        output_path: Path for output clip
        config: Pipeline configuration
        subtitle_format: Subtitle format ("srt" or "ass")
        encoder: Video encoder (select_video_encoder); None = libx264

    Returns:
        True if rendering successful, False otherwise
    """
    start = clip_params.get("start", 0)
    end = clip_params.get("end", 0)
    encoder = encoder or software_encoder(config.rendering)

    logger.info(
        "Rendering clip",
        start=start,
        end=end,
        duration=end - start,
        encoder=encoder.name,
        output=str(output_path),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(output_path)
    partial.unlink(missing_ok=True)

    cmd = build_render_command(
        video_path,
        clip_params,
        subtitle_path,
        partial,
        config,
        encoder,
        get_video_info(video_path),
        subtitle_format=subtitle_format,
    )
    render_timeout = getattr(config.rendering, "render_timeout_sec", _DEFAULT_RENDER_CLIP_TIMEOUT)

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=render_timeout)
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg rendering failed", error=str(e), stderr=e.stderr)
        partial.unlink(missing_ok=True)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Rendering timed out", timeout=render_timeout)
        partial.unlink(missing_ok=True)
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found")
        raise RenderingError("ffmpeg is not installed. Please install it first.")

    if not partial.exists():
        logger.error("Output file not created", output=str(output_path))
        return False
    partial.replace(output_path)
    logger.info("Clip rendered successfully", output=str(output_path))
    return True


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get video resolution information using ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_FFPROBE_TIMEOUT
        )
        data = json.loads(result.stdout)

        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise RenderingError("No video stream found")

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": video_stream.get("r_frame_rate", "30/1"),
        }
    except subprocess.CalledProcessError as e:
        logger.error("ffprobe failed", error=str(e))
        raise RenderingError(f"Failed to probe video: {e}")
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timeout", timeout=_FFPROBE_TIMEOUT)
        raise RenderingError(f"ffprobe timed out after {_FFPROBE_TIMEOUT}s")


def validate_output_file(file_path: Path) -> bool:
    """
    Validate output video file using ffprobe.

    Returns True if file is valid, False otherwise.
    """
    if not file_path.exists():
        logger.error("Output file does not exist", path=str(file_path))
        return False

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_FFPROBE_TIMEOUT
        )
        data = json.loads(result.stdout)

        # Check for video and audio streams
        has_video = any(s.get("codec_type") == "video" for s in data.get("streams", []))
        has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))

        # Check duration
        duration = float(data.get("format", {}).get("duration", 0))

        if not has_video:
            logger.error("Output file has no video stream", path=str(file_path))
            return False

        if not has_audio:
            logger.warning("Output file has no audio stream", path=str(file_path))

        if duration <= 0:
            logger.error("Output file has invalid duration", path=str(file_path), duration=duration)
            return False

        logger.info(
            "Output file validated",
            path=str(file_path),
            duration=duration,
            has_video=has_video,
            has_audio=has_audio,
        )
        return True

    except subprocess.CalledProcessError as e:
        logger.error("ffprobe validation failed", error=str(e), stderr=e.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffprobe validation timeout", timeout=_FFPROBE_TIMEOUT)
        return False
    except json.JSONDecodeError as e:
        logger.error("Failed to parse ffprobe output", error=str(e))
        return False


def delete_invalid_file(file_path: Path) -> None:
    """Delete invalid output file."""
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info("Invalid file deleted", path=str(file_path))
        except Exception as e:
            logger.warning("Failed to delete invalid file", path=str(file_path), error=str(e))


def generate_clip_metadata(
    clip_params: Dict[str, Any],
    scored_segment: Dict[str, Any],
    source_meta: Dict[str, Any],
    render_time_sec: float,
    clip_index: int,
) -> Dict[str, Any]:
    """
    Generate metadata for a rendered clip.

    Args:
        clip_params: Clip rendering parameters
        scored_segment: Scored segment with score and tags
        source_meta: Source video metadata
        render_time_sec: Time taken to render in seconds
        clip_index: Index of the clip (for numbering)

    Returns:
        Metadata dictionary
    """
    # Extract terms from tags
    terms_found = []
    tags = scored_segment.get("tags", [])
    for tag in tags:
        if tag.startswith("terms:"):
            terms_found = tag.replace("terms:", "").split(",")
            break

    return {
        "clip_id": f"clip_{clip_index:03d}",
        "start_time": clip_params.get("start", 0),
        "end_time": clip_params.get("end", 0),
        "duration_sec": round(clip_params.get("end", 0) - clip_params.get("start", 0), 2),
        "score": scored_segment.get("score", 0),
        "title": scored_segment.get("title", ""),
        "tags": tags,
        "terms_found": terms_found,
        "source_hash": source_meta.get("source_hash", ""),
        "render_time_sec": round(render_time_sec, 2),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolution": [1080, 1920],
        "self_contained": scored_segment.get("self_contained", False),
    }


def save_clip_metadata(meta: Dict[str, Any], output_path: Path) -> None:
    """Save clip metadata to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Clip metadata saved", path=str(output_path))


def generate_manifest(
    clips_metadata: List[Dict[str, Any]], source_meta: Dict[str, Any], output_path: Path
) -> None:
    """
    Generate manifest.json summarizing all rendered clips.

    Args:
        clips_metadata: List of clip metadata dictionaries
        source_meta: Source video metadata
        output_path: Path for manifest file
    """
    manifest = {
        "source": source_meta.get("source", ""),
        "source_hash": source_meta.get("source_hash", ""),
        "source_duration_sec": source_meta.get("duration_sec", 0),
        "total_clips": len(clips_metadata),
        "clips": clips_metadata,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_version": __version__,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest generated", path=str(output_path), total_clips=len(clips_metadata))


# Bump when the ffmpeg command changes in a way that changes the clip.
_RENDER_KEY_VERSION = 1


def clip_render_key(
    clip_params: Dict[str, Any],
    subtitle_content: str,
    config: Config,
    source_meta: Dict[str, Any],
) -> str:
    """A fingerprint of everything that goes into a clip's video and sound.

    The encoder is left out: a clip from libx264 and one from Quick Sync are
    interchangeable. So are timeouts and the sidecar subtitle format.
    """
    rendering = config.rendering.model_dump(
        exclude={"video_encoder", "render_timeout_sec", "subtitle_format"}
    )
    payload = {
        "version": _RENDER_KEY_VERSION,
        "source_hash": source_meta.get("source_hash", ""),
        "clip": clip_params,
        "subtitles": subtitle_content,
        "rendering": rendering,
        "size": [config.cropping.output_width, config.cropping.output_height],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reusable_clip(video_path: Path, meta_path: Path, render_key: str) -> Optional[Dict[str, Any]]:
    """The metadata of a clip rendered earlier from the same inputs, if it is intact.

    ``meta.json`` is written after the video is complete, so a clip cut short by
    a stop or a crash has none and is rendered again.
    """
    if not (video_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict) or meta.get("render_key") != render_key:
        return None
    if not validate_output_file(video_path):
        return None
    return meta


def remove_stale_clips(output_dir: Path, clip_count: int) -> None:
    """Delete clips numbered above ``clip_count`` left by an earlier run with more clips."""
    for path in output_dir.glob("clip_*"):
        number = path.name[len("clip_") :].split(".", 1)[0]
        if number.isdigit() and int(number) > clip_count:
            path.unlink(missing_ok=True)
            logger.info("stale_clip_removed", path=str(path))


def render_and_export(
    config: Config, dry_run: bool = False, reuse: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Main rendering and export function.

    Args:
        config: Pipeline configuration
        dry_run: If True, validate without executing
        reuse: Keep clips rendered earlier from the same inputs (False: --force)

    Returns:
        List of clip metadata on success, None on failure

    Raises:
        RenderingError: On rendering failure
    """
    logger.info("Starting rendering and export", dry_run=dry_run)

    artifacts_dir = config.work_dir / "artifacts"
    output_dir = config.work_dir / config.output.clips_dir

    video_path = artifacts_dir / "video" / "prep.mp4"
    scored_path = artifacts_dir / "scored_segments.json"
    crop_params_path = artifacts_dir / "crop_params.json"
    transcript_path = artifacts_dir / "transcript.json"
    meta_path = artifacts_dir / "meta.json"

    # Validate inputs exist
    required_files = [video_path, scored_path, crop_params_path, transcript_path, meta_path]
    for f in required_files:
        if not f.exists():
            raise RenderingError(f"Required input file not found: {f}")

    # Load input data
    scored_segments = load_json_file(scored_path)
    crop_params_list = load_json_file(crop_params_path)
    transcript = load_json_file(transcript_path)
    source_meta = load_json_file(meta_path)

    if dry_run:
        logger.info("Dry run - skipping actual rendering")
        return [{"status": "dry_run_passed"}]

    clear_notices(config.work_dir, "rendering")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_clips(output_dir, len(scored_segments))
    for partial in output_dir.glob("clip_*.part.mp4"):
        partial.unlink(missing_ok=True)

    # Chosen before the first clip that needs rendering: a resumed job whose
    # clips are all done does not test the GPU encoder.
    encoder: Optional[VideoEncoder] = None

    # Build crop params lookup by clip_id or time range
    crop_lookup = {}
    for crop in crop_params_list:
        key = (round(crop.get("start", 0), 2), round(crop.get("end", 0), 2))
        crop_lookup[key] = crop

    clips_metadata = []
    successful_clips = 0
    reused_clips = 0

    for i, segment in enumerate(scored_segments):
        clip_index = i + 1
        clip_id = f"clip_{clip_index:03d}"

        logger.info(f"Processing {clip_id}", segment=segment)

        start = segment.get("start", 0)
        end = segment.get("end", 0)

        # Find matching crop params
        crop_key = (round(start, 2), round(end, 2))
        crop_params = crop_lookup.get(crop_key, {"frames": [], "fallback": "center_crop"})

        # Prepare clip parameters
        clip_params = {
            "start": start,
            "end": end,
            "crop_frames": crop_params.get("frames", []),
            "segments": crop_params.get("segments", []),
            "fallback": crop_params.get("fallback", "center_crop"),
        }

        # Define output paths
        clip_video_path = output_dir / f"{clip_id}.mp4"
        clip_meta_path = output_dir / f"{clip_id}.meta.json"

        # Burned-in subtitles always come from ASS so the style applies;
        # subtitle_format only picks the sidecar file delivered with the clip.
        clip_subtitle_path = output_dir / f"{clip_id}.ass"
        subtitle_content = generate_ass_from_transcript(
            transcript,
            start,
            end,
            config.rendering,
            config.cropping.output_width,
            config.cropping.output_height,
        )
        save_ass_file(subtitle_content, clip_subtitle_path)
        sidecar_paths = [clip_subtitle_path]
        if config.rendering.subtitle_format == "srt":
            max_chars, max_lines = subtitle_layout(config.rendering)
            srt_path = output_dir / f"{clip_id}.srt"
            save_srt_file(
                generate_srt_from_transcript(transcript, start, end, max_chars, max_lines),
                srt_path,
            )
            sidecar_paths.append(srt_path)

        render_key = clip_render_key(clip_params, subtitle_content, config, source_meta)
        previous = reusable_clip(clip_video_path, clip_meta_path, render_key) if reuse else None
        if previous is not None:
            # Rendered before from the same inputs (a stopped job resumed):
            # keep the video, refresh the metadata (titles may have changed).
            clip_meta = generate_clip_metadata(
                clip_params=clip_params,
                scored_segment=segment,
                source_meta=source_meta,
                render_time_sec=previous.get("render_time_sec", 0.0),
                clip_index=clip_index,
            )
            clip_meta["render_key"] = render_key
            clip_meta["encoder"] = previous.get("encoder")
            save_clip_metadata(clip_meta, clip_meta_path)
            clips_metadata.append(clip_meta)
            successful_clips += 1
            reused_clips += 1
            logger.info("clip_reused", clip_id=clip_id)
            continue
        clip_meta_path.unlink(missing_ok=True)
        if encoder is None:
            encoder = select_video_encoder(config)

        # Render clip
        render_start = time.time()
        success = render_clip(
            video_path=video_path,
            clip_params=clip_params,
            subtitle_path=clip_subtitle_path,
            output_path=clip_video_path,
            config=config,
            subtitle_format="ass",
            encoder=encoder,
        )
        if not success and encoder.kind != "software" and config.rendering.video_encoder == "auto":
            # The GPU encoder passed its test but failed on a real clip: finish
            # the job on the CPU rather than lose the clips.
            failed = encoder
            encoder = software_encoder(config.rendering)
            add_notice(
                config.work_dir,
                "rendering",
                "video_encoded_on_cpu",
                f"The GPU video encoder ({failed.name}) failed on {clip_id}; "
                "it and the clips after it were encoded on the CPU, which takes longer.",
                encoder=failed.name,
                clip_id=clip_id,
            )
            success = render_clip(
                video_path=video_path,
                clip_params=clip_params,
                subtitle_path=clip_subtitle_path,
                output_path=clip_video_path,
                config=config,
                subtitle_format="ass",
                encoder=encoder,
            )
        render_time = time.time() - render_start

        if success:
            # Validate output
            is_valid = validate_output_file(clip_video_path)

            if not is_valid:
                logger.error("Rendered clip failed validation", clip_id=clip_id)
                delete_invalid_file(clip_video_path)
                for path in sidecar_paths:
                    path.unlink(missing_ok=True)
                continue

            # Generate and save metadata
            clip_meta = generate_clip_metadata(
                clip_params=clip_params,
                scored_segment=segment,
                source_meta=source_meta,
                render_time_sec=render_time,
                clip_index=clip_index,
            )
            clip_meta["render_key"] = render_key
            clip_meta["encoder"] = encoder.name
            save_clip_metadata(clip_meta, clip_meta_path)

            clips_metadata.append(clip_meta)
            successful_clips += 1

            logger.info(
                f"{clip_id} complete",
                render_time=round(render_time, 2),
                score=segment.get("score", 0),
            )
        else:
            logger.error("Clip rendering failed", clip_id=clip_id)
            if clip_video_path.exists():
                clip_video_path.unlink()
            for path in sidecar_paths:
                path.unlink(missing_ok=True)

    if not clips_metadata:
        logger.warning("No clips were successfully rendered")
        return []

    # Generate manifest
    manifest_path = config.work_dir / config.output.manifest_file
    generate_manifest(clips_metadata, source_meta, manifest_path)

    logger.info(
        "Rendering and export complete",
        successful_clips=successful_clips,
        reused_clips=reused_clips,
        total_requested=len(scored_segments),
        encoder=encoder.name if encoder else None,
    )

    return clips_metadata
