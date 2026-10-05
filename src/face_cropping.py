"""
TASK-05: Face Cropping & Tracking Module

Detects speaker's face in video segments, smooths bounding boxes,
and provides fallback to center crop when face detection fails.

Inputs:
- artifacts/video/prep.mp4
- artifacts/scored_segments.json

Outputs:
- artifacts/crop_params.json (list of clip crop parameters with frames and fallback info)
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import structlog

from src.config import Config, CroppingConfig
from src.temp_files import job_temp_dir

logger = structlog.get_logger("face_cropping")


@dataclass
class FrameCropInfo:
    """Represents crop information for a single frame."""

    time: float
    x: int
    y: int
    w: int
    h: int
    conf: float


@dataclass
class ClipCropParams:
    """Represents crop parameters for an entire clip segment."""

    clip_id: str
    start: float
    end: float
    frames: List[FrameCropInfo] = field(default_factory=list)
    fallback: str = "center_crop"
    fallback_ratio: float = 0.0
    # Tracking segments: [{"start", "end", "cx", "cy", "has_face"}] with
    # absolute video timestamps. Each segment gets its own crop window so
    # shot/zoom changes in the source are followed instead of averaged away.
    segments: List[Dict[str, Any]] = field(default_factory=list)


# Timeouts for external commands (seconds)
_FFPROBE_TIMEOUT = 60
_FFMPEG_TIMEOUT = 60


class FaceCroppingError(Exception):
    """Custom exception for face cropping errors."""

    pass


def load_scored_segments(scored_path: Path) -> List[Dict[str, Any]]:
    """Load scored segments from JSON file."""
    if not scored_path.exists():
        raise FaceCroppingError(f"Scored segments file not found: {scored_path}")

    with open(scored_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    logger.info("Scored segments loaded", path=str(scored_path), count=len(segments))
    return segments


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    """
    Extract a single frame from video at specified timestamp.

    Returns True if successful, False otherwise.
    """
    cmd = [
        "ffmpeg",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-q:v",
        "2",
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT)
        return output_path.exists()
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to extract frame", timestamp=timestamp, error=str(e))
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Frame extraction timeout", timestamp=timestamp, timeout=_FFMPEG_TIMEOUT)
        return False


_SCENE_CUT_RE = re.compile(r"lavfi\.scd\.time:\s*([0-9.]+)")


# Shot cuts are looked for at this rate, not on every decoded frame: with GPU
# decoding, copying every frame back from the GPU ate the gain. On the
# reference lecture the same cuts within 0.04 s (one frame at 25 fps); 5 fps
# was faster but moved crop switches by up to 0.09 s.
SCENE_FPS = 12.5
_DECODER_PROBE_TIMEOUT = 30


@dataclass(frozen=True)
class FrameDecoder:
    """How frames are decoded: ``software`` (CPU) or ``vaapi`` on a render node."""

    kind: str = "software"
    device: Optional[str] = None

    def input_args(self) -> List[str]:
        if self.kind == "vaapi":
            return [
                "-hwaccel",
                "vaapi",
                "-hwaccel_device",
                str(self.device),
                "-hwaccel_output_format",
                "vaapi",
            ]
        return []

    def downscale(self, width: Optional[int]) -> List[str]:
        """Filters that bring a decoded frame to ``width`` (None: keep) in CPU memory."""
        if self.kind == "vaapi":
            scale = [f"scale_vaapi=w={width}:h=-2"] if width else []
            return scale + ["hwdownload", "format=nv12"]
        return [f"scale={width}:-2:flags=area"] if width else []


SOFTWARE_DECODER = FrameDecoder()


def _frames_filter(
    decoder: FrameDecoder, fps: float, scene_threshold: float, frame_width: Optional[int]
) -> Tuple[str, bool]:
    """The filtergraph of extract_frames; True if it is a complex graph ([f] and [s])."""
    if scene_threshold <= 0:
        chain = [f"fps={fps}"] + decoder.downscale(frame_width)
        return ",".join(chain), False
    analysis_fps = max(SCENE_FPS, fps)
    head = ",".join([f"fps={analysis_fps}"] + decoder.downscale(frame_width))
    graph = (
        f"[0:v]{head},split=2[a][b];[a]fps={fps}[f];"
        f"[b]scale=320:-2,scdet=threshold={scene_threshold}[s]"
    )
    return graph, True


def extract_frames(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    output_dir: Path,
    scene_threshold: float = 0.0,
    decoder: FrameDecoder = SOFTWARE_DECODER,
    frame_width: Optional[int] = None,
) -> Tuple[List[Tuple[float, Path]], List[float]]:
    """
    Extract frames at ``fps`` over [start, end] with a single ffmpeg call.

    A separate ffmpeg per frame seeks and decodes from the previous keyframe
    every time (~0.9 s/frame on the run-2 machine); one call decodes the span
    once (~0.07 s/frame). With scene_threshold > 0 the same pass runs the
    scdet filter on a downscaled copy at SCENE_FPS to find shot cuts: 1 fps
    samples alone place a cut up to a second late.

    Args:
        video_path: Source video.
        start: Span start (absolute seconds).
        end: Span end (absolute seconds).
        fps: Sampling rate.
        output_dir: Directory for frame_NNNN.jpg files (old ones are removed).
        scene_threshold: scdet threshold (0-100); 0 disables cut detection.
        decoder: CPU or GPU (VAAPI) decoding, see select_frame_decoder.
        frame_width: Width of the saved frames (the face detector's input);
            None keeps the video's size.

    Returns:
        ((timestamp, path) pairs with frame k at start + k / fps, absolute cut
        times); ([], []) on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("frame_*.jpg"):
        old.unlink()

    duration = max(end - start, 0.0)
    max_frames = int(duration * fps + 1e-6) + 1
    frames_out = str(output_dir / "frame_%04d.jpg")
    graph, complex_graph = _frames_filter(decoder, fps, scene_threshold, frame_width)
    # -t on the input limits every output (the scdet branch included).
    cmd = ["ffmpeg", "-nostdin", "-nostats", *decoder.input_args()]
    cmd += ["-ss", str(start), "-t", str(duration + 1.0 / fps), "-i", str(video_path)]
    if complex_graph:
        cmd += ["-filter_complex", graph, "-map", "[f]"]
        cmd += ["-frames:v", str(max_frames), "-q:v", "2", "-y", frames_out]
        cmd += ["-map", "[s]", "-f", "null", "-"]
    else:
        cmd += ["-vf", graph, "-frames:v", str(max_frames), "-q:v", "2", "-y", frames_out]
    timeout = _FFMPEG_TIMEOUT + duration
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        logger.warning(
            "frames_extraction_failed",
            start=start,
            end=end,
            decoder=decoder.kind,
            stderr=(e.stderr or "")[-1000:],
        )
        return [], []
    except subprocess.TimeoutExpired:
        logger.warning("frames_extraction_timeout", start=start, end=end, timeout=timeout)
        return [], []

    frames = sorted(output_dir.glob("frame_*.jpg"))
    cuts = sorted(
        round(start + float(m.group(1)), 3)
        for m in _SCENE_CUT_RE.finditer(result.stderr or "")
        if float(m.group(1)) <= duration
    )
    return [(round(start + k / fps, 3), path) for k, path in enumerate(frames)], cuts


def probe_frame_decoder(decoder: FrameDecoder, video_path: Path) -> Optional[str]:
    """Decode one second of the video with ``decoder``; None if it works, else the error."""
    graph = ",".join(decoder.downscale(320)) or "null"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *decoder.input_args()]
    cmd += ["-t", "1", "-i", str(video_path), "-vf", graph, "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_DECODER_PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"test decode timed out after {_DECODER_PROBE_TIMEOUT}s"
    except FileNotFoundError:
        return "ffmpeg is not installed"
    if result.returncode == 0:
        return None
    lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
    return " | ".join(lines[-3:]) or f"ffmpeg exited with {result.returncode}"


def select_frame_decoder(config: CroppingConfig, video_path: Path) -> FrameDecoder:
    """GPU decoding (VAAPI) if a test decode of this video works, else the CPU.

    ``cropping.frame_decoder``: auto, software or vaapi (vaapi falls back to
    the CPU too: the crop is the same, only slower). NVIDIA decodes on the CPU:
    its GPU path is not measured yet.
    """
    from src import rendering  # its _render_nodes is replaced in tests

    if config.frame_decoder == "software":
        logger.info("frame_decoder", kind="software", reason="config")
        return SOFTWARE_DECODER
    errors = {}
    for node in rendering._render_nodes():
        decoder = FrameDecoder("vaapi", node)
        error = probe_frame_decoder(decoder, video_path)
        if error is None:
            logger.info("frame_decoder", kind="vaapi", device=node)
            return decoder
        errors[node] = error
    logger.info("frame_decoder", kind="software", errors=errors or None)
    return SOFTWARE_DECODER


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get video resolution and FPS information."""
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
            raise FaceCroppingError("No video stream found")

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")

        # Parse FPS
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den > 0 else 30
        else:
            fps = float(fps_str)

        return {"width": width, "height": height, "fps": fps}

    except subprocess.CalledProcessError as e:
        logger.error("ffprobe failed", error=str(e))
        raise FaceCroppingError(f"Failed to probe video: {e}")
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timeout", timeout=_FFPROBE_TIMEOUT)
        raise FaceCroppingError(f"ffprobe timed out after {_FFPROBE_TIMEOUT}s")


def compute_center_crop_params(
    width: int, height: int, target_width: int, target_height: int
) -> Dict[str, int]:
    """
    Compute center crop parameters for given video dimensions.

    Returns dict with x, y, w, h for center crop.
    """
    # Calculate scale to fit target aspect ratio
    target_aspect = target_width / target_height
    current_aspect = width / height

    if current_aspect > target_aspect:
        # Video is wider than target, scale by height
        new_width = int(height * target_aspect)
        new_height = height
        x = (width - new_width) // 2
        y = 0
    else:
        # Video is taller than target, scale by width
        new_width = width
        new_height = int(width / target_aspect)
        x = 0
        y = (height - new_height) // 2

    return {"x": x, "y": y, "w": new_width, "h": new_height}


# Face detector: YuNet from the OpenCV Zoo, MIT license. It
# replaced insightface buffalo_l, whose models allow only non-commercial use.
# It runs in OpenCV's DNN module on the CPU; the model file is ~230 KB. On the
# reference lecture (861 frames, Xeon E5 v2) it found the face on every frame
# insightface did, at 30 ms per frame instead of 386, and the crop windows
# stayed within 9 px of insightface's (95th percentile): no GPU needed.
FACE_DETECTOR_REPO = "opencv/face_detection_yunet"
FACE_DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
# Project-relative, next to the other weights (MODELS_DIR in Docker).
FACE_DETECTOR_PATH = Path("artifacts/cache/models") / FACE_DETECTOR_FILE
# Frames wider than this are downscaled for detection and the boxes scaled
# back. Full 1080p frames took 190 ms and moved the crop window by up to
# 64 px on the reference lecture; 640 (insightface's det_size too) matched
# insightface best.
DETECTION_WIDTH = 640
# YuNet's own cutoffs; cropping.face_confidence_threshold applies on top.
_YUNET_SCORE = 0.5
_YUNET_NMS = 0.3
_YUNET_TOP_K = 50


def face_detector_model() -> Path:
    """The YuNet model, downloaded from its pinned revision if it is missing.

    Jobs of the Docker worker run offline (HF_HUB_OFFLINE=1): there the model
    comes from the prefetch step, and a missing file is an error that says so.
    """
    from src.model_registry import (
        ModelIntegrityError,
        download_pinned_file,
        find_pinned_file,
        link_into_place,
        verify_file,
    )

    pin = find_pinned_file(FACE_DETECTOR_REPO, FACE_DETECTOR_FILE)
    try:
        if FACE_DETECTOR_PATH.exists():
            if pin is not None:
                verify_file(FACE_DETECTOR_PATH, pin)
            return FACE_DETECTOR_PATH
        if os.environ.get("HF_HUB_OFFLINE") == "1" or pin is None:
            raise FaceCroppingError(
                f"Face detector model not found: {FACE_DETECTOR_PATH}. Download the models: "
                "just prefetch-models (Docker: ./setup.sh)"
            )
        link_into_place(FACE_DETECTOR_PATH, download_pinned_file(pin))
    except ModelIntegrityError as e:
        raise FaceCroppingError(f"Face detector model is damaged: {e}") from e
    return FACE_DETECTOR_PATH


class FaceDetector:
    """YuNet in OpenCV: face boxes for frames of any size."""

    def __init__(self, model_path: Path, detection_width: Optional[int] = DETECTION_WIDTH):
        import cv2

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), _YUNET_SCORE, _YUNET_NMS, _YUNET_TOP_K
        )
        self._width = detection_width
        self._size: Optional[Tuple[int, int]] = None

    def detect(self, image: Any) -> List[Tuple[float, float, float, float, float]]:
        """Faces as (x1, y1, x2, y2, score) in the pixels of ``image``."""
        import cv2

        height, width = image.shape[:2]
        scale = 1.0
        if self._width and width > self._width:
            scale = self._width / width
            image = cv2.resize(
                image, (self._width, max(1, round(height * scale))), interpolation=cv2.INTER_AREA
            )
        size = (image.shape[1], image.shape[0])
        if size != self._size:
            self._detector.setInputSize(size)
            self._size = size
        _, faces = self._detector.detect(image)
        if faces is None:
            return []
        return [
            (f[0] / scale, f[1] / scale, (f[0] + f[2]) / scale, (f[1] + f[3]) / scale, float(f[14]))
            for f in faces
        ]


def load_face_detector() -> FaceDetector:
    """Load YuNet; a missing or damaged model is a FaceCroppingError with the fix."""
    import cv2

    path = face_detector_model()
    try:
        detector = FaceDetector(path)
    except cv2.error as e:
        raise FaceCroppingError(f"Failed to load the face detector {path}: {e}") from e
    logger.info(
        "face_detector_loaded",
        model=FACE_DETECTOR_FILE,
        engine="opencv_dnn",
        device="cpu",
        detection_width=DETECTION_WIDTH,
    )
    return detector


def detect_faces_in_frame(
    model: FaceDetector,
    frame_path: Path,
    confidence_threshold: float,
    video_width: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Detect faces in a single frame.

    Returns the best face detection (highest confidence) or None if no faces
    found, in the pixels of a ``video_width``-wide frame (extract_frames saves
    them downscaled to the detector's input) or of the frame itself.
    """
    try:
        import cv2

        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning("Failed to read frame", path=str(frame_path))
            return None

        faces = model.detect(frame)
        if not faces:
            return None

        x1, y1, x2, y2, score = max(faces, key=lambda f: f[4])
        if score < confidence_threshold:
            logger.debug(
                "Face confidence below threshold",
                confidence=score,
                threshold=confidence_threshold,
            )
            return None

        # A face at the frame's edge may reach past it.
        height, width = frame.shape[:2]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        k = video_width / width if video_width else 1.0
        return {
            "x": int(x1 * k),
            "y": int(y1 * k),
            "w": int((x2 - x1) * k),
            "h": int((y2 - y1) * k),
            "conf": score,
        }

    except Exception as e:
        logger.warning("Face detection failed for frame", path=str(frame_path), error=str(e))
        return None


def apply_moving_average(frames: List[FrameCropInfo], window_size: int) -> List[FrameCropInfo]:
    """
    Apply moving average smoothing to frame coordinates.

    This reduces jitter in face tracking across consecutive frames.
    """
    if len(frames) <= window_size:
        return frames

    smoothed = []

    for i, frame in enumerate(frames):
        # Determine window bounds
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(frames), i + window_size // 2 + 1)

        # Calculate average coordinates within window
        avg_x = sum(frames[j].x for j in range(start_idx, end_idx)) // (end_idx - start_idx)
        avg_y = sum(frames[j].y for j in range(start_idx, end_idx)) // (end_idx - start_idx)
        avg_w = sum(frames[j].w for j in range(start_idx, end_idx)) // (end_idx - start_idx)
        avg_h = sum(frames[j].h for j in range(start_idx, end_idx)) // (end_idx - start_idx)
        avg_conf = sum(frames[j].conf for j in range(start_idx, end_idx)) / (end_idx - start_idx)

        smoothed.append(
            FrameCropInfo(time=frame.time, x=avg_x, y=avg_y, w=avg_w, h=avg_h, conf=avg_conf)
        )

    return smoothed


def _is_shot_change(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    size_ratio_threshold: float,
    center_jump_threshold: float,
) -> bool:
    """Compare two face boxes given as (cx, cy, w)."""
    a_w, b_w = max(a[2], 1.0), max(b[2], 1.0)
    size_ratio = max(a_w / b_w, b_w / a_w)
    center_jump = abs(a[0] - b[0]) / max(a_w, b_w)
    return size_ratio > size_ratio_threshold or center_jump > center_jump_threshold


def _group_box(group: List[FrameCropInfo]) -> Optional[Tuple[float, float, float]]:
    """Median face box (cx, cy, w) of a group, or None if faces are a minority.

    The median ignores single outliers (a stray detection on the other shot
    or a wrong face) that would drag a mean away from the speaker.
    """
    confident = [f for f in group if f.conf > 0]
    if not confident or len(confident) * 2 < len(group):
        return None
    return (
        statistics.median(f.x + f.w / 2 for f in confident),
        statistics.median(f.y + f.h / 2 for f in confident),
        statistics.median(f.w for f in confident),
    )


def build_track_segments(
    frames: List[FrameCropInfo],
    clip_start: float,
    clip_end: float,
    size_ratio_threshold: float = 1.35,
    center_jump_threshold: float = 0.5,
    min_segment_sec: float = 2.0,
    cuts: Optional[Sequence[float]] = None,
    window_width: Optional[float] = None,
    min_reframe_share: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Group consecutive detection frames into tracking segments.

    Splitting runs on RAW detections: smoothing first would smear a hard cut
    over the averaging window and shift the boundary by seconds. A new segment
    starts when the face box size (shot/zoom change) or its center (cut/pan)
    jumps, or when the face appears/disappears. Segments shorter than
    ``min_segment_sec`` (a missed detection, a stray box) merge into the
    longer neighbour; neighbours that then show the same shot are joined.

    Args:
        frames: Raw per-sample detections (conf == 0 means no face).
        clip_start: Clip start (absolute seconds); the first segment always
            starts here even if the first sample could not be extracted.
        clip_end: Clip end (absolute seconds).
        size_ratio_threshold: Box width ratio treated as a shot change.
        center_jump_threshold: Center jump (share of box width) treated as a cut.
        min_segment_sec: Minimum segment duration.
        cuts: Exact shot-cut times (scdet). A boundary moves to a cut found
            between the last sample of one segment and the first of the next,
            so the window switches on the cut instead of up to a sample
            interval later (a wrong crop flashing for a fraction of a second).
            None = cut detection was not run.
        window_width: Crop window width in source pixels. With cuts known,
            a boundary without a cut (the speaker moved within one shot) is
            kept only if the face center shifts by more than
            min_reframe_share of it; otherwise the window would jerk with no
            cut to hide the move.
        min_reframe_share: See window_width.

    Returns:
        List of {"start", "end", "cx", "cy", "has_face"} with absolute video
        timestamps; cx/cy are the median face center, None without a face.
    """
    if not frames:
        return [{"start": clip_start, "end": clip_end, "cx": None, "cy": None, "has_face": False}]

    def box(f: FrameCropInfo) -> Tuple[float, float, float]:
        return (f.x + f.w / 2, f.y + f.h / 2, float(f.w))

    groups: List[List[FrameCropInfo]] = [[frames[0]]]
    for f in frames[1:]:
        prev = groups[-1][-1]
        if (prev.conf > 0) != (f.conf > 0):
            split = True
        elif f.conf <= 0:
            split = False
        else:
            split = _is_shot_change(box(prev), box(f), size_ratio_threshold, center_jump_threshold)
        if split:
            groups.append([f])
        else:
            groups[-1].append(f)

    def spans() -> List[Tuple[float, float]]:
        starts = [clip_start] + [g[0].time for g in groups[1:]]
        ends = starts[1:] + [clip_end]
        return list(zip(starts, ends))

    # Absorb short segments into the longer neighbour, shortest first.
    while len(groups) > 1:
        durations = [end - start for start, end in spans()]
        i = min(range(len(groups)), key=durations.__getitem__)
        if durations[i] >= min_segment_sec:
            break
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if durations[i - 1] >= durations[i + 1] else i + 1
        lo, hi = min(i, j), max(i, j)
        groups[lo : hi + 1] = [groups[lo] + groups[hi]]

    # Join neighbours that show the same shot once an outlier between them is gone.
    joined: List[List[FrameCropInfo]] = [groups[0]]
    for group in groups[1:]:
        a, b = _group_box(joined[-1]), _group_box(group)
        same_shot = (a is None and b is None) or (
            a is not None
            and b is not None
            and not _is_shot_change(a, b, size_ratio_threshold, center_jump_threshold)
        )
        if same_shot:
            joined[-1] = joined[-1] + group
        else:
            joined.append(group)
    groups = joined

    bounds = [list(span) for span in spans()]
    on_cut = [True] * len(groups)
    for i in range(1, len(groups)):
        prev_last, next_first = groups[i - 1][-1].time, groups[i][0].time
        # The fps filter takes the frame nearest to each tick, so a sample
        # may show up to half an interval after its nominal time.
        half = (next_first - prev_last) / 2
        between = [c for c in (cuts or ()) if prev_last - half < c <= next_first + half]
        if between:
            bounds[i - 1][1] = bounds[i][0] = between[-1]
        else:
            on_cut[i] = False

    if cuts is not None and window_width:
        # Same shot (no cut), same framing, small move: keep one window.
        kept_groups, kept_bounds = [groups[0]], [bounds[0]]
        for group, bound, cut in zip(groups[1:], bounds[1:], on_cut[1:]):
            a, b = _group_box(kept_groups[-1]), _group_box(group)
            small_move = (
                not cut
                and a is not None
                and b is not None
                and max(a[2], b[2]) / max(min(a[2], b[2]), 1.0) <= size_ratio_threshold
                and abs(a[0] - b[0]) < min_reframe_share * window_width
            )
            if small_move:
                kept_groups[-1] = kept_groups[-1] + group
                kept_bounds[-1][1] = bound[1]
            else:
                kept_groups.append(group)
                kept_bounds.append(bound)
        groups, bounds = kept_groups, kept_bounds

    result = []
    for (seg_start, seg_end), group in zip(bounds, groups):
        center = _group_box(group)
        result.append(
            {
                "start": seg_start,
                "end": seg_end,
                "cx": round(center[0], 1) if center else None,
                "cy": round(center[1], 1) if center else None,
                "has_face": center is not None,
            }
        )

    return result


def process_clip_segment(
    video_path: Path,
    segment: Dict[str, Any],
    video_info: Dict[str, Any],
    model: Any,
    config: CroppingConfig,
    temp_dir: Path,
    decoder: FrameDecoder = SOFTWARE_DECODER,
) -> ClipCropParams:
    """
    Process a single clip segment to extract face crop parameters.

    Samples frames at sample_fps, detects faces, and applies smoothing.
    Frames are saved at the detector's width (DETECTION_WIDTH) and face boxes
    scaled back to the video's pixels.
    """
    clip_id = segment.get("id", f"clip_{segment['start']}_{segment['end']}")
    start = segment["start"]
    end = segment["end"]
    duration = end - start

    logger.info("Processing clip segment", clip_id=clip_id, start=start, end=end, duration=duration)

    # Sample frames at specified FPS
    sample_interval = 1.0 / config.sample_fps
    frame_times = []
    current_time = start
    while current_time <= end:
        frame_times.append(current_time)
        current_time += sample_interval

    # Ensure we have at least one frame
    if not frame_times:
        frame_times = [start]

    logger.debug("clip_frames_sampling", clip_id=clip_id, frames=len(frame_times))

    # One ffmpeg call for the whole clip; per-frame seeks only as a fallback.
    video_width = int(video_info["width"])
    frame_width = DETECTION_WIDTH if DETECTION_WIDTH and video_width > DETECTION_WIDTH else None
    sampled, cuts = extract_frames(
        video_path,
        start,
        end,
        config.sample_fps,
        temp_dir,
        config.scene_cut_threshold,
        decoder=decoder,
        frame_width=frame_width,
    )
    if not sampled and decoder.kind != "software":
        # The GPU decoder passed its test but failed here: same frames from the CPU.
        logger.warning("frame_decoder_failed", clip_id=clip_id, decoder=decoder.kind)
        sampled, cuts = extract_frames(
            video_path,
            start,
            end,
            config.sample_fps,
            temp_dir,
            config.scene_cut_threshold,
            frame_width=frame_width,
        )
    if not sampled:
        sampled = []
        for i, t in enumerate(frame_times):
            frame_path = temp_dir / f"frame_{i:04d}.jpg"
            if extract_frame(video_path, t, frame_path):
                sampled.append((t, frame_path))
            else:
                logger.warning("Failed to extract frame", time=t)

    # Detect faces in sampled frames
    frame_results = []
    low_confidence_count = 0

    for t, frame_path in sampled:
        # Detect face
        face_info = detect_faces_in_frame(
            model, frame_path, config.face_confidence_threshold, video_width=video_width
        )

        if face_info is None:
            low_confidence_count += 1
            # Use center crop for this frame
            center_params = compute_center_crop_params(
                video_info["width"], video_info["height"], config.output_width, config.output_height
            )
            frame_results.append(
                FrameCropInfo(
                    time=t,
                    x=center_params["x"],
                    y=center_params["y"],
                    w=center_params["w"],
                    h=center_params["h"],
                    conf=0.0,
                )
            )
        else:
            frame_results.append(
                FrameCropInfo(
                    time=t,
                    x=face_info["x"],
                    y=face_info["y"],
                    w=face_info["w"],
                    h=face_info["h"],
                    conf=face_info["conf"],
                )
            )

        # Cleanup temporary frame
        try:
            frame_path.unlink()
        except Exception:
            pass

    # Tracking segments are split on raw detections (see build_track_segments).
    raw_frames = list(frame_results)

    # Calculate fallback ratio
    total_frames = len(frame_results)
    fallback_ratio = low_confidence_count / total_frames if total_frames > 0 else 1.0

    # Determine if fallback should be used
    fallback_threshold = 0.7  # 70% as per spec
    use_fallback = fallback_ratio > fallback_threshold

    if use_fallback:
        logger.warning(
            "Fallback to center crop triggered",
            clip_id=clip_id,
            fallback_ratio=fallback_ratio,
            threshold=fallback_threshold,
        )

        # Replace all frames with center crop
        center_params = compute_center_crop_params(
            video_info["width"], video_info["height"], config.output_width, config.output_height
        )

        frame_results = [
            FrameCropInfo(
                time=f.time,
                x=center_params["x"],
                y=center_params["y"],
                w=center_params["w"],
                h=center_params["h"],
                conf=0.0,
            )
            for f in frame_results
        ]

        fallback_mode = "center_crop"
        track_segments = build_track_segments([], start, end)
    else:
        # Smoothed boxes are kept in "frames" for single-window consumers.
        frame_results = apply_moving_average(frame_results, config.moving_average_window)
        fallback_mode = "none"
        track_segments = build_track_segments(
            raw_frames,
            start,
            end,
            size_ratio_threshold=config.shot_size_ratio_threshold,
            center_jump_threshold=config.shot_center_jump_threshold,
            min_segment_sec=config.min_track_segment_sec,
            cuts=cuts if config.scene_cut_threshold > 0 else None,
            window_width=min(
                float(video_info["width"]),
                video_info["height"] * config.output_width / config.output_height,
            ),
            min_reframe_share=config.min_reframe_share,
        )

    return ClipCropParams(
        clip_id=clip_id,
        start=start,
        end=end,
        frames=frame_results,
        fallback=fallback_mode,
        fallback_ratio=fallback_ratio,
        segments=track_segments,
    )


def save_crop_params(params: List[ClipCropParams], output_path: Path) -> None:
    """Save crop parameters to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    output_data = []
    for clip in params:
        clip_dict = {
            "clip_id": clip.clip_id,
            "start": clip.start,
            "end": clip.end,
            "frames": [
                {"time": f.time, "x": f.x, "y": f.y, "w": f.w, "h": f.h, "conf": f.conf}
                for f in clip.frames
            ],
            "fallback": clip.fallback,
            "fallback_ratio": clip.fallback_ratio,
            "segments": clip.segments,
        }
        output_data.append(clip_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Crop parameters saved", path=str(output_path), clips=len(params))


def detect_faces(config: Config, dry_run: bool = False) -> Optional[List[Dict[str, Any]]]:
    """
    Main face detection function.

    Args:
        config: Pipeline configuration
        dry_run: If True, validate without executing

    Returns:
        Crop parameters list on success, None on failure

    Raises:
        FaceCroppingError: On detection failure
    """
    logger.info("Starting face detection", dry_run=dry_run)

    artifacts_dir = config.work_dir / "artifacts"
    video_path = artifacts_dir / "video" / "prep.mp4"
    scored_path = artifacts_dir / "scored_segments.json"
    crop_params_path = artifacts_dir / "crop_params.json"

    # Validate inputs exist
    if not video_path.exists():
        raise FaceCroppingError(f"Video file not found: {video_path}")

    if not scored_path.exists():
        raise FaceCroppingError(f"Scored segments file not found: {scored_path}")

    if dry_run:
        logger.info("Dry run - skipping actual face detection")
        return [{"status": "dry_run_passed"}]

    # Load scored segments
    segments = load_scored_segments(scored_path)

    if not segments:
        logger.warning("No segments to process")
        # Still write an empty artifact so downstream stages can complete
        # gracefully instead of failing on a missing input file.
        save_crop_params([], crop_params_path)
        return []

    # Get video info
    video_info = get_video_info(video_path)
    logger.info(
        "Video info loaded",
        width=video_info["width"],
        height=video_info["height"],
        fps=video_info["fps"],
    )

    # Frames go to the job's own scratch directory, not /tmp (RAM on many
    # distributions): the worker removes it even if the job is killed.
    temp_dir = Path(tempfile.mkdtemp(prefix="face_crop_", dir=job_temp_dir(config.work_dir)))

    try:
        model = load_face_detector()
        decoder = select_frame_decoder(config.cropping, video_path)

        # Process each segment
        crop_params = []

        for segment in segments:
            clip_params = process_clip_segment(
                video_path, segment, video_info, model, config.cropping, temp_dir, decoder
            )
            crop_params.append(clip_params)

        # Save results
        save_crop_params(crop_params, crop_params_path)

        logger.info("Face detection complete", clips_processed=len(crop_params))

        # Return serializable format
        return [
            {
                "clip_id": cp.clip_id,
                "start": cp.start,
                "end": cp.end,
                "frames": [
                    {"time": f.time, "x": f.x, "y": f.y, "w": f.w, "h": f.h, "conf": f.conf}
                    for f in cp.frames
                ],
                "fallback": cp.fallback,
                "fallback_ratio": cp.fallback_ratio,
            }
            for cp in crop_params
        ]

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
