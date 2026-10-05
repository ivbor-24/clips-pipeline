"""
TASK-02: Ingestion & Preprocessing Module

Handles video ingestion from local files or YouTube URLs,
extracts metadata, and converts to intermediate formats.

Outputs:
- artifacts/video/prep.mp4 (h264 <= video_max_height; remuxed if the source already is)
- artifacts/audio/raw.wav (16kHz, mono, PCM16)
- artifacts/meta.json (duration_sec, resolution, fps, source_hash)
"""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from src.config import Config, PreprocessingConfig

logger = structlog.get_logger("ingestion")

# Timeouts for external commands (seconds)
_FFPROBE_TIMEOUT = 60
_FFMPEG_TIMEOUT = 7200
_YTDLP_TIMEOUT = 600


class IngestionError(Exception):
    """Custom exception for ingestion errors."""

    pass


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract metadata from video file using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_FFPROBE_TIMEOUT
        )
        data = json.loads(result.stdout)

        # Extract video stream info
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise IngestionError("No video stream found in file")

        duration = float(data.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "0/1")

        # Parse FPS
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den > 0 else 0
        else:
            fps = float(fps_str)

        audio_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
        )

        return {
            "duration_sec": round(duration, 2),
            "resolution": [width, height],
            "fps": round(fps, 2),
            "video_codec": video_stream.get("codec_name"),
            "pix_fmt": video_stream.get("pix_fmt"),
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        }
    except subprocess.CalledProcessError as e:
        logger.error("ffprobe failed", error=str(e), stderr=e.stderr)
        raise IngestionError(f"Failed to probe video: {e}")
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timeout", timeout=_FFPROBE_TIMEOUT)
        raise IngestionError(f"ffprobe timed out after {_FFPROBE_TIMEOUT}s")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse ffprobe output", error=str(e))
        raise IngestionError(f"Invalid ffprobe output: {e}")


def download_youtube_video(url: str, output_path: Path, temp_dir: Path) -> Path:
    """Download video from YouTube URL using yt-dlp."""
    logger.info("Downloading YouTube video", url=url)

    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "yt-dlp",
            "-f",
            "best[ext=mp4]/best",
            "-o",
            str(temp_dir / "downloaded.%(ext)s"),
            "--no-playlist",
            "--merge-output-format",
            "mp4",
            url,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=_YTDLP_TIMEOUT)

        # Find the downloaded file
        downloaded_files = list(temp_dir.glob("downloaded.*"))
        if not downloaded_files:
            raise IngestionError("yt-dlp did not produce output file")

        # Move to final location (outside temp_dir)
        downloaded_file = downloaded_files[0]
        final_path = output_path.with_suffix(downloaded_file.suffix)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_file.rename(final_path)

        # Cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("YouTube download complete", path=str(final_path))
        return final_path

    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timeout", timeout=_YTDLP_TIMEOUT)
        raise IngestionError(f"yt-dlp timed out after {_YTDLP_TIMEOUT}s")
    except subprocess.CalledProcessError as e:
        logger.error("yt-dlp failed", error=str(e), stderr=e.stderr)
        raise IngestionError(f"Failed to download YouTube video: {e}")
    except FileNotFoundError:
        logger.error("yt-dlp not found")
        raise IngestionError("yt-dlp is not installed. Please install it first.")


def can_copy_video(source_meta: Dict[str, Any], config: PreprocessingConfig) -> bool:
    """Return True if the source video stream can be used without re-encoding.

    Args:
        source_meta: Output of get_video_metadata() for the source.
        config: Preprocessing configuration.

    Returns:
        True for 8-bit 4:2:0 H.264 no taller than video_max_height (what a
        re-encode would produce anyway) when copy_compatible_video is on.
    """
    if not config.copy_compatible_video:
        return False
    height = (source_meta.get("resolution") or [0, 0])[1]
    return (
        source_meta.get("video_codec") == "h264"
        and source_meta.get("pix_fmt") == "yuv420p"
        and 0 < height <= config.video_max_height
    )


def preprocess_video(
    input_path: Path,
    output_path: Path,
    config: PreprocessingConfig,
    source_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Preprocess video into an h264 MP4 no taller than video_max_height.

    Args:
        input_path: Source video.
        output_path: Destination (artifacts/video/prep.mp4).
        config: Preprocessing configuration.
        source_meta: Source metadata; when it shows a compatible stream
            (see can_copy_video) the video is remuxed instead of re-encoded,
            which avoids a slow, lossy second generation.
    """
    copy_video = source_meta is not None and can_copy_video(source_meta, config)
    logger.info(
        "Preprocessing video",
        input=str(input_path),
        output=str(output_path),
        mode="remux" if copy_video else "reencode",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-i", str(input_path)]
    if copy_video:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy"]
        if source_meta.get("audio_codec") == "aac":
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += ["-movflags", "+faststart"]
    else:
        cmd += [
            # Downscale only: a smaller source is not blown up to max height.
            "-vf",
            f"scale=-2:'min({config.video_max_height},ih)'",
            "-c:v",
            config.video_codec,
            "-crf",
            str(config.video_crf),
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
    cmd += ["-y", str(output_path)]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT)
        logger.info("Video preprocessing complete", output=str(output_path))
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg video preprocessing timed out", timeout=_FFMPEG_TIMEOUT)
        raise IngestionError(f"ffmpeg video preprocessing timed out after {_FFMPEG_TIMEOUT}s")
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg video processing failed", error=str(e), stderr=e.stderr)
        raise IngestionError(f"Failed to preprocess video: {e}")
    except FileNotFoundError:
        logger.error("ffmpeg not found")
        raise IngestionError("ffmpeg is not installed. Please install it first.")


def extract_audio(video_path: Path, output_path: Path, config: PreprocessingConfig) -> None:
    """Extract audio from video: 16kHz, mono, PCM16 WAV."""
    logger.info("Extracting audio", input=str(video_path), output=str(output_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # PCM16
        "-ar",
        str(config.audio_sample_rate),
        "-ac",
        str(config.audio_channels),
        "-y",  # Overwrite output
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT)
        logger.info("Audio extraction complete", output=str(output_path))
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg audio extraction timed out", timeout=600)
        raise IngestionError("Audio extraction timed out (10 minutes)")
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg audio extraction failed", error=str(e), stderr=e.stderr)
        raise IngestionError(f"Failed to extract audio: {e}")
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg audio extraction timeout", timeout=_FFMPEG_TIMEOUT)
        raise IngestionError(f"ffmpeg audio extraction timed out after {_FFMPEG_TIMEOUT}s")
    except FileNotFoundError:
        logger.error("ffmpeg not found")
        raise IngestionError("ffmpeg is not installed. Please install it first.")


def denoise_audio(input_path: Path, output_path: Path, config: PreprocessingConfig) -> None:
    """
    Apply noise reduction to audio file.

    Args:
        input_path: Path to input WAV file
        output_path: Path to output denoised WAV file
        config: Preprocessing configuration with denoise settings
    """
    if not config.denoise:
        logger.debug("Denoising disabled, skipping")
        return

    logger.info(
        "Denoising audio",
        input=str(input_path),
        output=str(output_path),
        method=config.denoise_method,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.denoise_method == "afftdn":
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-af",
            "afftdn=nf=-20:tn=1",  # noise floor -20dB, tracking 1
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(config.audio_sample_rate),
            "-ac",
            str(config.audio_channels),
            "-y",
            str(output_path),
        ]
    elif config.denoise_method == "rnnoise":
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-af",
            "arnndn=m=rnnoise-default",  # requires rnnoise model
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(config.audio_sample_rate),
            "-ac",
            str(config.audio_channels),
            "-y",
            str(output_path),
        ]
    else:
        raise IngestionError(f"Unsupported denoise method: {config.denoise_method}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT)
        logger.info("Audio denoising complete", output=str(output_path))
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg denoising timed out", timeout=600)
        raise IngestionError("Audio denoising timed out (10 minutes)")
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg denoising failed", error=str(e), stderr=e.stderr)
        raise IngestionError(f"Failed to denoise audio: {e}")
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg denoising timeout", timeout=_FFMPEG_TIMEOUT)
        raise IngestionError(f"ffmpeg denoising timed out after {_FFMPEG_TIMEOUT}s")
    except FileNotFoundError:
        logger.error("ffmpeg not found")
        raise IngestionError("ffmpeg is not installed. Please install it first.")


def save_metadata(meta: Dict[str, Any], output_path: Path) -> None:
    """Save metadata to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Metadata saved", path=str(output_path))


def ingest(input_source: str, config: Config, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """
    Main ingestion function.

    Args:
        input_source: Path to local file or YouTube URL
        config: Pipeline configuration
        dry_run: If True, validate without executing

    Returns:
        Metadata dict on success, None on failure
    """
    logger.info("Starting ingestion", source=input_source, dry_run=dry_run)

    artifacts_dir = config.work_dir / "artifacts"
    video_dir = artifacts_dir / "video"
    audio_dir = artifacts_dir / "audio"

    prep_video_path = video_dir / "prep.mp4"
    raw_audio_path = audio_dir / "raw.wav"
    denoised_audio_path = audio_dir / "denoised.wav"
    meta_path = artifacts_dir / "meta.json"

    if dry_run:
        logger.info("Dry run - skipping actual processing")
        return {"source": input_source, "status": "dry_run_passed"}

    try:
        # Determine if input is URL or local file
        is_url = input_source.startswith(("http://", "https://"))

        temp_dir = artifacts_dir / "temp_download"
        temp_input = None

        if is_url:
            # Download from YouTube into video_dir (outside temp_dir) so that
            # temp_dir cleanup below cannot delete the downloaded source.
            temp_input = video_dir / "source.mp4"
            source_path = download_youtube_video(input_source, temp_input, temp_dir)
        else:
            # Local file
            source_path = Path(input_source)
            if not source_path.exists():
                raise IngestionError(f"Input file not found: {input_source}")

            # Copy to temp location if needed (to avoid modifying original)
            # For now, we'll work directly with the source

        # Compute source hash
        source_hash = compute_file_hash(source_path)
        logger.info("Source hash computed", hash=source_hash[:16] + "...")

        # Get metadata from source
        source_meta = get_video_metadata(source_path)
        logger.info("Source metadata extracted", **source_meta)

        # Preprocess video (remux when the source stream is already compatible)
        preprocess_video(
            source_path, prep_video_path, config.preprocessing, source_meta=source_meta
        )

        # Extract audio
        extract_audio(prep_video_path, raw_audio_path, config.preprocessing)

        # Apply denoising if enabled
        denoised = False
        if config.preprocessing.denoise:
            denoise_audio(raw_audio_path, denoised_audio_path, config.preprocessing)
            denoised = True

        # Get metadata from preprocessed video
        prep_meta = get_video_metadata(prep_video_path)

        # Compile final metadata
        meta = {
            "source": input_source,
            "source_hash": f"sha256:{source_hash}",
            "duration_sec": prep_meta["duration_sec"],
            "resolution": prep_meta["resolution"],
            "fps": prep_meta["fps"],
            "denoised": denoised,
            "denoise_method": config.preprocessing.denoise_method if denoised else None,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Save metadata
        save_metadata(meta, meta_path)

        # Cleanup temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("Ingestion complete", **meta)
        return meta

    except IngestionError as e:
        logger.error("Ingestion failed", error=str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error during ingestion", error=str(e))
        raise IngestionError(f"Unexpected error: {e}")
