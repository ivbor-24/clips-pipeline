"""
TASK-03: Transcription & Caching Module

Integrates faster-whisper for transcription with chunking, caching,
and word-level timestamps.

Inputs:
- artifacts/audio/raw.wav
- artifacts/meta.json

Outputs:
- artifacts/transcript.json (list of segments with words and timestamps)
"""

import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.config import Config, TranscriptionConfig
from src.gpu_utils import GPUConfig, get_auto_gpu_config
from src.model_registry import find_pinned_snapshot
from src.notices import add_notice, clear_notices

logger = structlog.get_logger("transcription")

# Timeouts for external commands (seconds)
_FFPROBE_TIMEOUT = 60
_FFMPEG_TIMEOUT = 300
_WHISPER_CPP_TIMEOUT = 3600
_WHISPER_CPP_DETECT_SECONDS = 30


class TranscriptionError(Exception):
    """Custom exception for transcription errors."""

    pass


def compute_audio_hash(audio_path: Path) -> str:
    """Compute SHA256 hash of audio file."""
    sha256_hash = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_model_config_hash(config: TranscriptionConfig, engine: Optional[str] = None) -> str:
    """Compute hash of model configuration for cache key.

    Args:
        config: Transcription configuration.
        engine: Engine that actually runs (differs from ``config.engine`` when
            whisper.cpp is unavailable and faster-whisper is used instead).

    Returns:
        16-char hex digest.
    """
    engine = engine or config.engine
    config_str = (
        f"{config.model}:{config.language}:{config.batch_size}:{config.compute_type}:{engine}"
    )
    if engine == "whisper_cpp":
        config_str += (
            f":{config.whisper_cpp_model_path}:{config.whisper_cpp_beam_size}"
            f":{config.whisper_cpp_max_context}"
        )
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_cache_key(audio_hash: str, config_hash: str) -> str:
    """Generate cache key from audio and config hashes."""
    return f"{audio_hash}_{config_hash}"


def load_transcript_from_cache(cache_dir: Path, cache_key: str) -> Optional[List[Dict[str, Any]]]:
    """Load transcript from cache if exists."""
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        logger.info("Loading transcript from cache", cache_file=str(cache_file))
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def save_transcript_to_cache(
    cache_dir: Path, cache_key: str, transcript: List[Dict[str, Any]]
) -> None:
    """Save transcript to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump(transcript, f, indent=2)
    logger.info("Transcript saved to cache", cache_file=str(cache_file))


def chunk_audio_by_duration(audio_path: Path, chunk_duration_min: int = 30) -> List[Path]:
    """
    Split audio into chunks of specified duration to avoid OOM.

    Returns list of temporary chunk file paths.
    """
    # Get audio duration
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_FFPROBE_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timeout", timeout=_FFPROBE_TIMEOUT)
        raise TranscriptionError(f"ffprobe timed out after {_FFPROBE_TIMEOUT}s")
    data = json.loads(result.stdout)
    total_duration = float(data.get("format", {}).get("duration", 0))

    logger.info(
        "Audio duration analyzed",
        total_seconds=total_duration,
        chunk_duration_min=chunk_duration_min,
    )

    chunk_duration_sec = chunk_duration_min * 60
    chunks = []
    temp_dir = audio_path.parent / "temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if total_duration <= chunk_duration_sec:
        # No chunking needed
        return [audio_path]

    # Create chunks
    num_chunks = int(total_duration // chunk_duration_sec) + 1
    for i in range(num_chunks):
        start_time = i * chunk_duration_sec
        chunk_path = temp_dir / f"chunk_{i:03d}.wav"

        cmd = [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-ss",
            str(start_time),
            "-t",
            str(chunk_duration_sec),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(chunk_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=_FFMPEG_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg chunk timeout", chunk=i, timeout=_FFMPEG_TIMEOUT)
            raise TranscriptionError(f"ffmpeg chunking timed out after {_FFMPEG_TIMEOUT}s")
        chunks.append(chunk_path)

    logger.info("Audio chunked", num_chunks=len(chunks), chunk_duration_sec=chunk_duration_sec)
    return chunks


def cleanup_audio_chunks(chunks: List[Path], original_audio: Path) -> None:
    """Remove temporary chunk files."""
    for chunk in chunks:
        if chunk != original_audio and chunk.exists():
            try:
                chunk.unlink()
                logger.debug("Cleaned up chunk", path=str(chunk))
            except Exception as e:
                logger.warning("Failed to clean up chunk", path=str(chunk), error=str(e))

    # Remove temp directory if empty
    temp_dir = original_audio.parent / "temp_chunks"
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass  # Directory not empty


def _get_chunk_duration(chunk_path: Path) -> float:
    """Get duration of an audio chunk using ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(chunk_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_FFPROBE_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        logger.error("ffprobe chunk timeout", timeout=_FFPROBE_TIMEOUT)
        raise TranscriptionError(f"ffprobe timed out after {_FFPROBE_TIMEOUT}s")
    chunk_data = json.loads(result.stdout)
    return float(chunk_data.get("format", {}).get("duration", 0))


def transcribe_chunk(
    model: Any, audio_path: Path, batch_size: int, language: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Transcribe a single audio chunk using faster-whisper.

    Returns list of segments with word-level timestamps.
    """
    segments_info = []

    # Use word_timestamps=True for word-level timing.
    # Note: faster-whisper >= 1.2 removed batch_size from transcribe().
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=language if language != "auto" else None,
        vad_filter=True,  # Enable VAD to skip silent parts
    )

    logger.info(
        "Chunk transcription complete",
        detected_language=info.language,
        language_probability=info.language_probability,
    )

    for segment in segments:
        segment_dict = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "words": [],
        }

        # Extract word-level timestamps
        if hasattr(segment, "words") and segment.words:
            for word in segment.words:
                segment_dict["words"].append(
                    {
                        "word": word.word.strip(),
                        "start": round(word.start, 2),
                        "end": round(word.end, 2),
                    }
                )

        segments_info.append(segment_dict)

    return segments_info


def resolve_whisper_cpp_binary(binary: str) -> Optional[str]:
    """Resolve whisper-cli binary path: absolute/relative path or name via PATH."""
    if not binary:
        return None
    candidate = Path(binary)
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(binary)


def _validate_whisper_cpp_setup(config: TranscriptionConfig) -> None:
    """Ensure whisper.cpp binary and model are available.

    There is no fallback to faster-whisper: on Intel and AMD GPUs it runs on
    the CPU, hours instead of minutes, so a missing piece is an error that
    says how to fix it.
    """
    if not resolve_whisper_cpp_binary(config.whisper_cpp_binary):
        raise TranscriptionError(
            "transcription.engine is whisper_cpp, but whisper-cli was not found at "
            f"{config.whisper_cpp_binary}. Build it with "
            "`just install-whisper-cpp <cpu|cuda|openvino>` or set "
            "transcription.whisper_cpp_binary. To use faster-whisper instead, set "
            "transcription.engine: faster_whisper (on Intel and AMD GPUs it runs on the CPU)."
        )
    model_file = Path(config.whisper_cpp_model_path)
    if not model_file.is_file():
        raise TranscriptionError(
            "transcription.engine is whisper_cpp, but its model was not found at "
            f"{config.whisper_cpp_model_path}. Download it with "
            "`just prefetch-models config/config.yaml <profile>` (the profile the job uses) "
            "or set transcription.whisper_cpp_model_path."
        )


def _detect_language_whisper_cpp(
    binary: str, model_path: str, audio_path: Path, threads: int
) -> Optional[str]:
    """Detect spoken language on a short audio slice via whisper.cpp.

    whisper.cpp auto-detection misfires on long recordings (returns a wrong
    language and transcribes garbage), so detection runs on the first
    _WHISPER_CPP_DETECT_SECONDS and the result is reused for the full chunk.
    Returns the detected language code or None if detection failed.
    """
    slice_path = audio_path.with_suffix(".detect.wav")
    try:
        slice_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "quiet",
            "-i",
            str(audio_path),
            "-t",
            str(_WHISPER_CPP_DETECT_SECONDS),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(slice_path),
        ]
        try:
            subprocess.run(slice_cmd, capture_output=True, check=True, timeout=_FFMPEG_TIMEOUT)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            logger.warning("whisper_cpp_detect_slice_failed", audio=str(audio_path))
            return None

        if not slice_path.is_file():
            return None

        stem = str(slice_path.with_suffix(""))
        cmd = [
            binary,
            "-m",
            model_path,
            "-f",
            str(slice_path),
            "-l",
            "auto",
            "-oj",
            "-of",
            stem,
            "-t",
            str(threads),
            "-np",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=_WHISPER_CPP_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning("whisper_cpp_detect_timeout")
            return None

        if result.returncode != 0:
            logger.warning("whisper_cpp_detect_failed", returncode=result.returncode)
            return None

        json_file = Path(stem + ".json")
        if not json_file.is_file():
            return None

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        finally:
            try:
                json_file.unlink()
            except OSError:
                pass

        detected = (data.get("result") or {}).get("language")
        if detected and detected != "unknown":
            logger.info("whisper_cpp_language_detected", language=detected)
            return detected
        return None
    finally:
        try:
            slice_path.unlink()
        except OSError:
            pass


# whisper-cli names the backend it picked on stderr, e.g.
#   whisper_backend_init_gpu: using Vulkan0 backend
#   whisper_backend_init_gpu: no GPU found
_WHISPER_BACKEND_RE = re.compile(
    r"whisper_backend_init_gpu: (?:using (\S+) backend|(no GPU found))"
)


def whisper_cpp_device(stderr: str) -> Optional[str]:
    """The device whisper-cli ran on (e.g. "Vulkan0", "CPU"), None if it did not say."""
    match = _WHISPER_BACKEND_RE.search(stderr or "")
    if match is None:
        return None
    return "CPU" if match.group(2) else match.group(1)


def transcribe_chunk_whisper_cpp(
    binary: str,
    model_path: str,
    audio_path: Path,
    language: str = "auto",
    threads: int = 8,
    beam_size: int = 4,
    max_context: int = 0,
    work_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Transcribe a single audio chunk using whisper.cpp (whisper-cli subprocess).

    Args:
        binary: whisper-cli path or name resolved via PATH.
        model_path: ggml model file.
        audio_path: 16 kHz mono WAV chunk.
        language: Language code or "auto" (detected on a short slice first).
        threads: CPU threads (-t).
        beam_size: Beam size (-bs).
        max_context: Text-context tokens carried between windows (-mc);
            0 disables carry-over, -1 uses the model maximum.
        work_dir: Job directory: a run on the CPU becomes a notice there.

    Returns:
        Segments in the same format as the faster-whisper path:
        [{"start": sec, "end": sec, "text": str, "words": []}, ...]
    """
    binary_path = resolve_whisper_cpp_binary(binary)
    if not binary_path:
        raise TranscriptionError(f"whisper.cpp binary not found: {binary}")

    model_file = Path(model_path)
    if not model_file.is_file():
        raise TranscriptionError(f"whisper.cpp model not found: {model_path}")

    output_stem = str(audio_path.with_suffix(""))
    json_file = Path(output_stem + ".json")

    effective_language = language
    if language == "auto":
        detected = _detect_language_whisper_cpp(binary_path, model_path, audio_path, threads)
        if detected:
            effective_language = detected

    cmd = [
        binary_path,
        "-m",
        str(model_file),
        "-f",
        str(audio_path),
        "-oj",
        "-of",
        output_stem,
        "-t",
        str(threads),
        "-bs",
        str(beam_size),
        "-mc",
        str(max_context),
        # No -np: it also silences the line that names the GPU backend.
    ]
    if effective_language and effective_language != "auto":
        cmd += ["-l", effective_language]

    logger.info(
        "whisper_cpp_chunk_started",
        binary=binary_path,
        model=model_path,
        audio=str(audio_path),
        threads=threads,
        beam_size=beam_size,
        max_context=max_context,
    )

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=_WHISPER_CPP_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        logger.error("whisper_cpp_timeout", timeout=_WHISPER_CPP_TIMEOUT)
        raise TranscriptionError(f"whisper.cpp timed out after {_WHISPER_CPP_TIMEOUT}s")

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-400:]
        logger.error(
            "whisper_cpp_failed",
            returncode=result.returncode,
            stderr=stderr_tail,
        )
        raise TranscriptionError(f"whisper.cpp failed with exit code {result.returncode}")

    if not json_file.is_file():
        raise TranscriptionError(f"whisper.cpp did not produce JSON output: {json_file}")

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise TranscriptionError(f"Failed to parse whisper.cpp output: {e}")
    finally:
        try:
            json_file.unlink()
        except OSError:
            pass

    segments_info = []
    for seg in data.get("transcription", []):
        offsets = seg.get("offsets", {})
        start_ms = offsets.get("from", 0)
        end_ms = offsets.get("to", start_ms)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments_info.append(
            {
                "start": round(start_ms / 1000.0, 2),
                "end": round(end_ms / 1000.0, 2),
                "text": text,
                "words": [],
            }
        )

    detected_lang = (data.get("result") or {}).get("language")
    device = whisper_cpp_device(result.stderr)
    logger.info(
        "whisper_cpp_chunk_complete",
        segments=len(segments_info),
        detected_language=detected_lang,
        device=device,
    )
    if device and device.upper().startswith("CPU"):
        logger.warning("whisper_cpp_on_cpu", hint="GPU driver or whisper-cli build")
        if work_dir is not None:
            add_notice(
                work_dir,
                "transcription",
                "whisper_cpp_on_cpu",
                "whisper.cpp found no GPU and transcribed on the CPU, several times slower. "
                "Check the GPU self-check (Docker: ./setup.sh; native: just check-gpu).",
            )

    return segments_info


def validate_transcript_duration(
    transcript: List[Dict[str, Any]], expected_duration: float, tolerance_percent: float = 2.0
) -> bool:
    """Check that the transcript reaches the end of the video (not cut short).

    Same rule as the resume check in src/artifact_validator.py, so a transcript
    that passes here is also accepted when a job resumes.

    Returns True if validation passes.
    """
    from src.artifact_validator import transcript_reaches_end

    if not transcript:
        logger.warning("Empty transcript cannot be validated")
        return False

    is_valid = transcript_reaches_end(transcript, expected_duration, tolerance_percent)
    logger.info(
        "Duration validation",
        expected_duration=expected_duration,
        last_segment_end=round(max(float(seg["end"]) for seg in transcript), 2),
        is_valid=is_valid,
    )
    return is_valid


def load_whisper_model(config: TranscriptionConfig, gpu_config: GPUConfig):
    """
    Load faster-whisper model with specified configuration.

    Implements memory-safe loading for 12GB VRAM constraint.
    """
    try:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model",
            model=config.model,
            compute_type=config.compute_type,
            batch_size=config.batch_size,
        )

        # Determine device and compute type from GPU auto-configuration
        device = gpu_config.whisper_device
        compute_type = gpu_config.whisper_compute_type

        # Validate: CPU cannot use float16
        if device == "cpu" and compute_type == "float16":
            logger.warning(
                "whisper_cpu_float16_downgrade",
                reason="CPU does not support float16, downgrading to int8",
            )
            compute_type = "int8"

        logger.info(
            "whisper_gpu_config",
            device=device,
            compute_type=compute_type,
            backend=gpu_config.backend,
        )

        # Known model names download from the revision pinned in
        # config/models.lock.yaml; local paths and other names load as given.
        pin = find_pinned_snapshot(config.model)
        model = WhisperModel(
            config.model,
            device=device,
            compute_type=compute_type,
            revision=pin.revision if pin else None,
        )

        logger.info("Whisper model loaded successfully")
        return model

    except ImportError:
        logger.error("faster-whisper not installed")
        raise TranscriptionError(
            "faster-whisper is not installed. Install with: pip install faster-whisper"
        )
    except RuntimeError as e:
        if "CUDA" in str(e) or "OOM" in str(e):
            logger.error("CUDA out of memory during model load", error=str(e))
            raise TranscriptionError(f"CUDA OOM: {e}")
        raise


# Expose WhisperModel for mocking in tests
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


def _is_oom_error(error: Exception) -> bool:
    """Return True if a runtime error looks like an out-of-memory failure."""
    message = str(error)
    return "OOM" in message or "out of memory" in message.lower()


def unload_whisper_model(model: Any, gpu_config: GPUConfig) -> None:
    """Explicitly unload Whisper model and clear CUDA cache."""
    logger.info("Unloading Whisper model and clearing CUDA cache")

    del model
    gpu_config.cleanup_fn()


def transcription_gpu_fields(gpu_config: GPUConfig, engine: str) -> Dict[str, Any]:
    """Fields for the ``transcription_gpu`` log line.

    whisper.cpp transcribes on its own backend (Vulkan on Intel), and the real
    device is logged per chunk (``whisper_cpp_chunk_complete``); logging the
    faster-whisper ``whisper_device`` (which is "cpu" there) would mislead.
    """
    fields = {"engine": engine, "backend": gpu_config.backend}
    if engine != "whisper_cpp":
        fields["whisper_device"] = gpu_config.whisper_device
    return fields


def transcribe(config: Config, dry_run: bool = False) -> Optional[List[Dict[str, Any]]]:
    """
    Main transcription function.

    Args:
        config: Pipeline configuration
        dry_run: If True, validate without executing

    Returns:
        Transcript list on success, None on failure

    Raises:
        TranscriptionError: On transcription failure
    """
    logger.info("Starting transcription", dry_run=dry_run)
    if not dry_run:
        clear_notices(config.work_dir, "transcription")

    gpu_config = get_auto_gpu_config(config.gpu)
    engine = config.transcription.engine
    logger.info("transcription_gpu", **transcription_gpu_fields(gpu_config, engine))

    artifacts_dir = config.work_dir / "artifacts"
    raw_audio_path = artifacts_dir / "audio" / "raw.wav"
    denoised_audio_path = artifacts_dir / "audio" / "denoised.wav"
    meta_path = artifacts_dir / "meta.json"
    transcript_path = artifacts_dir / "transcript.json"
    cache_dir = artifacts_dir / "cache" / "transcription"

    audio_path = denoised_audio_path if denoised_audio_path.exists() else raw_audio_path

    if denoised_audio_path.exists():
        logger.info("Using denoised audio for transcription", path=str(denoised_audio_path))
    else:
        logger.info("Using raw audio for transcription", path=str(raw_audio_path))

    # Validate inputs exist
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    if not meta_path.exists():
        raise TranscriptionError(f"Metadata file not found: {meta_path}")

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    expected_duration = meta.get("duration_sec", 0)
    logger.info("Loaded metadata", duration_sec=expected_duration)

    if dry_run:
        logger.info("Dry run - skipping actual transcription")
        return [{"status": "dry_run_passed"}]

    use_whisper_cpp = engine == "whisper_cpp"
    if use_whisper_cpp:
        _validate_whisper_cpp_setup(config.transcription)

    audio_hash = compute_audio_hash(audio_path)
    config_hash = compute_model_config_hash(config.transcription, engine)
    cache_key = get_cache_key(audio_hash, config_hash)

    logger.info(
        "Cache key computed",
        audio_hash=audio_hash[:16] + "...",
        config_hash=config_hash,
        engine=engine,
    )

    # Check cache
    cached_transcript = load_transcript_from_cache(cache_dir, cache_key)
    if cached_transcript is not None:
        logger.info("Using cached transcript")
        # Save to standard output location
        with open(transcript_path, "w") as f:
            json.dump(cached_transcript, f, indent=2)
        return cached_transcript

    # Load model
    model = None
    model_gpu_config = gpu_config
    batch_size = config.transcription.batch_size
    chunks: List[Path] = []

    try:
        logger.info("transcription_engine", engine=engine)
        if not use_whisper_cpp:
            model = load_whisper_model(config.transcription, model_gpu_config)

        # Chunk audio if needed
        chunks = chunk_audio_by_duration(audio_path, config.transcription.chunk_duration_min)

        all_segments = []
        global_time_offset = 0.0

        for i, chunk_path in enumerate(chunks):
            logger.info(
                "Transcribing chunk",
                chunk_index=i + 1,
                total_chunks=len(chunks),
            )

            if use_whisper_cpp:
                chunk_segments = transcribe_chunk_whisper_cpp(
                    config.transcription.whisper_cpp_binary,
                    config.transcription.whisper_cpp_model_path,
                    chunk_path,
                    language=config.transcription.language,
                    threads=config.transcription.whisper_cpp_threads,
                    beam_size=config.transcription.whisper_cpp_beam_size,
                    max_context=config.transcription.whisper_cpp_max_context,
                    work_dir=config.work_dir,
                )
            else:
                try:
                    chunk_segments = transcribe_chunk(
                        model,
                        chunk_path,
                        batch_size=batch_size,
                        language=config.transcription.language,
                    )
                except RuntimeError as e:
                    if not _is_oom_error(e):
                        raise
                    if model_gpu_config.whisper_compute_type == "int8":
                        logger.error("whisper_oom_at_int8", error=str(e))
                        raise TranscriptionError(f"OOM even with int8 compute type: {e}") from e
                    # Reload the model in int8 (about half the memory of
                    # float16) and retry this chunk; later chunks keep int8.
                    logger.warning(
                        "whisper_oom_fallback_int8",
                        chunk_index=i + 1,
                        old_compute_type=model_gpu_config.whisper_compute_type,
                    )
                    unload_whisper_model(model, model_gpu_config)
                    model = None
                    model_gpu_config = dataclasses.replace(
                        model_gpu_config, whisper_compute_type="int8"
                    )
                    model = load_whisper_model(config.transcription, model_gpu_config)
                    chunk_segments = transcribe_chunk(
                        model,
                        chunk_path,
                        batch_size=batch_size,
                        language=config.transcription.language,
                    )

            # Adjust timestamps for global offset
            for seg in chunk_segments:
                seg["start"] += global_time_offset
                seg["end"] += global_time_offset
                for word in seg["words"]:
                    word["start"] += global_time_offset
                    word["end"] += global_time_offset

            all_segments.extend(chunk_segments)

            # Update offset for next chunk
            global_time_offset += _get_chunk_duration(chunk_path)

            logger.info(
                "Chunk processed",
                chunk_index=i + 1,
                segments_in_chunk=len(chunk_segments),
            )

        # Add segment IDs
        transcript = []
        for i, seg in enumerate(all_segments):
            seg_with_id = {"id": f"seg_{i+1:04d}", **seg, "language": config.transcription.language}
            transcript.append(seg_with_id)

        # Validate transcript duration
        if not validate_transcript_duration(transcript, expected_duration):
            logger.warning("Transcript duration validation failed", expected=expected_duration)
            # Don't fail, just warn - could be legitimate silence gaps

        # Save to cache
        save_transcript_to_cache(cache_dir, cache_key, transcript)

        # Save to standard output location
        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2)

        logger.info(
            "Transcription complete",
            total_segments=len(transcript),
            output_path=str(transcript_path),
        )

        return transcript

    finally:
        # Always unload model
        if model is not None:
            unload_whisper_model(model, model_gpu_config)

        # Cleanup temporary chunks
        if len(chunks) > 1:
            cleanup_audio_chunks(chunks, audio_path)
