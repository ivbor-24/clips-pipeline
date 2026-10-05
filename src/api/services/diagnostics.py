"""What the Diagnostics page shows: worker, GPU, models, ffmpeg, disk.

The GPU self-check comes from the job worker (worker_status): only it sees the
GPU. Everything else the API checks itself: it runs from the same image, with
the same models, data and job directories mounted.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src import __version__
from src.api.models.worker import WorkerStatus

logger = structlog.get_logger("api.diagnostics")

# The worker writes its heartbeat every 10 s (src/worker.py), also during jobs.
WORKER_STALE_SEC = 45
MODELS_DIR = Path("artifacts/cache/models")


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:  # SQLite returns naive datetimes; they are UTC
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def worker_state(status: Optional[WorkerStatus], running_job: Optional[Dict[str, Any]]) -> Dict:
    """Whether the worker runs, from its heartbeat; the job it runs, if any."""
    if status is None or status.heartbeat_at is None:
        return {"state": "never_started", "heartbeat_at": None, "job": None}
    heartbeat = status.heartbeat_at
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    state = "busy" if running_job else "idle"
    if age > WORKER_STALE_SEC:
        state = "not_responding"
    return {
        "state": state,
        "heartbeat_at": _iso(status.heartbeat_at),
        "started_at": _iso(status.started_at),
        "version": status.version,
        "job": running_job,
    }


def gpu_check(status: Optional[WorkerStatus]) -> Optional[Dict[str, Any]]:
    """The worker's last GPU self-check and whether a new one is waiting."""
    if status is None:
        return None
    result: Dict[str, Any] = {}
    if status.gpu_check:
        try:
            result = json.loads(status.gpu_check)
        except json.JSONDecodeError:
            result = {"ok": False, "error": status.gpu_check[:500], "results": []}
    requested, checked = status.gpu_check_requested_at, status.gpu_check_at
    result["checked_at"] = _iso(checked)
    result["requested_at"] = _iso(requested)
    result["pending"] = requested is not None and (checked is None or requested > checked)
    return result


def _file_model(name: str, purpose: str, path: Path, pin: Any) -> Dict[str, Any]:
    from src.model_registry import file_state

    state = file_state(path, pin)
    size = Path(path).resolve().stat().st_size if state != "missing" else None
    return {
        "name": name,
        "purpose": purpose,
        "path": str(path),
        "state": state,
        "size": size,
        "expected_size": pin.size if pin is not None else None,
    }


def _snapshot_model(name: str, purpose: str, repo: str, revision: Optional[str]) -> Dict:
    """A Hugging Face snapshot (faster-whisper, KeyBERT): present in the cache or not."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo, revision=revision, local_files_only=True)
        state = "present"
    except Exception:  # LocalEntryNotFoundError and friends: not downloaded
        path, state = None, "missing"
    return {"name": name, "purpose": purpose, "path": path, "state": state, "repo": repo}


def models(config_path: str = "config/config.yaml") -> List[Dict[str, Any]]:
    """The model files the configured pipeline needs, and how far each is checked."""
    from src.face_cropping import FACE_DETECTOR_FILE, FACE_DETECTOR_PATH, FACE_DETECTOR_REPO
    from src.model_registry import find_pinned_file, find_pinned_file_by_name, find_pinned_snapshot
    from src.user_settings import load_app_config

    config = load_app_config(config_path)
    entries = []
    llm = config.scoring.llm
    if llm.enabled and llm.provider == "llama_cpp" and llm.model_path:
        pin = find_pinned_file(llm.model_repo, llm.model_file) if llm.model_repo else None
        entries.append(_file_model(Path(llm.model_path).name, "LLM", Path(llm.model_path), pin))
    elif llm.enabled:
        entries.append({"name": llm.model, "purpose": "LLM", "state": f"api:{llm.provider}"})

    transcription = config.transcription
    if transcription.engine == "whisper_cpp":
        path = Path(
            transcription.whisper_cpp_model_path or MODELS_DIR / f"ggml-{transcription.model}.bin"
        )
        pin = find_pinned_file_by_name(path.name)
        entries.append(_file_model(path.name, "speech recognition (whisper.cpp)", path, pin))
    else:
        pin = find_pinned_snapshot(transcription.model)
        repo = pin.repo if pin else f"Systran/faster-whisper-{transcription.model}"
        entries.append(
            _snapshot_model(
                transcription.model,
                "speech recognition (faster-whisper)",
                repo,
                pin.revision if pin else None,
            )
        )

    entries.append(
        _file_model(
            FACE_DETECTOR_FILE,
            "face detection",
            FACE_DETECTOR_PATH,
            find_pinned_file(FACE_DETECTOR_REPO, FACE_DETECTOR_FILE),
        )
    )
    if config.scoring.term_extraction_method == "keybert":
        from src.model_registry import KEYBERT_MODEL

        entries.append(_snapshot_model(KEYBERT_MODEL, "keywords (KeyBERT)", KEYBERT_MODEL, None))
    return entries


def ffmpeg() -> Dict[str, Any]:
    """ffmpeg's version line, or why it does not run."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "version": None, "error": str(e)}
    first = (proc.stdout or "").splitlines()[:1]
    return {"ok": proc.returncode == 0, "version": first[0] if first else None}


def disks() -> List[Dict[str, Any]]:
    """Free space where the data lives (several may share one file system)."""
    places = [
        ("jobs", Path("jobs")),
        ("uploads", Path("uploads")),
        ("database", Path("data")),
        ("models", MODELS_DIR),
    ]
    result = []
    for name, path in places:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            result.append({"name": name, "path": str(path), "free": None, "total": None})
            continue
        result.append({"name": name, "path": str(path), "free": usage.free, "total": usage.total})
    return result


def system() -> Dict[str, Any]:
    return {
        "version": __version__,
        # The Docker image names its backend; a native install leaves it empty.
        "backend": os.environ.get("PIPELINE_HARDWARE") or None,
        "docker": Path("/.dockerenv").exists(),
    }
