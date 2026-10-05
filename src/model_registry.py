"""
Pinned model weights (config/models.lock.yaml): revisions and checksums.

Inputs:
- config/models.lock.yaml

Outputs:
- downloaded model files; a `<file>.sha256` sidecar next to each verified file
"""

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
import yaml

logger = structlog.get_logger("model_registry")

MODELS_LOCK = Path(__file__).resolve().parent.parent / "config" / "models.lock.yaml"
# The embedding model of KeyBERT (scoring.term_extraction_method: keybert).
KEYBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHUNK = 8 * 1024 * 1024


class ModelIntegrityError(Exception):
    """A model file does not match its pinned size or sha256."""


@dataclass(frozen=True)
class PinnedFile:
    """One pinned single-file model (GGUF / ggml)."""

    repo: str
    file: str
    revision: str
    sha256: str
    size: int


@dataclass(frozen=True)
class PinnedSnapshot:
    """A pinned multi-file model repo (faster-whisper)."""

    repo: str
    revision: str


@lru_cache(maxsize=4)
def _load_lock(path: Path = MODELS_LOCK) -> Dict[str, Any]:
    if not path.exists():
        logger.warning("models_lock_missing", path=str(path))
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_pinned_file(
    repo: str, filename: str, lock_path: Path = MODELS_LOCK
) -> Optional[PinnedFile]:
    """Return the pin for repo/filename, or None if the model is not listed."""
    for entry in _load_lock(lock_path).get("files") or []:
        if entry.get("repo") == repo and entry.get("file") == filename:
            return PinnedFile(
                repo=entry["repo"],
                file=entry["file"],
                revision=entry["revision"],
                sha256=entry["sha256"].lower(),
                size=int(entry["size"]),
            )
    return None


def find_pinned_file_by_name(filename: str, lock_path: Path = MODELS_LOCK) -> Optional[PinnedFile]:
    """Return the pin for a file name in any repo (e.g. a ggml model path)."""
    for entry in _load_lock(lock_path).get("files") or []:
        if entry.get("file") == filename:
            return find_pinned_file(entry["repo"], filename, lock_path)
    return None


def find_pinned_snapshot(name: str, lock_path: Path = MODELS_LOCK) -> Optional[PinnedSnapshot]:
    """Return the pin for a faster-whisper model name (e.g. "large-v3-turbo")."""
    entry = (_load_lock(lock_path).get("faster_whisper") or {}).get(name)
    if not entry:
        return None
    return PinnedSnapshot(repo=entry["repo"], revision=entry["revision"])


def sha256_file(path: Path) -> str:
    """Stream a file through sha256."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _stat_key(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def verify_file(path: Path, pin: PinnedFile) -> None:
    """Check size and sha256 of a model file against its pin.

    The full hash is computed once; a `<file>.sha256` sidecar records it
    together with the file's size and mtime, so later calls only stat the
    file. Symlinks are resolved, so the sidecar sits next to the real file.

    Raises:
        ModelIntegrityError: On a size or hash mismatch.
    """
    real = Path(path).resolve()
    size = real.stat().st_size
    if size != pin.size:
        raise ModelIntegrityError(
            f"{real}: size {size} != pinned {pin.size} for {pin.repo}/{pin.file} "
            "(incomplete download or a different model version)"
        )

    sidecar = _sidecar(real)
    expected_record = f"{pin.sha256} {_stat_key(real)}"
    try:
        if sidecar.read_text().strip() == expected_record:
            logger.debug("model_checksum_cached", path=str(real))
            return
    except OSError:
        pass

    logger.info("model_checksum_verifying", path=str(real), size=size)
    actual = sha256_file(real)
    if actual != pin.sha256:
        raise ModelIntegrityError(
            f"{real}: sha256 {actual} != pinned {pin.sha256} for {pin.repo}/{pin.file}"
        )
    try:
        sidecar.write_text(expected_record + "\n")
    except OSError as e:
        logger.warning("model_checksum_sidecar_not_written", path=str(sidecar), error=str(e))
    logger.info("model_checksum_ok", path=str(real))


def file_state(path: Path, pin: Optional[PinnedFile]) -> str:
    """How far a model file is known to be good, without hashing it.

    Returns one of: ``missing``; ``unpinned`` (not in models.lock.yaml);
    ``wrong_size``; ``verified`` (sha256 checked and the file unchanged since,
    per the sidecar of verify_file); ``unverified`` (right size, sha256 not
    checked yet: the next prefetch or job checks it).
    """
    path = Path(path)
    if not path.exists():
        return "missing"
    if pin is None:
        return "unpinned"
    real = path.resolve()
    if real.stat().st_size != pin.size:
        return "wrong_size"
    try:
        if _sidecar(real).read_text().strip() == f"{pin.sha256} {_stat_key(real)}":
            return "verified"
    except OSError:
        pass
    return "unverified"


def check_size(path: Path, pin: PinnedFile) -> bool:
    """Cheap check before loading a model: log a warning on a size mismatch."""
    size = Path(path).resolve().stat().st_size
    if size != pin.size:
        logger.warning(
            "model_size_mismatch",
            path=str(path),
            size=size,
            pinned_size=pin.size,
            model=f"{pin.repo}/{pin.file}",
            hint="run `just prefetch-models` to re-download and verify",
        )
        return False
    return True


def download_pinned_file(pin: PinnedFile, token: Optional[str] = None) -> Path:
    """Download a pinned file at its revision into the HF cache and verify it."""
    from huggingface_hub import hf_hub_download

    logger.info("model_downloading", repo=pin.repo, file=pin.file, revision=pin.revision)
    path = Path(
        hf_hub_download(
            repo_id=pin.repo,
            filename=pin.file,
            revision=pin.revision,
            token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        )
    )
    verify_file(path, pin)
    return path


def link_into_place(target: Path, source: Path) -> None:
    """Point `target` at `source` (symlink, or copy if symlinks fail)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        import shutil

        shutil.copy2(source, target)
