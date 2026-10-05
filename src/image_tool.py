"""
Image build/save/load/verify/release utilities for Docker deployment.

This module provides a Python facade for working with the pipeline Docker
image end-to-end: building, exporting to a portable tarball, loading on
a target host, verifying the GPU integration inside the image, and
promoting a verified image to a stable tag.

Design notes:
- No global state. All functions take explicit parameters.
- External dependencies (docker CLI, sha256sum) are invoked via subprocess
  with explicit timeouts and structured error handling.
- A thin dataclass ``ImageRef`` encapsulates the parsing/normalization of
  image references (e.g. ``pipeline:openvino-a1b2c3d``).
- ``ReleaseManifest`` is a tiny JSON-on-disk store for deployment state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from src.logger import get_logger

logger = get_logger("image_tool")

VALID_BACKENDS = ("cpu", "cuda", "openvino", "rocm")
DEFAULT_DIST_DIR = Path("dist")
MANIFEST_PATH = Path("dist/release-manifest.json")
DOCKERFILE = Path("Dockerfile.backend")
# Accept both short (7+) and full (40) hex git SHAs, plus the literal
# "unknown" sentinel used when the working tree is not a git repo.
SHA_RE = re.compile(r"^(?:[a-f0-9]{7,40}|unknown)$")


class ImageError(Exception):
    """Raised when an image lifecycle operation fails."""


def require_docker() -> str:
    """Locate the ``docker`` binary or raise ImageError.

    Returns:
        Absolute path to the docker executable.
    """
    binary = shutil.which("docker")
    if not binary:
        raise ImageError("docker CLI not found in PATH")
    return binary


def _run(cmd: List[str], timeout: int = 1800, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess with logging and consistent error handling.

    Args:
        cmd: Command and arguments.
        timeout: Maximum runtime in seconds.
        check: Whether to raise on non-zero exit.

    Returns:
        CompletedProcess instance.
    """
    logger.info("exec", cmd=" ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
    except FileNotFoundError as e:
        raise ImageError(f"binary not found: {cmd[0]}") from e
    except subprocess.TimeoutExpired as e:
        raise ImageError(f"timeout after {timeout}s: {' '.join(cmd)}") from e
    except subprocess.CalledProcessError as e:
        logger.error("cmd_failed", rc=e.returncode, stderr=e.stderr.strip())
        raise ImageError(f"command failed (rc={e.returncode}): {' '.join(cmd)}") from e
    if result.stderr.strip():
        logger.debug("cmd_stderr", stderr=result.stderr.strip()[:2000])
    return result


def _git_short_sha(repo: Path = Path(".")) -> Optional[str]:
    """Return the current short git SHA, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def build_image_tag(backend: str, git_sha: Optional[str] = None) -> str:
    """Compose a deterministic image tag.

    Format: ``pipeline:<backend>-<sha>`` where sha is either the provided
    short SHA (validated) or the result of ``git rev-parse --short HEAD``.
    A date suffix is appended to allow multiple builds per day.

    Args:
        backend: One of VALID_BACKENDS.
        git_sha: Optional explicit short SHA.

    Returns:
        Tag string like ``pipeline:openvino-a1b2c3d-20260610``.
    """
    if backend not in VALID_BACKENDS:
        raise ImageError(f"invalid backend: {backend!r}; expected one of {VALID_BACKENDS}")
    sha = git_sha or _git_short_sha() or "unknown"
    if sha != "unknown" and not SHA_RE.match(sha):
        raise ImageError(f"invalid git sha: {sha!r}")
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"pipeline:{backend}-{sha}-{date}"


def build_image(
    backend: str,
    dockerfile: Path = DOCKERFILE,
    no_cache: bool = False,
    push_latest: bool = True,
) -> str:
    """Build the Docker image with the requested GPU backend.

    Args:
        backend: One of VALID_BACKENDS.
        dockerfile: Path to the Dockerfile.
        no_cache: Disable Docker build cache.
        push_latest: Also tag the result as ``pipeline:<backend>-latest``.

    Returns:
        The primary tag the image was built with.
    """
    binary = require_docker()
    if not dockerfile.exists():
        raise ImageError(f"dockerfile not found: {dockerfile}")
    if backend not in VALID_BACKENDS:
        raise ImageError(f"invalid backend: {backend!r}")

    tag = build_image_tag(backend)
    cmd = [
        binary,
        "build",
        "--build-arg",
        f"GPU_BACKEND={backend}",
        # The image is meant to move to another machine (image-save / load):
        # no CPU instructions beyond AVX (scripts/ggml_cpu_flags.sh).
        "--build-arg",
        "CPU_TARGET=portable",
        "-f",
        str(dockerfile),
        "-t",
        tag,
    ]
    if no_cache:
        cmd.append("--no-cache")
    if push_latest:
        cmd += ["-t", f"pipeline:{backend}-latest"]
    cmd.append(".")

    logger.info("build_start", backend=backend, tag=tag)
    _run(cmd, timeout=3600)
    logger.info("build_done", tag=tag)
    return tag


def image_size_mb(image: str) -> float:
    """Return the compressed size of an image in megabytes.

    Uses ``docker images --format '{{.Size}}'`` and parses the human-readable
    string (e.g. ``"1.05GB"``). Falls back to 0.0 on parse error.
    """
    binary = require_docker()
    result = _run(
        [binary, "images", image, "--format", "{{.Size}}"],
        timeout=60,
        check=False,
    )
    raw = (result.stdout or "").strip()
    if not raw:
        return 0.0
    m = re.match(r"^([\d.]+)\s*([KMGT]?B)$", raw, re.IGNORECASE)
    if not m:
        return 0.0
    value = float(m.group(1))
    unit = m.group(2).upper()
    multiplier = {"B": 1, "KB": 1 / 1024, "MB": 1, "GB": 1024, "TB": 1024 * 1024}.get(unit, 1)
    return round(value * multiplier, 1)


def save_image(image: str, dist_dir: Path = DEFAULT_DIST_DIR) -> Path:
    """Export a Docker image to a gzipped tarball with a sidecar sha256.

    Args:
        image: Fully qualified image:tag reference.
        dist_dir: Output directory (created if missing).

    Returns:
        Path to the produced .tar.gz file.
    """
    binary = require_docker()
    dist_dir = Path(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    safe_name = image.replace(":", "-").replace("/", "-")
    out = dist_dir / f"{safe_name}.tar.gz"
    sha_file = out.with_suffix(out.suffix + ".sha256")

    logger.info("save_start", image=image, out=str(out))
    with open(out, "wb") as fh:
        proc = subprocess.Popen(
            [binary, "save", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        with subprocess.Popen(
            ["gzip", "-c"], stdin=proc.stdout, stdout=fh, stderr=subprocess.PIPE
        ) as gz:
            proc.stdout.close()
            gz_err = gz.communicate(timeout=1800)[1]
        proc_err = proc.communicate(timeout=10)[1]
        rc = proc.wait()
    if rc != 0:
        raise ImageError(f"docker save failed (rc={rc}): {proc_err.decode(errors='replace')[:500]}")
    if gz_err:
        logger.warning("gzip_stderr", stderr=gz_err.decode(errors="replace")[:500])

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    sha_file.write_text(f"{digest}  {out.name}\n")
    logger.info(
        "save_done", out=str(out), size_mb=round(out.stat().st_size / 1024 / 1024, 1), sha256=digest
    )
    return out


def load_image(archive: Path) -> List[str]:
    """Load a previously saved image tarball into local Docker.

    Verifies the sidecar sha256 file if present.

    Args:
        archive: Path to a .tar or .tar.gz file.

    Returns:
        List of image tags loaded.
    """
    binary = require_docker()
    archive = Path(archive)
    if not archive.exists():
        raise ImageError(f"archive not found: {archive}")

    sha_file = archive.with_suffix(archive.suffix + ".sha256")
    if sha_file.exists():
        expected = sha_file.read_text().split()[0]
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if expected != actual:
            raise ImageError(f"sha256 mismatch: expected {expected[:12]}..., got {actual[:12]}...")
        logger.info("sha256_ok", sha256=actual[:12])

    logger.info("load_start", archive=str(archive))
    cmd = [binary, "load"]
    if archive.suffix == ".gz":
        proc = subprocess.Popen(
            ["gunzip", "-c", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            assert proc.stdout is not None
            result = subprocess.run(
                cmd, stdin=proc.stdout, capture_output=True, text=True, timeout=1800
            )
        finally:
            proc.stdout.close()
            proc.wait(timeout=10)
    else:
        with open(archive, "rb") as fh:
            result = subprocess.run(cmd, stdin=fh, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise ImageError(f"docker load failed: {result.stderr.strip()[:500]}")
    tags = [
        line.split(": ", 1)[-1].strip()
        for line in result.stdout.splitlines()
        if "Loaded image" in line
    ]
    logger.info("load_done", tags=tags)
    return tags


def verify_image(image: str, model: str = "tiny") -> Tuple[bool, str]:
    """Run smoke checks inside the image.

    Performs:
    1. ``clinfo -l`` to confirm the GPU device is visible.
    2. ``scripts/check_gpu.py``: llama.cpp (and whisper.cpp on Intel) on the GPU.
    3. ``faster-whisper`` model load (defaults to ``tiny`` for speed).

    Args:
        image: Image:tag to verify.
        model: Whisper model to attempt loading.

    Returns:
        Tuple ``(ok, summary)`` where summary is a human-readable report.
    """
    binary = require_docker()
    logger.info("verify_start", image=image, model=model)

    snippet = (
        "set -e; "
        "echo '=== clinfo ==='; clinfo -l 2>&1 | head -10; "
        "echo '=== GPU self-check ==='; python3 scripts/check_gpu.py; "
        "echo '=== faster-whisper load ==='; "
        f'python3 -c "from faster_whisper import WhisperModel; '
        f"m = WhisperModel('{model}', device='cpu', compute_type='int8'); print('OK')\"; "
        "echo '=== all checks passed ==='"
    )
    cmd = [
        binary,
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        "/dev/dri:/dev/dri",
        image,
        "bash",
        "-lc",
        snippet,
    ]
    try:
        result = _run(cmd, timeout=600, check=False)
    except ImageError as e:
        return False, f"verify failed: {e}"

    summary = (result.stdout or "") + (result.stderr or "")
    ok = result.returncode == 0 and "all checks passed" in summary
    logger.info("verify_done", image=image, ok=ok)
    return ok, summary[-4000:]


def release_image(image: str, stable_tag: Optional[str] = None) -> str:
    """Promote a verified image to the ``stable`` tag and update the manifest.

    Args:
        image: Image:tag that has passed verification.
        stable_tag: Explicit stable tag (default: ``pipeline:<backend>-stable``).

    Returns:
        The stable tag that was applied.
    """
    binary = require_docker()
    backend = image.split(":", 1)[1].split("-", 1)[0] if ":" in image else "cpu"
    if stable_tag is None:
        stable_tag = f"pipeline:{backend}-stable"

    logger.info("release_start", image=image, stable=stable_tag)
    _run([binary, "tag", image, stable_tag])

    manifest = _read_manifest()
    manifest["last_stable"] = {
        "image": stable_tag,
        "source": image,
        "released_at": datetime.now(timezone.utc).isoformat(),
        "size_mb": image_size_mb(stable_tag),
    }
    _write_manifest(manifest)
    logger.info("release_done", stable_tag=stable_tag)
    return stable_tag


def record_deployment(host: str, image: str, status: str = "ok", note: str = "") -> None:
    """Append a deployment entry to the release manifest.

    Args:
        host: Identifier of the target host (e.g. "cachyos-local", "vds-prod-01").
        image: Image:tag that was deployed.
        status: Free-form status (e.g. "ok", "failed", "rolled-back").
        note: Optional free-form note.
    """
    manifest = _read_manifest()
    manifest.setdefault("deployments", []).append(
        {
            "host": host,
            "image": image,
            "status": status,
            "note": note,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_manifest(manifest)
    logger.info("deployment_recorded", host=host, image=image, status=status)


def _read_manifest() -> dict:
    """Read the deployment manifest, returning an empty skeleton if absent."""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ImageError(f"manifest is not valid JSON: {MANIFEST_PATH}: {e}") from e


def _write_manifest(data: dict) -> None:
    """Atomically write the deployment manifest as pretty JSON."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + os.linesep,
        encoding="utf-8",
    )
    tmp.replace(MANIFEST_PATH)


def list_images(backend: Optional[str] = None) -> List[dict]:
    """List pipeline images available locally, optionally filtered by backend.

    Args:
        backend: If given, restrict to images whose tag starts with this backend.

    Returns:
        List of dicts with keys ``repository``, ``tag``, ``size``, ``created``.
    """
    binary = require_docker()
    fmt = "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    result = _run(
        [binary, "images", "--format", fmt, "pipeline"],
        timeout=60,
        check=False,
    )
    rows: List[dict] = []
    for line in (result.stdout or "").splitlines():
        if not line.strip():
            continue
        repo, tag, size, created = line.split("\t", 3)
        if backend is not None and not tag.startswith(backend + "-"):
            continue
        rows.append({"repository": repo, "tag": tag, "size": size, "created": created})
    return rows


def prune_local(backend: str, keep: Tuple[str, ...] = ("stable", "latest")) -> int:
    """Remove locally built images for ``backend`` that are NOT in the keep set.

    Args:
        backend: GPU backend whose images should be considered.
        keep: Suffix tokens that protect an image from removal.

    Returns:
        Number of images removed.
    """
    binary = require_docker()
    removed = 0
    for entry in list_images(backend=backend):
        tag = entry["tag"]
        if any(token in tag for token in keep):
            continue
        full = f"{entry['repository']}:{tag}"
        logger.info("prune_remove", image=full)
        _run([binary, "rmi", full], check=False, timeout=120)
        removed += 1
    return removed
