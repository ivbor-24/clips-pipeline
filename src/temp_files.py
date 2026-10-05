"""
Temporary files of a job

Stages keep scratch data (audio chunks for transcription, frames for face
detection, a partial download) inside the job's work directory, never in
``/tmp``: ``/tmp`` is RAM on many distributions, and a job stopped by a signal
gets no chance to clean up after itself. The job worker removes these
directories when a job's process ends, whatever the outcome, and for every job
when it starts (after a crash or reboot).

Inputs:
- <work_dir>/artifacts/{tmp,temp_download,audio/temp_chunks}

Outputs:
- the same directories, removed
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Union

import structlog

logger = structlog.get_logger("temp_files")

# Relative to the job's work directory. Nothing in them is a resume input:
# the stage that made them recreates them when it runs again.
JOB_TEMP_DIRS = (
    Path("artifacts") / "tmp",
    Path("artifacts") / "temp_download",
    Path("artifacts") / "audio" / "temp_chunks",
)

# Older versions extracted face-detection frames here.
LEGACY_TMP_PREFIX = "face_crop_"


def job_temp_dir(work_dir: Union[str, Path]) -> Path:
    """Scratch directory for a stage of the job in ``work_dir``."""
    path = Path(work_dir) / JOB_TEMP_DIRS[0]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _remove_dirs(dirs: Iterable[Path]) -> int:
    freed = 0
    for path in dirs:
        if not path.is_dir():
            continue
        try:
            freed += _size(path)
        except OSError:
            pass
        shutil.rmtree(path, ignore_errors=True)
    return freed


def remove_job_temp_files(work_dir: Union[str, Path]) -> int:
    """Remove the temporary directories of one job.

    Returns:
        Bytes freed.
    """
    freed = _remove_dirs(Path(work_dir) / rel for rel in JOB_TEMP_DIRS)
    if freed:
        logger.info("job_temp_files_removed", work_dir=str(work_dir), bytes_freed=freed)
    return freed


def remove_legacy_tmp_files() -> int:
    """Remove frame directories older versions left in ``/tmp`` when a job was killed.

    Only directories with this pipeline's prefix that belong to the current user.

    Returns:
        Bytes freed.
    """
    uid = os.getuid()
    candidates = [
        p
        for p in Path(tempfile.gettempdir()).glob(f"{LEGACY_TMP_PREFIX}*")
        if p.is_dir() and p.stat().st_uid == uid
    ]
    freed = _remove_dirs(candidates)
    if candidates:
        logger.info("legacy_tmp_files_removed", dirs=len(candidates), bytes_freed=freed)
    return freed
