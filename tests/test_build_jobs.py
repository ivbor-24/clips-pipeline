"""Tests for scripts/build_jobs.sh (the BUILD_JOBS compile-jobs formula)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")
SCRIPT = REPO / "scripts" / "build_jobs.sh"


def run_build_jobs(ncpu, mem_kib, build_jobs="", parallel=False):
    env = {**os.environ, "BUILD_JOBS": build_jobs}
    if parallel:
        env["BUILD_JOBS_PARALLEL"] = "1"
    else:
        env["BUILD_JOBS_PARALLEL"] = "0"
    return subprocess.run(
        [BASH, str(SCRIPT), str(ncpu), str(mem_kib)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout.strip()


@pytest.mark.parametrize(
    ("ncpu", "mem_kib", "expected"),
    [
        (12, 16174720, "3"),  # ~15.4 GiB: min(12, floor(15.4/4)) = 3
        (12, 33554432, "8"),  # 32 GiB: min(12, 8) = 8
        (4, 8388608, "2"),  # 8 GiB, 4 cores: min(4, 2) = 2
        (12, 1048576, "1"),  # 1 GiB: max(1, 0) = 1
        (2, 33554432, "2"),  # nproc smaller than the memory bound
    ],
)
def test_automatic_jobs_parallel_build(ncpu, mem_kib, expected):
    """Two parallel Docker builds are each capped so the pair fits in RAM."""
    assert run_build_jobs(ncpu, mem_kib, parallel=True) == expected


@pytest.mark.parametrize(
    ("ncpu", "mem_kib"),
    [
        (12, 16174720),  # low memory must not throttle a single build
        (12, 1048576),
        (4, 8388608),
    ],
)
def test_single_build_uses_every_core(ncpu, mem_kib):
    """A single native build (install.sh) uses nproc regardless of memory."""
    assert run_build_jobs(ncpu, mem_kib) == str(ncpu)


def test_build_jobs_override():
    assert run_build_jobs(12, 16174720, build_jobs="2") == "2"
    assert run_build_jobs(12, 16174720, build_jobs="2", parallel=True) == "2"
