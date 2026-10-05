#!/usr/bin/env bash
# Number of parallel jobs for the heavy compile steps (whisper.cpp and
# llama-cpp-python).
#
# BUILD_JOBS (from .env, passed as a build arg) overrides the number.
#
# Without BUILD_JOBS the number is automatic:
#   - a single build (the native `./install.sh` compiles whisper.cpp and
#     llama-cpp-python one after the other) gets every core: nproc;
#   - the Docker build compiles the whisper.cpp stage and the llama-cpp-python
#     wheel stage in parallel, so each is capped to stay within the host's RAM.
#     The Dockerfile sets BUILD_JOBS_PARALLEL=1 for that; the cap is
#     min(nproc, max(1, MemTotal_GiB / 4)), computed from /proc/meminfo, which
#     during a build shows the host's memory.
#
# Usage: build_jobs.sh [ncpu] [memtotal_kib]
#   The two optional arguments let the tests exercise the formula; without
#   them nproc and /proc/meminfo are read.
set -euo pipefail

if [[ -n "${BUILD_JOBS:-}" ]]; then
    echo "$BUILD_JOBS"
    exit 0
fi

ncpu=${1:-$(nproc 2>/dev/null || echo 1)}

# One build runs alone: it can use every core. Only the Docker build's two
# parallel compile stages must cap each other so the pair fits in RAM.
if [[ "${BUILD_JOBS_PARALLEL:-0}" != "1" ]]; then
    echo "$ncpu"
    exit 0
fi

mem_kib=${2:-$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)}

# MemTotal is in KiB, 1 GiB = 1048576 KiB. A Vulkan compile takes ~100 MB of
# RAM per job; two parallel builds at MemTotal/4 jobs each fit comfortably.
by_mem=$(awk -v m="$mem_kib" 'BEGIN { j = int(m / 1048576 / 4); if (j < 1) j = 1; print j }')

jobs=$ncpu
[[ "$jobs" -gt "$by_mem" ]] && jobs=$by_mem
echo "$jobs"
