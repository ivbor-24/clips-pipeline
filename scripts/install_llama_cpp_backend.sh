#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Build and install llama-cpp-python for the selected GPU backend.
#
# PyPI ships llama-cpp-python only as source, and the prebuilt wheel indexes
# stop at versions too old for Qwen3/Gemma 4, so it is always compiled here
# with the backend's CMake flags. The version is the one pinned in uv.lock,
# so every install gets the same llama.cpp. install.sh, Dockerfile.backend and
# CI all call this script, so they cannot drift apart.
#
# Usage: install_llama_cpp_backend.sh <cpu|cuda|openvino|rocm> [python]
#   python   interpreter of the target environment (default: .venv/bin/python)
#
# Environment:
#   LLAMA_CMAKE_EXTRA  extra CMake flags. Builds that run on other machines
#                      pass `scripts/ggml_cpu_flags.sh portable`: a native
#                      build uses the build host's CPU instructions, and plain
#                      -DGGML_NATIVE=OFF still assumes AVX2 (SIGILL on older CPUs).
#   LLAMA_CPP_VERSION  version to build, instead of reading it from uv.lock.
#                      Dockerfile.backend passes the version it extracted into
#                      /llama-version, so the wheel stages never see uv.lock and
#                      are cached by version. Unset: read from uv.lock (native
#                      install.sh, which has no /llama-version).
#   LLAMA_FORCE_REBUILD=1  rebuild even if the installed build is up to date.
#   LLAMA_WHEEL_DIR    build a wheel into this directory instead of installing
#                      (python needs pip). Dockerfile.backend compiles the wheel
#                      this way in a stage with the build toolchain.
#
# A build that is already installed with the same version and CMake flags
# (recorded in llama_cpp/.build-info) is kept, so re-running install.sh does
# not spend several minutes recompiling llama.cpp.
# ============================================================================

BACKEND="${1:?Usage: $0 <cpu|cuda|openvino|rocm> [python]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${2:-$ROOT/.venv/bin/python}"

log() { echo "[install_llama_cpp_backend] $*"; }

case "$BACKEND" in
    cpu)
        CMAKE_FLAGS=""; BACKEND_LIB="" ;;
    openvino)
        # llama.cpp has no OpenVINO backend. Intel Arc GPUs (e.g. B580) run it
        # through the Vulkan backend; needs the Vulkan loader and headers
        # (libvulkan-dev, glslc) at build time.
        CMAKE_FLAGS="-DGGML_VULKAN=on"; BACKEND_LIB="libggml-vulkan.so" ;;
    cuda)
        # Builds against the CUDA toolkit on the host/image (needs nvcc).
        CMAKE_FLAGS="-DGGML_CUDA=on"; BACKEND_LIB="libggml-cuda.so" ;;
    rocm)
        # Not officially supported.
        CMAKE_FLAGS="-DGGML_HIP=on"; BACKEND_LIB="libggml-hip.so" ;;
    *)
        echo "[install_llama_cpp_backend] Unknown backend: $BACKEND" >&2
        echo "Usage: $0 <cpu|cuda|openvino|rocm> [python]" >&2
        exit 1 ;;
esac

# The package's version line follows its name line in uv.lock.
if [[ -n "${LLAMA_CPP_VERSION:-}" ]]; then
    VERSION="$LLAMA_CPP_VERSION"
else
    VERSION="$(awk '/^name = "llama-cpp-python"$/ { found = 1; next }
        found && /^version = / { gsub(/"/, "", $3); print $3; exit }' "$ROOT/uv.lock")"
    [ -n "$VERSION" ] || { echo "llama-cpp-python not found in uv.lock" >&2; exit 1; }
fi

CMAKE_ARGS="$(echo "$CMAKE_FLAGS ${LLAMA_CMAKE_EXTRA:-}" | xargs)"
BUILD_INFO="version=$VERSION backend=$BACKEND cmake_args=$CMAKE_ARGS"
# Parallel compile jobs (BUILD_JOBS from .env or automatic). A single build
# uses every core; the Docker build sets BUILD_JOBS_PARALLEL=1 so build_jobs.sh
# caps this stage and whisper.cpp, which compile side by side, to fit the RAM.
# Not part of BUILD_INFO: changing the number does not force a rebuild.
JOBS="$("$ROOT/scripts/build_jobs.sh")"

# Directory of the installed llama_cpp package, empty if it is not installed.
package_dir() {
    "$PYTHON_BIN" -c 'import os, llama_cpp; print(os.path.dirname(llama_cpp.__file__))' \
        2>/dev/null || true
}

# The stamp alone is not enough: a plain `uv sync --reinstall` replaces the
# package with a build without GPU flags and leaves the stamp behind, so the
# backend library must be present too.
if [[ -n "${LLAMA_WHEEL_DIR:-}" ]]; then
    log "backend=$BACKEND: building a wheel of llama-cpp-python==$VERSION into $LLAMA_WHEEL_DIR (CMAKE_ARGS='$CMAKE_ARGS', jobs=$JOBS)"
    CMAKE_ARGS="$CMAKE_ARGS" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" "$PYTHON_BIN" -m pip wheel \
        --no-deps --no-cache-dir \
        --wheel-dir "$LLAMA_WHEEL_DIR" "llama-cpp-python==$VERSION"
    exit 0
fi

PKG_DIR="$(package_dir)"
if [[ "${LLAMA_FORCE_REBUILD:-}" != "1" && -n "$PKG_DIR" \
    && -f "$PKG_DIR/.build-info" && "$(cat "$PKG_DIR/.build-info")" == "$BUILD_INFO" \
    && ( -z "$BACKEND_LIB" || -e "$PKG_DIR/lib/$BACKEND_LIB" ) ]]; then
    log "llama-cpp-python is up to date ($BUILD_INFO); set LLAMA_FORCE_REBUILD=1 to rebuild"
    exit 0
fi

log "backend=$BACKEND: building llama-cpp-python==$VERSION (CMAKE_ARGS='$CMAKE_ARGS', jobs=$JOBS)"
log "this compiles llama.cpp and takes several minutes"

# --no-cache: uv would otherwise reuse a wheel built earlier with other flags.
CMAKE_ARGS="$CMAKE_ARGS" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" uv pip install \
    --python "$PYTHON_BIN" --no-cache \
    --reinstall-package llama-cpp-python --no-deps "llama-cpp-python==$VERSION"

echo "$BUILD_INFO" > "$(package_dir)/.build-info"

log "llama-cpp-python $VERSION is installed for backend=$BACKEND."
