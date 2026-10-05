#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Build whisper.cpp's whisper-cli for transcription.engine=whisper_cpp.
#
# faster-whisper (CTranslate2) has no Intel GPU backend, so on Intel Arc the
# transcription runs through whisper.cpp with Vulkan (profile intel_arc).
# The binary lands where that profile expects it:
#   artifacts/cache/whisper-cpp-bin/whisper-cli
# The ggml model is downloaded by `just prefetch-models` (pinned and
# checksummed in config/models.lock.yaml).
#
# Usage: install_whisper_cpp.sh <cpu|cuda|openvino|rocm> [--force]
#   --force  replace a whisper-cli that this script did not build (for example
#            a symlink to a manual build)
#
# Environment:
#   WHISPER_CPP_VERSION  git tag to build (default below)
#   WHISPER_CMAKE_EXTRA  extra CMake flags; builds that run on other machines
#                        need `scripts/ggml_cpu_flags.sh portable`
#   WHISPER_CPP_BIN_DIR  where whisper-cli goes (default: artifacts/cache/
#                        whisper-cpp-bin; the Docker image uses a directory in
#                        PATH that no volume hides)
# ============================================================================

BACKEND="${1:?Usage: $0 <cpu|cuda|openvino|rocm> [--force]}"
FORCE="${2:-}"
VERSION="${WHISPER_CPP_VERSION:-v1.9.4}"
REPO_URL="https://github.com/ggml-org/whisper.cpp"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/artifacts/cache/whisper-cpp-src"
BIN_DIR="${WHISPER_CPP_BIN_DIR:-$ROOT/artifacts/cache/whisper-cpp-bin}"
BIN="$BIN_DIR/whisper-cli"
STAMP="$BIN_DIR/.build-info"

log() { echo "[install_whisper_cpp] $*"; }

case "$BACKEND" in
    cpu)      CMAKE_FLAGS="" ;;
    openvino) CMAKE_FLAGS="-DGGML_VULKAN=ON" ;;  # Intel Arc: Vulkan (libvulkan-dev, glslc)
    cuda)     CMAKE_FLAGS="-DGGML_CUDA=ON" ;;
    rocm)     CMAKE_FLAGS="-DGGML_HIP=ON" ;;     # not officially supported
    *)
        echo "[install_whisper_cpp] Unknown backend: $BACKEND" >&2
        echo "Usage: $0 <cpu|cuda|openvino|rocm> [--force]" >&2
        exit 1 ;;
esac
CMAKE_FLAGS="$(echo "$CMAKE_FLAGS ${WHISPER_CMAKE_EXTRA:-}" | xargs)"
BUILD_INFO="version=$VERSION backend=$BACKEND flags=$CMAKE_FLAGS"
# Parallel compile jobs (BUILD_JOBS from .env or automatic). A single build
# uses every core; the Docker build sets BUILD_JOBS_PARALLEL=1 so build_jobs.sh
# caps this stage and the llama-cpp-python wheel stage, which compile side by
# side, to fit the host's RAM.
JOBS="$("$ROOT/scripts/build_jobs.sh")"

if [[ -e "$BIN" || -L "$BIN" ]]; then
    if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$BUILD_INFO" && ! -L "$BIN" ]]; then
        log "whisper-cli is up to date ($BUILD_INFO)"
        exit 0
    fi
    if [[ ! -f "$STAMP" && "$FORCE" != "--force" ]]; then
        log "$BIN exists but was not built by this script (e.g. a symlink to a manual build)."
        log "Leaving it as is. Re-run with --force to replace it."
        exit 0
    fi
fi

for tool in git cmake c++; do
    command -v "$tool" >/dev/null || { echo "[install_whisper_cpp] $tool is required" >&2; exit 1; }
done

log "building whisper.cpp $VERSION for backend=$BACKEND (CMake flags: '${CMAKE_FLAGS}', jobs=$JOBS)"
rm -rf "$SRC_DIR"
git clone --quiet --depth 1 --branch "$VERSION" "$REPO_URL" "$SRC_DIR"

# Static build: whisper-cli must not depend on libraries in the build tree,
# which is deleted below.
# shellcheck disable=SC2086
cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    $CMAKE_FLAGS
cmake --build "$SRC_DIR/build" --target whisper-cli -j "$JOBS"

mkdir -p "$BIN_DIR"
rm -f "$BIN"
install -m 755 "$SRC_DIR/build/bin/whisper-cli" "$BIN"
echo "$BUILD_INFO" >"$STAMP"
rm -rf "$SRC_DIR"

"$BIN" --help >/dev/null 2>&1 || { echo "[install_whisper_cpp] $BIN does not run" >&2; exit 1; }
log "installed $BIN ($BUILD_INFO)"
