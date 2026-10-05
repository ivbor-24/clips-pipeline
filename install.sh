#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# AI Video Clips Pipeline — Installation Script
# Usage: ./install.sh [cpu|cuda|openvino|rocm]
# ============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

readonly GPU_MODE="${1:-cpu}"
readonly VALID_MODES=("cpu" "cuda" "openvino" "rocm")
readonly REPO_URL="https://github.com/ivbor-24/clips-pipeline.git"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SYSTEM_PACKAGES=(ffmpeg just python3 python3-dev build-essential cmake git curl)
readonly UV_VERSION="0.12.20"

# GPU-backend-specific system packages needed on top of SYSTEM_PACKAGES.
# openvino here means "Intel Arc via llama.cpp's Vulkan backend" (see
# scripts/install_llama_cpp_backend.sh) which needs a Vulkan ICD loader.
case "$GPU_MODE" in
    openvino) readonly GPU_SYSTEM_PACKAGES=(mesa-vulkan-drivers libvulkan1 libvulkan-dev glslc spirv-headers) ;;
    cuda)     readonly GPU_SYSTEM_PACKAGES=(nvidia-cuda-toolkit) ;;
    rocm)     readonly GPU_SYSTEM_PACKAGES=(rocm-opencl-runtime) ;;
    *)        readonly GPU_SYSTEM_PACKAGES=() ;;
esac

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()    { echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

validate_gpu_mode() {
    local valid=false
    for mode in "${VALID_MODES[@]}"; do
        if [[ "$mode" == "$GPU_MODE" ]]; then
            valid=true
            break
        fi
    done
    if [[ "$valid" == "false" ]]; then
        error "Invalid GPU mode: '$GPU_MODE'"
        echo "Usage: $0 [cpu|cuda|openvino|rocm]"
        echo "  cpu       — CPU-only (default)"
        echo "  cuda      — NVIDIA GPU acceleration"
        echo "  openvino  — Intel Arc GPU acceleration"
        echo "  rocm      — AMD GPU acceleration"
        exit 1
    fi
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root is not recommended. Consider running as a regular user."
    fi
}

install_system_deps() {
    step "Step 1/8: System dependencies"

    if command -v apt-get &>/dev/null; then
        install_apt_packages
    else
        info "No apt-get here: checking the requirements instead of installing them."
    fi

    # Whatever the distro, stop before anything is compiled if a tool or header
    # is missing; the check prints the install command for this system.
    "$SCRIPT_DIR/scripts/check_system_deps.sh" "$GPU_MODE" || exit 1
}

install_apt_packages() {
    info "Updating package lists..."
    sudo apt-get update -qq

    local to_install=()
    for pkg in "${SYSTEM_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null 2>&1; then
            to_install+=("$pkg")
        fi
    done

    if [[ ${#to_install[@]} -eq 0 ]]; then
        info "All system dependencies already installed."
    else
        info "Installing: ${to_install[*]}"
        sudo apt-get install -y "${to_install[@]}"
    fi

    # GPU-backend packages (Vulkan for openvino/Intel Arc, CUDA toolkit, ROCm).
    # These often live in vendor apt repos that may not be configured on every
    # host. A failure is reported here, and check_system_deps.sh then stops the
    # install: a GPU mode never silently turns into a CPU install.
    if [[ ${#GPU_SYSTEM_PACKAGES[@]} -gt 0 ]]; then
        info "Installing GPU backend packages for '$GPU_MODE': ${GPU_SYSTEM_PACKAGES[*]}"
        if ! sudo apt-get install -y "${GPU_SYSTEM_PACKAGES[@]}"; then
            warn "Failed to install GPU backend packages (${GPU_SYSTEM_PACKAGES[*]})."
            warn "You may need to add the vendor apt repo first."
        fi
    fi
}

ensure_repo() {
    step "Step 2/8: Checking project repository"

    if [[ -f "pyproject.toml" ]] && [[ -d "src" ]]; then
        info "Repository already present at $(pwd)."
    else
        info "Cloning repository..."
        if [[ -d "clips-pipeline" ]]; then
            warn "Directory 'clips-pipeline' already exists. Skipping clone."
        else
            git clone "$REPO_URL"
            cd clips-pipeline
        fi
    fi

    info "Repository ready."
}

ensure_uv() {
    if command -v uv &>/dev/null; then
        return
    fi
    info "Installing uv $UV_VERSION (Python package manager)..."
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null || { error "uv installation failed"; exit 1; }
}

install_python_deps() {
    step "Step 3/8: Installing Python dependencies (GPU mode: $GPU_MODE)"
    ensure_uv

    # One extra per backend picks the torch build. ROCm is not officially
    # supported: it gets the CPU build and only llama.cpp is compiled for the GPU.
    local extra="$GPU_MODE"
    [[ "$GPU_MODE" == "rocm" ]] && extra="cpu"

    # Exact versions from uv.lock; llama-cpp-python is compiled below.
    # install.sh is the development setup (end users get Docker), so it also
    # installs the test and lint tools: `just test` and `just lint` work
    # right after it.
    info "Running: uv sync --frozen --extra $extra --extra test --extra dev"
    uv sync --frozen --inexact --extra "$extra" --extra test --extra dev \
        --no-install-package llama-cpp-python

    info "Building llama-cpp-python for GPU backend: $GPU_MODE"
    "$SCRIPT_DIR/scripts/install_llama_cpp_backend.sh" "$GPU_MODE" "$SCRIPT_DIR/.venv/bin/python"

    # faster-whisper (CTranslate2) cannot use Intel GPUs, so on Intel Arc the
    # transcription runs through whisper.cpp with Vulkan (profile intel_arc).
    if [[ "$GPU_MODE" == "openvino" ]]; then
        info "Building whisper.cpp (Vulkan) for transcription on Intel Arc"
        if ! "$SCRIPT_DIR/scripts/install_whisper_cpp.sh" openvino; then
            error "whisper.cpp could not be built with Vulkan (see the output above)."
            error "Without it transcription would run on the CPU, several times slower."
            error "Fix the error and run ./install.sh openvino again, or install for CPU: ./install.sh cpu"
            exit 1
        fi
    fi

    info "Python dependencies installed into .venv."
}

setup_dirs() {
    step "Step 4/8: Creating working directories"

    if command -v just &>/dev/null; then
        just setup-dirs
    else
        warn "'just' not found. Creating directories manually."
        mkdir -p artifacts/logs artifacts/cache output/clips input_videos
    fi

    info "Directories ready."
}

validate_config() {
    step "Step 5/8: Validating configuration"

    if command -v just &>/dev/null; then
        just validate-config || warn "Configuration validation failed. Check config/config.yaml."
    else
        warn "'just' not found. Skipping config validation."
        warn "Run manually: just validate-config"
    fi
}

prefetch_models_prompt() {
    step "Step 6/8: Pre-download AI models"

    info "Whisper + the configured LLM model can be downloaded now (this can"
    info "be several GB and take a while) instead of on the first 'just process'."

    echo ""
    read -rp "$(echo -e "${YELLOW}Pre-download models now? [y/N] ${NC}")" answer
    answer="${answer:-N}"

    if [[ "${answer,,}" =~ ^(y|yes)$ ]]; then
        if command -v just &>/dev/null; then
            # Intel Arc transcribes with whisper.cpp: fetch its ggml model instead.
            local profile=""
            [[ "$GPU_MODE" == "openvino" ]] && profile="intel_arc"
            just prefetch-models config/config.yaml "$profile" \
                || warn "Model prefetch failed. Re-run later with: just prefetch-models"
        else
            warn "'just' not found. Skipping model prefetch."
            warn "Run manually: python3 scripts/prefetch_models.py"
        fi
    else
        info "Skipping. Models will download on first use, or run: just prefetch-models"
    fi
}

check_gpu() {
    step "Step 7/8: GPU self-check"

    # Asks llama.cpp, whisper.cpp and CTranslate2 which device they would use.
    # A GPU install that ends up on the CPU is an error, not a slow success.
    if ! "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/scripts/check_gpu.py" --backend "$GPU_MODE"; then
        error "The GPU is not used as expected (see the checks above)."
        error "Fix it and check again with: just check-gpu $GPU_MODE"
        exit 1
    fi
}

purge_uv_cache() {
    step "Step 8/8: Post-install cleanup"

    local cache_size
    cache_size=$(du -sh "$(uv cache dir 2>/dev/null)" 2>/dev/null | cut -f1 || echo "unknown")
    info "Current uv cache size: ${cache_size:-unknown}"

    echo ""
    read -rp "$(echo -e "${YELLOW}Clean the uv download cache to free disk space? [Y/n] ${NC}")" answer
    answer="${answer:-Y}"

    if [[ "${answer,,}" =~ ^(y|yes)$ ]]; then
        uv cache clean 2>/dev/null || warn "uv cache clean failed (may already be empty)."
        info "uv cache cleaned."
    else
        info "Skipping cache cleanup."
    fi
}

print_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   Installation complete!                     ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  GPU mode:        ${CYAN}${GPU_MODE}${NC}"
    echo -e "  Dry-run test:    ${CYAN}just process /dev/null config/config.yaml True${NC}"
    echo -e "  Prefetch models: ${CYAN}just prefetch-models${NC}"
    echo -e "  GPU check:       ${CYAN}just check-gpu${NC}"
    echo -e "  Run tests:       ${CYAN}just test${NC}"
    echo -e "  Process video:   ${CYAN}just process input_videos/video.mp4${NC}"
    echo ""
}

main() {
    echo -e "${BLUE}"
    echo "  ╔═══════════════════════════════════════════╗"
    echo "  ║   AI Video Clips Pipeline — Installer     ║"
    echo "  ╚═══════════════════════════════════════════╝"
    echo -e "${NC}"

    validate_gpu_mode
    check_root
    install_system_deps
    ensure_repo
    install_python_deps
    setup_dirs
    validate_config
    prefetch_models_prompt
    check_gpu
    purge_uv_cache
    print_summary
}

main "$@"
