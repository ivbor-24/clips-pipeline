#!/usr/bin/env bash
# One-command installation with Docker (docs/INSTALL.md).
#
# Usage: ./setup.sh [options]
#   --backend cpu|cuda|openvino  GPU backend (default: detected)
#   --models-dir DIR             where the model weights live (default: asks)
#   --yes                        take the default answers instead of asking
#   --no-models                  do not download the models
#   --no-start                   do not start the web UI at the end
#   --build                      build the images here even if published ones fit
#
# Checks Docker, detects the GPU, writes .env (keeping the values already
# there), creates the data directories, downloads the published images of
# this release (or builds them when the code is not a published release),
# downloads the models, checks that the job worker really uses the GPU and
# starts the web UI. Run it again after `git pull`: it updates the images and
# keeps the settings.
#
# Environment (used by the tests):
#   SETUP_DRM_DIR         where to look for GPUs (default: /sys/class/drm)
#   SETUP_DEV_DRI         GPU device nodes (default: /dev/dri)
#   SETUP_NEED_GB_IMAGES  free space for the images and build cache (default: 20)
#   SETUP_NEED_GB_MODELS  free space for the models (default: 12)
#   SETUP_IMAGE_REPO      published images (default: ghcr.io/ivbor-24/clips-pipeline)
set -euo pipefail

# shellcheck source=scripts/docker_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/scripts/docker_common.sh"
cd "$ROOT"

DRM_DIR="${SETUP_DRM_DIR:-/sys/class/drm}"
DEV_DRI="${SETUP_DEV_DRI:-/dev/dri}"
MIN_COMPOSE_VERSION="2.20"
# CUDA 12 (Dockerfile.backend) runs on NVIDIA drivers from 525 on.
MIN_NVIDIA_DRIVER=525
BACKENDS=(cpu cuda openvino)
# Suggested when an API provider is chosen; the user may type another.
DEFAULT_OPENAI_MODEL="gpt-6-luna"
DEFAULT_ANTHROPIC_MODEL="claude-opus-5"

BACKEND_FLAG=""
MODELS_DIR_FLAG=""
NO_MODELS=false
NO_START=false
BUILD_FLAG=false
IMAGE_REPO="${SETUP_IMAGE_REPO:-ghcr.io/ivbor-24/clips-pipeline}"
# What goes into the images (Dockerfile.backend, Dockerfile.frontend): with
# changes there the published images do not fit. config/ is mounted instead.
IMAGE_INPUTS=(src scripts web Dockerfile.backend Dockerfile.frontend pyproject.toml uv.lock nginx.conf)

# Set while the script runs.
BACKEND=""
RENDER_GROUP=""
CUDA_ARCHS=""
MODELS_CHOICE=""
LOW_GPU_MEMORY=false
# What the jobs use as the LLM now (llm_status fills these).
LLM_ENABLED=""
LLM_PROVIDER=""
LLM_MODEL=""
LLM_API_BASE=""
LLM_READY=""
# The published images of this exact commit (find_published_images), empty
# when the images are built here.
PUBLISHED_BACKEND=""
PUBLISHED_WEB=""
# Output of the last `docker compose build` (build_once sets it; build_images
# reads it to tell an out-of-memory kill apart from other failures).
BUILD_LOG=""

usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

step() { echo -e "\n${C_CYAN}━━━ $* ━━━${C_OFF}"; }

interactive() { [[ "$ASSUME_YES" != true && -t 0 ]]; }

# ask QUESTION DEFAULT: read a line, or take the default without a terminal.
ask() {
    local question=$1 default=$2 answer
    if ! interactive; then
        echo "$default"
        return
    fi
    read -rp "$question [$default]: " answer
    echo "${answer:-$default}"
}

version_ge() { printf '%s\n%s\n' "$2" "$1" | sort -V -C; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --backend)
                BACKEND_FLAG=${2:-}
                shift ;;
            --models-dir)
                MODELS_DIR_FLAG=${2:-}
                shift ;;
            --yes | -y) ASSUME_YES=true ;;
            --no-models) NO_MODELS=true ;;
            --no-start) NO_START=true ;;
            --build) BUILD_FLAG=true ;;
            -h | --help)
                usage
                exit 0 ;;
            *)
                usage >&2
                exit 2 ;;
        esac
        shift
    done
    if [[ -n "$BACKEND_FLAG" ]] && ! valid_backend "$BACKEND_FLAG"; then
        die "Unknown backend '$BACKEND_FLAG': use cpu, cuda or openvino."
    fi
}

valid_backend() {
    local b
    for b in "${BACKENDS[@]}"; do [[ "$1" == "$b" ]] && return 0; done
    return 1
}

# ---- 1. Docker --------------------------------------------------------------

check_tools() {
    local version
    check_docker
    if ! version=$(docker compose version --short 2>/dev/null); then
        die "Docker Compose v2 is missing. $(docker_install_hint)"
    fi
    version=${version#v}
    version_ge "$version" "$MIN_COMPOSE_VERSION" ||
        die "Docker Compose $version is too old: $MIN_COMPOSE_VERSION or newer is needed. $(docker_install_hint)"
    docker buildx version >/dev/null 2>&1 ||
        die "Docker buildx (needed to build the images) is missing. $(docker_install_hint)"
    info "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null), Compose $version, buildx: OK"
}

# ---- 2. GPU -----------------------------------------------------------------

# gpu_name DRM_ENTRY: a readable name of the GPU behind /sys/class/drm/<entry>.
gpu_name() {
    local entry=$1 slot
    slot=$(basename "$(readlink -f "$entry/device")")
    if command -v lspci >/dev/null 2>&1 && lspci -s "$slot" >/dev/null 2>&1; then
        lspci -s "$slot" | head -n 1 | cut -d: -f3- | sed 's/^ *//'
    else
        echo "PCI device $(cat "$entry/device/vendor" 2>/dev/null):$(cat "$entry/device/device" 2>/dev/null)"
    fi
}

# intel_render_node: the first Intel GPU render node (renderD*), empty if none.
intel_render_node() {
    local entry
    for entry in "$DRM_DIR"/renderD*; do
        [[ "$(cat "$entry/device/vendor" 2>/dev/null)" == "0x8086" ]] || continue
        basename "$entry"
        return
    done
}

# detect_gpu: prints "backend<TAB>name of the GPU".
detect_gpu() {
    local node entry
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        printf 'cuda\t%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
        return
    fi
    node=$(intel_render_node)
    if [[ -n "$node" ]]; then
        printf 'openvino\t%s\n' "$(gpu_name "$DRM_DIR/$node")"
        return
    fi
    for entry in "$DRM_DIR"/renderD*; do
        if [[ "$(cat "$entry/device/vendor" 2>/dev/null)" == "0x1002" ]]; then
            printf 'cpu\t%s (AMD GPUs are not supported, the CPU is used)\n' "$(gpu_name "$entry")"
            return
        fi
    done
    printf 'cpu\tno supported GPU found\n'
}

choose_backend() {
    local detected name current default answer
    IFS=$'\t' read -r detected name < <(detect_gpu)
    info "GPU: $name -> backend $detected"
    if [[ -n "$BACKEND_FLAG" ]]; then
        BACKEND=$BACKEND_FLAG
    else
        # A backend chosen in an earlier run stays, unless changed here.
        current=""
        [[ -f "$ENV_FILE" ]] && current=$(env_value GPU_BACKEND)
        default=${current:-$detected}
        if [[ -n "$current" && "$current" != "$detected" ]]; then
            warn ".env has GPU_BACKEND=$current, the detected GPU suggests $detected."
        fi
        while true; do
            answer=$(ask "GPU backend: cpu, cuda (NVIDIA) or openvino (Intel)" "$default")
            valid_backend "$answer" && break
            interactive || die "Unknown backend '$answer' (GPU_BACKEND in .env): use cpu, cuda or openvino."
            warn "Unknown backend '$answer'."
        done
        BACKEND=$answer
    fi
    info "Backend: $BACKEND"
    check_backend
}

check_backend() {
    local node driver
    case "$BACKEND" in
        cuda)
            if ! { command -v nvidia-smi && nvidia-smi -L; } >/dev/null 2>&1; then
                die "The cuda backend needs the NVIDIA driver (nvidia-smi does not work)."
            fi
            driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
            if [[ -n "$driver" ]] && ! version_ge "$driver" "$MIN_NVIDIA_DRIVER"; then
                die "NVIDIA driver $driver is too old for CUDA 12: update it to $MIN_NVIDIA_DRIVER or newer."
            fi
            # Compile llama.cpp for this machine's GPUs only: minutes instead
            # of the long build for every GPU generation. "8.6" -> "86".
            CUDA_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
                tr -d ' .' | grep -E '^[0-9]+$' | sort -u | paste -sd ';' || true)
            [[ "$(docker info --format '{{json .Runtimes}}' 2>/dev/null)" == *'"nvidia"'* ]] ||
                die "Docker cannot use the NVIDIA GPU: install the NVIDIA Container Toolkit" \
                    "(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)," \
                    "then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
            ;;
        openvino)
            node=$(intel_render_node)
            [[ -n "$node" && -e "$DEV_DRI/$node" ]] ||
                die "The openvino backend needs an Intel GPU, and there is none under $DEV_DRI."
            # The containers get the group that owns the device, whatever its
            # name is on this system (render, video).
            RENDER_GROUP=$(stat -c %g "$DEV_DRI/$node")
            ;;
        cpu)
            warn "Without a GPU everything runs on the CPU: choosing clips with the LLM takes hours per video."
            ;;
    esac
}

# ---- 3. Settings ------------------------------------------------------------

has_models() {
    compgen -G "$1/*.gguf" >/dev/null || compgen -G "$1/ggml-*.bin" >/dev/null
}

abs_path() {
    case "$1" in
        /*) echo "$1" ;;
        *) echo "$ROOT/${1#./}" ;;
    esac
}

choose_models_dir() {
    local current default answer
    current=$(env_value MODELS_DIR ./models)
    default=$current
    # ./install.sh (the native install) keeps the weights here; offer it (for
    # --yes too) when the current directory has no models but this one does.
    if ! has_models "$(abs_path "$current")" && has_models "$ROOT/artifacts/cache/models"; then
        default=./artifacts/cache/models
    fi
    if [[ -n "$MODELS_DIR_FLAG" ]]; then
        answer=$MODELS_DIR_FLAG
    else
        answer=$(ask "Directory for the model weights (~11 GB; one that already has them saves the download)" "$default")
    fi
    answer=${answer/#\~/$HOME}
    MODELS_CHOICE=$answer
    set_env MODELS_DIR "$MODELS_CHOICE"
    if has_models "$(abs_path "$MODELS_CHOICE")"; then
        info "Models directory: $MODELS_CHOICE (models already there, the download is skipped)."
    else
        info "Models directory: $MODELS_CHOICE (no models found there, they will be downloaded)."
    fi
}

generate_secret() {
    head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 40
}

read_password() {
    local first second
    while true; do
        read -rsp "Password for the web UI (at least 8 characters): " first
        echo
        read -rsp "Repeat it: " second
        echo
        if [[ "$first" != "$second" ]]; then
            warn "The passwords differ."
        elif ((${#first} < 8)); then
            warn "Too short."
        elif [[ "$first" == *"'"* ]]; then
            warn "Please do not use the ' character."
        else
            set_env API_PASSWORD "$first"
            return
        fi
    done
}

choose_access() {
    local bind lan_default=N password secret
    bind=$(env_value BIND_ADDRESS 127.0.0.1)
    [[ "$bind" == "0.0.0.0" ]] && lan_default=Y
    password=$(env_value API_PASSWORD)
    if confirm "Open the web UI to other devices on your network (phone, laptop)?" "$lan_default"; then
        set_env BIND_ADDRESS 0.0.0.0
        # The network needs a password.
        if [[ -z "$password" ]]; then
            interactive || die "Opening the web UI to the network needs a password:" \
                "set API_PASSWORD in .env, or run ./setup.sh in a terminal."
            read_password
        elif interactive && ! confirm "Keep the current password?" Y; then
            read_password
        fi
    else
        set_env BIND_ADDRESS 127.0.0.1
    fi
    password=$(env_value API_PASSWORD)
    if [[ -n "$password" ]]; then
        secret=$(env_value API_JWT_SECRET)
        if ((${#secret} < 16)) || [[ "$secret" == "change-me-in-production" ]]; then
            set_env API_JWT_SECRET "$(generate_secret)"
            info "Generated API_JWT_SECRET (signs the login sessions)."
        fi
        chmod 600 "$ENV_FILE"
        info "Web UI password: set. Every browser, this computer's too, asks for it."
    fi
}

choose_port() {
    local port free owner
    port=$(env_value WEB_PORT 8080)
    if port_in_use "$port" && ! grep -qx frontend <<<"$(running_services)"; then
        owner=$(port_owner "$port")
        if [[ -n "$(app_version "$port")" ]]; then
            # Another copy of this project: `just up` offers to stop it.
            info "Port $port: another copy of AI Video Clips Pipeline runs there."
        else
            free=$port
            while port_in_use "$free"; do free=$((free + 1)); done
            warn "Port $port is taken by another program${owner:+ ($owner)}."
            if confirm "Use port $free for the web UI?" Y; then
                port=$free
            fi
        fi
    fi
    set_env WEB_PORT "$port"
}

write_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        cp "$ROOT/.env.example" "$ENV_FILE"
        info "Created .env from .env.example."
    else
        info "Updating .env (other values stay as they are)."
    fi
    set_env GPU_BACKEND "$BACKEND"
    set_env COMPOSE_PROFILES "$BACKEND"
    set_env HOST_UID "$(id -u)"
    set_env HOST_GID "$(id -g)"
    [[ "$BACKEND" == openvino ]] && set_env RENDER_GID "$RENDER_GROUP"
    if [[ "$BACKEND" == cuda && -n "$CUDA_ARCHS" ]]; then
        set_env CUDA_ARCHITECTURES "$CUDA_ARCHS"
    fi
    choose_models_dir
    choose_access
    choose_port
    info "Settings: backend $BACKEND, models in $MODELS_CHOICE, web UI on" \
        "$(env_value BIND_ADDRESS):$(env_value WEB_PORT)"
}

# ---- 4. Directories and disk space -----------------------------------------

make_dirs() {
    local dir bad=() foreign
    for dir in data jobs uploads cache cache/models "$(abs_path "$MODELS_CHOICE")"; do
        mkdir -p "$dir" 2>/dev/null || true
        [[ -d "$dir" && -w "$dir" ]] || bad+=("$dir")
    done
    ((${#bad[@]} == 0)) ||
        die "Not writable for you (created as root by an earlier run?): ${bad[*]}." \
            "Fix: sudo chown -R $(id -u):$(id -g) ${bad[*]}"
    # Containers run as you and cannot write into job files owned by root.
    foreign=$(find data jobs uploads -maxdepth 2 ! -user "$(id -u)" -print -quit 2>/dev/null)
    [[ -z "$foreign" ]] ||
        die "$foreign belongs to another user (an earlier run as root?)." \
            "Fix: sudo chown -R $(id -u):$(id -g) data jobs uploads cache"
    info "Directories: data jobs uploads cache $MODELS_CHOICE"
}

free_gb() { df -Pk "$1" 2>/dev/null | awk 'NR == 2 { print int($4 / 1048576) }' || true; }
filesystem() { df -Pk "$1" 2>/dev/null | awk 'NR == 2 { print $1 }' || true; }

check_disk() {
    local docker_dir models_dir need_images need_models free
    docker_dir=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    models_dir=$(abs_path "$MODELS_CHOICE")
    need_images=${SETUP_NEED_GB_IMAGES:-20}
    # NVIDIA: the CUDA toolkit image (~9 GB, only for compiling) and torch's
    # CUDA libraries.
    [[ "$BACKEND" == cuda ]] && need_images=$((need_images + 10))
    # Downloaded images: ~5 GB (openvino, cpu) or ~10 GB (cuda), no build cache.
    if [[ -n "$PUBLISHED_BACKEND" ]]; then
        need_images=8
        [[ "$BACKEND" == cuda ]] && need_images=14
    fi
    # A rebuild reuses most of the image and the build cache.
    docker image inspect "clips-pipeline:$BACKEND" >/dev/null 2>&1 && need_images=5
    need_models=${SETUP_NEED_GB_MODELS:-12}
    if [[ "$NO_MODELS" == true ]]; then
        need_models=0
    elif compgen -G "$models_dir/*.gguf" >/dev/null; then
        need_models=3
    fi
    if [[ -n "$docker_dir" && "$(filesystem "$docker_dir")" == "$(filesystem "$models_dir")" ]]; then
        need_images=$((need_images + need_models))
        need_models=0
    fi
    free=$(free_gb "$docker_dir")
    if [[ -n "$free" ]] && ((free < need_images)); then
        warn "$free GB free for Docker ($docker_dir); about $need_images GB is needed."
        confirm "Continue anyway?" N ||
            die "Free some disk space and run ./setup.sh again (old build cache: docker builder prune)."
    fi
    free=$(free_gb "$models_dir")
    if ((need_models > 0)) && [[ -n "$free" ]] && ((free < need_models)); then
        warn "$free GB free for the models ($models_dir); about $need_models GB is needed."
        confirm "Continue anyway?" N || die "Free some disk space, or choose another --models-dir."
    fi
    info "Disk space: OK"
}

# ---- 5. Images: download or build -------------------------------------------

# image_revision IMAGE: the commit a published image was built from (its
# org.opencontainers.image.revision label), read from the registry without
# downloading the image; empty when there is no such image.
image_revision() {
    # The JSON is indented: "key": "value", with a space after the colon.
    docker buildx imagetools inspect "$1" --format '{{json .Image}}' 2>/dev/null |
        grep -o '"org.opencontainers.image.revision": *"[0-9a-f]*"' | head -n 1 |
        sed 's/.*"\([0-9a-f]*\)"$/\1/' || true
}

# find_published_images: the published images fit when this checkout is a
# release whose images were built from this very commit (the release branch,
# a release tag) and nothing that goes into the images was changed here.
find_published_images() {
    local version sha changes
    if [[ "$BUILD_FLAG" == true ]]; then
        info "Images: built here (--build)."
        return
    fi
    if ! sha=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null); then
        info "Images: built here (not a git checkout, so no published images to match)."
        return
    fi
    changes=$(git -C "$ROOT" status --porcelain -- "${IMAGE_INPUTS[@]}" 2>/dev/null)
    if [[ -n "$changes" ]]; then
        info "Images: built here (local changes to the code)."
        return
    fi
    version=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' src/__init__.py)
    if [[ "$(image_revision "$IMAGE_REPO:$version-$BACKEND")" != "$sha" ||
        "$(image_revision "$IMAGE_REPO-web:$version")" != "$sha" ]]; then
        info "Images: built here (no published images of this commit; releases have them:" \
            "git clone -b release, see docs/INSTALL.md)."
        return
    fi
    PUBLISHED_BACKEND=$IMAGE_REPO:$version-$BACKEND
    PUBLISHED_WEB=$IMAGE_REPO-web:$version
    info "Images: the published ones of version $version ($PUBLISHED_BACKEND), nothing to compile."
}

# pull_images: download the published images under the names
# docker-compose.yml uses, so `docker compose up` does not build them.
pull_images() {
    info "Downloading the images (~$([[ "$BACKEND" == cuda ]] && echo 10 || echo 5) GB)..."
    docker pull "$PUBLISHED_BACKEND" ||
        die "Downloading $PUBLISHED_BACKEND failed (see above). Run ./setup.sh again," \
            "or build the images here: ./setup.sh --build"
    docker pull "$PUBLISHED_WEB" ||
        die "Downloading $PUBLISHED_WEB failed (see above). Run ./setup.sh again," \
            "or build the images here: ./setup.sh --build"
    docker tag "$PUBLISHED_BACKEND" "clips-pipeline:$BACKEND"
    docker tag "$PUBLISHED_WEB" "clips-pipeline-web:latest"
}

check_network() {
    local base probe network
    network=$(env_value SETUP_NETWORK)
    if [[ -n "$network" ]]; then
        info "Building and downloading over the $network network (SETUP_NETWORK in .env)."
        return
    fi
    base=$(sed -n 's/^FROM \([^ ]*\) AS base$/\1/p' Dockerfile.backend)
    probe='import socket; socket.create_connection(("pypi.org", 443), timeout=15)'
    info "Checking that containers can reach the internet..."
    docker pull -q "$base" >/dev/null ||
        die "Docker cannot download images (docker pull $base failed): check the internet connection and Docker's proxy settings."
    docker run --rm "$base" python3 -c "$probe" >/dev/null 2>&1 && return
    if docker run --rm --network host "$base" python3 -c "$probe" >/dev/null 2>&1; then
        warn "Containers have no internet access on Docker's network, but have it on the host network."
        warn "Usually a firewall (ufw, firewalld) blocks Docker's forwarding."
        confirm "Build the images and download the models over the host network?" Y ||
            die "Fix Docker's network access and run ./setup.sh again (docs/INSTALL.md, 'Если что-то не так')."
        set_env SETUP_NETWORK host
        warn "Jobs from a URL (YouTube) and LLM API providers fail until Docker's network gets internet" \
            "access; uploaded videos with the local LLM work."
    else
        die "Containers cannot reach the internet (pypi.org): check the connection and proxy settings."
    fi
}

# build_once: run `docker compose build`, keeping its output in BUILD_LOG for
# build_images. Returns the build's exit code.
build_once() {
    BUILD_LOG=$(mktemp)
    set +e
    docker compose build 2>&1 | tee "$BUILD_LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    return $rc
}

build_images() {
    info "The first build takes 15-40 minutes: llama.cpp is compiled for your processor and GPU."
    if [[ "$BACKEND" == cuda ]]; then
        info "NVIDIA: longer, up to an hour on a 4-core CPU. It also downloads the CUDA toolkit" \
            "(~4 GB, used only for compiling); llama.cpp is compiled for compute capability" \
            "${CUDA_ARCHS:-75-120 (all RTX; several times longer)}."
    fi
    info "Later runs (after git pull) take a few minutes."
    if build_once; then
        rm -f "$BUILD_LOG"
        return
    fi
    if grep -q 'exit code: 137' "$BUILD_LOG"; then
        rm -f "$BUILD_LOG"
        warn "The build ran out of memory: the compiler needed more RAM than the machine has."
        if [[ -z "$(env_value BUILD_JOBS)" ]]; then
            set_env BUILD_JOBS 2
            info "The number of parallel compile jobs is lowered (BUILD_JOBS=2) and the build is retried."
        else
            info "BUILD_JOBS=$(env_value BUILD_JOBS) is already set; the build is retried with it."
        fi
        if build_once; then
            rm -f "$BUILD_LOG"
            return
        fi
        rm -f "$BUILD_LOG"
        if [[ "$(env_value BUILD_JOBS)" == "1" ]]; then
            die "The build ran out of memory again even with BUILD_JOBS=1." \
                "Close other programs that use memory, then run ./setup.sh again."
        fi
        die "The build ran out of memory again. Close other programs that use memory," \
            "or set BUILD_JOBS=1 in .env, then run ./setup.sh again."
    fi
    rm -f "$BUILD_LOG"
    die "The build failed (see above). Fix it and run ./setup.sh again: finished steps are cached."
}

# ---- 6. Models and GPU check -----------------------------------------------

download_models() {
    docker compose run --rm prefetch-models python scripts/prefetch_models.py "$@" ||
        die "Downloading the models failed (see above). Run ./setup.sh again: finished files are kept."
}

check_gpu() {
    local log rc
    log=$(mktemp)
    info "Checking that the job worker uses the GPU..."
    set +e
    docker compose run --rm --no-deps "worker-$BACKEND" \
        python scripts/check_gpu.py --backend "$BACKEND" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    # check_gpu.py marks warnings with "!"; this one is about the LLM.
    grep -q '^ *! LLM memory' "$log" && LOW_GPU_MEMORY=true
    rm -f "$log"
    ((rc == 0)) || die "The job worker does not use the GPU as expected (see above)."
}

# ---- The LLM ----------------------------------------------------------------
# The LLM chooses the clips, so it is required:
# Qwen3-14B on this computer, or an API provider. scripts/llm_provider.py,
# run in the job worker container, reads and writes the choice in the user
# settings (data/settings.yaml, the file of the Settings page); the API key
# goes into .env.

llm_provider() {
    docker compose run --rm --no-deps "worker-$BACKEND" python scripts/llm_provider.py "$@"
}

# llm_status: what the jobs would use now, as LLM_PROVIDER, LLM_READY, ...
llm_status() {
    local out line
    out=$(llm_provider status) || die "Cannot read the LLM settings (see above)."
    while IFS= read -r line; do
        case "$line" in
            LLM_ENABLED=*) LLM_ENABLED=${line#*=} ;;
            LLM_PROVIDER=*) LLM_PROVIDER=${line#*=} ;;
            LLM_MODEL=*) LLM_MODEL=${line#*=} ;;
            LLM_API_BASE=*) LLM_API_BASE=${line#*=} ;;
            LLM_READY=*) LLM_READY=${line#*=} ;;
        esac
    done <<<"$out"
}

key_variable() {
    case "$1" in
        openai) echo OPENAI_API_KEY ;;
        anthropic) echo ANTHROPIC_API_KEY ;;
    esac
}

default_api_model() {
    case "$1" in
        openai) echo "$DEFAULT_OPENAI_MODEL" ;;
        anthropic) echo "$DEFAULT_ANTHROPIC_MODEL" ;;
    esac
}

# ask_required QUESTION: like ask, but without a default and not empty.
ask_required() {
    local answer=""
    while [[ -z "$answer" ]]; do
        read -rp "$1: " answer
    done
    echo "$answer"
}

use_local_llm() {
    if [[ "$LLM_PROVIDER" != llama_cpp || "$LLM_ENABLED" != yes ]]; then
        llm_provider use-local || die "Cannot change the LLM settings (see above)."
    fi
    info "Downloading the LLM Qwen3-14B (9 GB; a file that is already there is only checked)..."
    download_models --skip-whisper
}

# check_api_llm: one short request from the job worker, as a job sends it.
check_api_llm() {
    local out
    info "Checking the API provider with one short request..."
    if out=$(llm_provider check); then
        info "LLM: $out."
        return 0
    fi
    error "The API provider does not answer as expected (see above)."
    return 1
}

# use_api_llm PROVIDER MODEL [API_BASE]: save the choice and check it. A
# failed check takes the choice back, so the next ./setup.sh asks again.
use_api_llm() {
    local args=(--provider "$1" --model "$2")
    [[ -n "${3:-}" ]] && args+=(--api-base "$3")
    llm_provider use-api "${args[@]}" || die "Cannot change the LLM settings (see above)."
    check_api_llm && return 0
    llm_provider use-local >/dev/null || true
    return 1
}

# setup_api_llm: ask for the provider, the key and the model. Returns 1 when
# the check fails (the menu is shown again).
setup_api_llm() {
    local kind provider=openai base="" variable key current model
    echo "  1) OpenAI"
    echo "  2) Anthropic (Claude)"
    echo "  3) Another service with an OpenAI-compatible API (DeepSeek, OpenRouter, your own server)"
    while true; do
        kind=$(ask "Provider" 1)
        [[ "$kind" =~ ^[123]$ ]] && break
        warn "Please answer 1, 2 or 3."
    done
    [[ "$kind" == 2 ]] && provider=anthropic
    [[ "$kind" == 3 ]] && base=$(ask_required "Address of its API (for example https://api.deepseek.com/v1)")
    variable=$(key_variable "$provider")
    current=$(env_value "$variable")
    while true; do
        read -rsp "API key ($variable in .env; not shown${current:+, Enter keeps the current one}): " key
        echo
        key=${key:-$current}
        if [[ -z "$key" ]]; then
            warn "The key is needed."
        elif [[ "$key" == *"'"* ]]; then
            warn "A key with the ' character cannot be written to .env."
        else
            break
        fi
    done
    set_env "$variable" "$key"
    chmod 600 "$ENV_FILE"
    if [[ "$kind" == 3 ]]; then
        model=$(ask_required "Model name (see the service's documentation)")
    else
        model=$(ask "Model" "$(default_api_model "$provider")")
    fi
    if use_api_llm "$provider" "$model" "$base"; then
        info "Jobs send the transcripts to $provider; the key is in .env," \
            "the model can be changed on the Settings page."
        return 0
    fi
    if [[ "$(env_value SETUP_NETWORK)" == host ]]; then
        warn "Docker's network has no internet access here (see step 5), so the job worker cannot" \
            "reach API providers. Fix the firewall (docs/INSTALL.md, 'Если что-то не так') or download Qwen3-14B."
    else
        warn "Check the key, the model name, and that this computer reaches the service" \
            "(the job worker sends the requests)."
    fi
    return 1
}

# choose_llm_without_questions: --yes (or no terminal). An API key in .env
# chooses that provider; otherwise Qwen3-14B is downloaded.
choose_llm_without_questions() {
    local provider
    for provider in openai anthropic; do
        [[ -n "$(env_value "$(key_variable "$provider")")" ]] || continue
        info "LLM: $(key_variable "$provider") is set in .env, so jobs use the $provider API" \
            "(model $(default_api_model "$provider"))."
        use_api_llm "$provider" "$(default_api_model "$provider")" ||
            die "Fix the key in .env (or remove it to download Qwen3-14B instead), then run ./setup.sh again."
        return
    done
    if [[ "$LOW_GPU_MEMORY" == true ]]; then
        warn "Downloading it anyway. To use an API provider instead, run ./setup.sh in a terminal," \
            "or put OPENAI_API_KEY or ANTHROPIC_API_KEY into .env and run it again."
    fi
    use_local_llm
}

choose_llm() {
    local choice default=1
    llm_status
    if [[ "$LLM_READY" == yes ]]; then
        case "$LLM_PROVIDER" in
            llama_cpp)
                info "LLM: Qwen3-14B is already downloaded; checking the file..."
                download_models --skip-whisper ;;
            openai | anthropic)
                info "LLM: the $LLM_PROVIDER API, model $LLM_MODEL${LLM_API_BASE:+ at $LLM_API_BASE}" \
                    "(the Settings page changes it; the key is in .env)." ;;
            *)
                info "LLM: $LLM_PROVIDER, model $LLM_MODEL (as configured)." ;;
        esac
        return
    fi
    if [[ "$LLM_ENABLED" != yes ]]; then
        warn "The LLM is turned off in the settings: jobs would choose clips by simple heuristics, much worse."
    elif [[ "$LLM_PROVIDER" == openai || "$LLM_PROVIDER" == anthropic ]]; then
        warn "The settings choose the $LLM_PROVIDER API, but .env has no $(key_variable "$LLM_PROVIDER")."
    fi
    if [[ "$LOW_GPU_MEMORY" == true ]]; then
        warn "Qwen3-14B (9 GB) needs a GPU with about 10 GB of memory. Here it would run" \
            "(partly) on the CPU: hours per video instead of minutes. An API provider is faster."
        default=2
    fi
    if ! interactive; then
        choose_llm_without_questions
        return
    fi
    while true; do
        echo "The clips are chosen by an LLM, so one is required:"
        echo "  1) Download Qwen3-14B (9 GB) and run it on this computer; the videos stay here"
        echo "  2) Use an API provider (OpenAI, Anthropic or an OpenAI-compatible service):"
        echo "     paid per use, and the transcripts of the videos are sent to it"
        echo "  3) Stop here and decide later"
        choice=$(ask "Your choice" "$default")
        case "$choice" in
            1)
                use_local_llm
                return ;;
            2)
                setup_api_llm && return
                default=1 ;;
            3)
                die "Stopped: the LLM is required. Run ./setup.sh again to download Qwen3-14B" \
                    "or to set up an API provider (the images are built; that is quick)." ;;
            *)
                warn "Please answer 1, 2 or 3." ;;
        esac
    done
}

models_and_gpu_check() {
    if [[ "$NO_MODELS" == true ]]; then
        info "Skipping the model download (--no-models)."
    else
        info "Downloading speech recognition, keyword and face detection models (~2 GB)..."
        download_models --skip-llm
    fi
    check_gpu
    [[ "$NO_MODELS" == true ]] || choose_llm
}

# ---- 7. Start ---------------------------------------------------------------

start_stack() {
    local args=()
    [[ "$ASSUME_YES" == true ]] && args=(--yes)
    if [[ "$NO_START" == true ]] || ! confirm "Start the web UI now?" Y; then
        info "Start it later with: $(stack_command up)"
        return
    fi
    "$ROOT/scripts/stack.sh" up "${args[@]}"
}

stack_command() {
    if command -v just >/dev/null 2>&1; then
        echo "just $1"
    else
        echo "scripts/stack.sh $1"
    fi
}

print_summary() {
    echo
    info "Installation complete."
    echo "  Start:   $(stack_command up)"
    echo "  Stop:    $(stack_command stop)"
    echo "  Status:  $(stack_command status)"
    echo "  Web UI:  $(web_url)"
    echo "  Your data stays in data/, jobs/, uploads/ and $MODELS_CHOICE (see docs/INSTALL.md)."
    if ! command -v just >/dev/null 2>&1; then
        echo "  Tip: install just for shorter commands (apt/pacman/dnf install just)."
    fi
}

main() {
    parse_args "$@"
    if [[ $EUID -eq 0 && -n "${SUDO_UID:-}" ]]; then
        die "Run ./setup.sh as your user, without sudo: the files would belong to root." \
            "If Docker needs sudo, add your user to the docker group: sudo usermod -aG docker \$USER"
    fi
    echo "AI Video Clips Pipeline: installation with Docker"

    step "1/7 Docker"
    check_tools
    step "2/7 GPU"
    choose_backend
    step "3/7 Settings (.env)"
    write_env
    step "4/7 Directories and disk space"
    find_published_images
    make_dirs
    check_disk
    step "5/7 Images"
    # Also for downloaded images: the models are downloaded from a container.
    check_network
    if [[ -n "$PUBLISHED_BACKEND" ]]; then
        pull_images
    else
        build_images
    fi
    step "6/7 Models and GPU check"
    models_and_gpu_check
    step "7/7 Start"
    start_stack
    print_summary
}

main "$@"
