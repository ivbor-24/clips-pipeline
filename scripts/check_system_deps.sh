#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# Check that the system has everything ./install.sh <backend> needs, before
# anything is compiled.
#
# It checks capabilities rather than package names: commands in PATH, headers
# the C++ compiler can find, libraries it can link. That works on any distro.
# For whatever is missing it prints one install command for apt, pacman or
# dnf, so a missing Vulkan header shows up in seconds instead of as a CMake
# error ten minutes into the llama.cpp build.
#
# Usage: check_system_deps.sh <cpu|cuda|openvino|rocm>
# Exit:  0 everything found, 1 something is missing, 2 usage error
#
# Environment (used by the tests):
#   CHECK_PKG_MANAGER     apt | pacman | dnf | unknown, instead of detecting it
#   CHECK_VULKAN_ICD_DIRS where to look for Vulkan driver manifests
#   CXX                   C++ compiler (default: c++)
# ============================================================================

BACKEND="${1:-}"
case "$BACKEND" in
    cpu | cuda | openvino | rocm) ;;
    *)
        echo "Usage: $0 <cpu|cuda|openvino|rocm>" >&2
        exit 2 ;;
esac

CXX="${CXX:-c++}"
ICD_DIRS="${CHECK_VULKAN_ICD_DIRS:-/usr/share/vulkan/icd.d /etc/vulkan/icd.d}"

detect_pkg_manager() {
    if [[ -n "${CHECK_PKG_MANAGER:-}" ]]; then
        echo "$CHECK_PKG_MANAGER"
    elif command -v apt-get >/dev/null 2>&1; then
        echo apt
    elif command -v pacman >/dev/null 2>&1; then
        echo pacman
    elif command -v dnf >/dev/null 2>&1; then
        echo dnf
    else
        echo unknown
    fi
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

compiles() {
    # $1: C++ source on one line; the rest: extra compiler arguments.
    local src=$1
    shift
    has_cmd "$CXX" && echo "$src" | "$CXX" -x c++ - -o /dev/null "$@" >/dev/null 2>&1
}

has_header() { compiles "#include <$1>
int main() { return 0; }"; }

has_vulkan_driver() {
    local dir
    for dir in $ICD_DIRS; do
        compgen -G "$dir/*.json" >/dev/null && return 0
    done
    return 1
}

# Human-readable name of a requirement.
describe() {
    case "$1" in
        git) echo "git" ;;
        curl) echo "curl" ;;
        cmake) echo "cmake" ;;
        compiler) echo "C/C++ compiler (c++)" ;;
        make) echo "make or ninja" ;;
        ffmpeg) echo "ffmpeg and ffprobe" ;;
        just) echo "just (command runner)" ;;
        glslc) echo "glslc (Vulkan shader compiler)" ;;
        vulkan-headers) echo "Vulkan headers (vulkan/vulkan.hpp)" ;;
        spirv-headers) echo "SPIR-V headers (spirv/unified1/spirv.hpp)" ;;
        vulkan-loader) echo "Vulkan loader library (libvulkan)" ;;
        vulkan-driver) echo "Vulkan driver for the GPU (no ICD manifest found)" ;;
        nvcc) echo "CUDA toolkit (nvcc)" ;;
        nvidia-driver) echo "NVIDIA driver (nvidia-smi)" ;;
        hipcc) echo "ROCm HIP compiler (hipcc)" ;;
    esac
}

# Package that provides a requirement, per package manager.
package_for() {
    local pm=$1 item=$2
    case "$pm:$item" in
        *:git | *:curl | *:cmake | *:just) echo "$item" ;;
        apt:compiler | apt:make) echo "build-essential" ;;
        apt:ffmpeg) echo "ffmpeg" ;;
        apt:glslc) echo "glslc" ;;
        apt:vulkan-headers | apt:vulkan-loader) echo "libvulkan-dev" ;;
        apt:spirv-headers) echo "spirv-headers" ;;
        apt:vulkan-driver) echo "mesa-vulkan-drivers" ;;
        apt:nvcc) echo "nvidia-cuda-toolkit" ;;
        pacman:compiler | pacman:make) echo "base-devel" ;;
        pacman:ffmpeg) echo "ffmpeg" ;;
        pacman:glslc) echo "shaderc" ;;
        pacman:vulkan-headers) echo "vulkan-headers" ;;
        pacman:spirv-headers) echo "spirv-headers" ;;
        pacman:vulkan-loader) echo "vulkan-icd-loader" ;;
        pacman:vulkan-driver) echo "vulkan-intel" ;;
        pacman:nvcc) echo "cuda" ;;
        pacman:nvidia-driver) echo "nvidia-utils" ;;
        pacman:hipcc) echo "rocm-hip-sdk" ;;
        dnf:compiler) echo "gcc-c++" ;;
        dnf:make) echo "make" ;;
        dnf:ffmpeg) echo "ffmpeg-free" ;;
        dnf:glslc) echo "glslc" ;;
        dnf:vulkan-headers) echo "vulkan-headers" ;;
        dnf:spirv-headers) echo "spirv-headers-devel" ;;
        dnf:vulkan-loader) echo "vulkan-loader-devel" ;;
        dnf:vulkan-driver) echo "mesa-vulkan-drivers" ;;
        dnf:hipcc) echo "rocm-hip-devel" ;;
        # The NVIDIA driver (and CUDA on Fedora) come from vendor repositories
        # that differ per distro and card, so they get a hint instead.
        *) echo "" ;;
    esac
}

install_command() {
    case "$1" in
        apt) echo "sudo apt-get install -y" ;;
        pacman) echo "sudo pacman -S --needed" ;;
        dnf) echo "sudo dnf install -y" ;;
    esac
}

# Package that provides the default subtitle font (Nimbus Mono PS).
subtitle_font_package() {
    case "$1" in
        apt) echo "fonts-urw-base35" ;;
        pacman) echo "gsfonts" ;;
        dnf) echo "urw-base35-fonts" ;;
    esac
}

# The default subtitle font (Nimbus Mono PS, preset "typewriter") is used for
# burned-in captions. It is not a build requirement: libass falls back to
# another monospace font, so a missing font is only a warning with the package
# to install. Skipped silently when fc-list is not available.
check_subtitle_font() {
    has_cmd fc-list || return 0
    # fc-list exits 0 with no match too: look at its output.
    [[ -n "$(fc-list "Nimbus Mono PS" 2>/dev/null)" ]] && return 0
    local pm="$(detect_pkg_manager)"
    local pkg="$(subtitle_font_package "$pm")"
    echo "Warning: the default subtitle font 'Nimbus Mono PS' is not visible to fontconfig."
    echo "  Rendering falls back to another monospace font."
    if [[ -n "$pkg" ]]; then
        echo "  For the default look, install it with: $(install_command "$pm") $pkg"
    fi
}

missing=()
need() { missing+=("$1"); }

# Everything: fetching sources, building llama.cpp / whisper.cpp, running.
has_cmd git || need git
has_cmd curl || need curl
has_cmd cmake || need cmake
compiles "int main() { return 0; }" || need compiler
has_cmd make || has_cmd ninja || need make
{ has_cmd ffmpeg && has_cmd ffprobe; } || need ffmpeg
has_cmd just || need just

case "$BACKEND" in
    openvino)
        # Intel Arc: llama.cpp and whisper.cpp are built with the Vulkan backend.
        has_cmd glslc || need glslc
        has_header vulkan/vulkan.hpp || need vulkan-headers
        has_header spirv/unified1/spirv.hpp || need spirv-headers
        compiles "int main() { return 0; }" -lvulkan || need vulkan-loader
        has_vulkan_driver || need vulkan-driver ;;
    cuda)
        { has_cmd nvcc || [[ -x /usr/local/cuda/bin/nvcc ]]; } || need nvcc
        has_cmd nvidia-smi || need nvidia-driver ;;
    rocm)
        { has_cmd hipcc || [[ -x /opt/rocm/bin/hipcc ]]; } || need hipcc ;;
esac

check_subtitle_font

if [[ ${#missing[@]} -eq 0 ]]; then
    echo "All build requirements for '$BACKEND' are present."
    exit 0
fi

PM="$(detect_pkg_manager)"
echo "Missing for ./install.sh $BACKEND:"
packages=()
hints=()
for item in "${missing[@]}"; do
    echo "  - $(describe "$item")"
    pkg="$(package_for "$PM" "$item")"
    if [[ -n "$pkg" ]]; then
        [[ " ${packages[*]} " == *" $pkg "* ]] || packages+=("$pkg")
    else
        hints+=("$(describe "$item")")
    fi
done

echo ""
if [[ ${#packages[@]} -gt 0 ]]; then
    echo "Install them with:"
    echo "  $(install_command "$PM") ${packages[*]}"
fi
if [[ ${#hints[@]} -gt 0 ]]; then
    if [[ "$PM" == "unknown" ]]; then
        echo "Install these with your distribution's package manager:"
    else
        echo "Install these from your GPU vendor's repository:"
    fi
    for hint in "${hints[@]}"; do
        echo "  - $hint"
    done
fi
echo ""
echo "Then run ./install.sh $BACKEND again."
if [[ "$BACKEND" != "cpu" ]]; then
    echo "To install without GPU acceleration instead: ./install.sh cpu"
fi
exit 1
