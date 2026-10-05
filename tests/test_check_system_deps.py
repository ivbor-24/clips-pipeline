"""Tests for scripts/check_system_deps.sh.

The script runs with PATH limited to a directory of fake commands, so each
test decides exactly what the "system" has.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_system_deps.sh"
BASH = shutil.which("bash")

BASE_TOOLS = ["git", "curl", "cmake", "make", "ffmpeg", "ffprobe", "just"]

# A fake compiler: fails when the source includes a header or links a library
# listed in FAKE_CXX_MISSING (space separated), succeeds otherwise. It uses only
# bash builtins: PATH holds nothing but the fake commands.
FAKE_CXX = """#!/bin/bash
IFS= read -r -d '' src
for item in $FAKE_CXX_MISSING; do
    case "$src $*" in *"$item"*) exit 1 ;; esac
done
exit 0
"""


def _make_bin(tmp_path: Path, tools, cxx_missing=None, fc_list=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in tools:
        path = bin_dir / tool
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    if cxx_missing is not None:
        cxx = bin_dir / "c++"
        cxx.write_text(FAKE_CXX)
        cxx.chmod(0o755)
    if fc_list is not None:
        fc = bin_dir / "fc-list"
        # Like the real fc-list: exit 0 either way, a line per matching font.
        found = "echo '/usr/share/fonts/NimbusMonoPS-Regular.otf: Nimbus Mono PS:style=Regular'\n"
        fc.write_text("#!/bin/sh\n" + (found if fc_list == 0 else "") + "exit 0\n")
        fc.chmod(0o755)
    return bin_dir


def _run(tmp_path, backend, tools, cxx_missing="", pm="apt", icd=True, fc_list=None):
    bin_dir = _make_bin(tmp_path, tools, cxx_missing, fc_list)
    icd_dir = tmp_path / "icd.d"
    icd_dir.mkdir()
    if icd:
        (icd_dir / "intel_icd.json").write_text("{}")
    env = {
        "PATH": str(bin_dir),
        "CHECK_PKG_MANAGER": pm,
        "CHECK_VULKAN_ICD_DIRS": str(icd_dir),
        "FAKE_CXX_MISSING": cxx_missing,
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    return subprocess.run(
        [BASH, str(SCRIPT), backend], capture_output=True, text=True, env=env, timeout=30
    )


VULKAN_TOOLS = BASE_TOOLS + ["glslc"]


class TestCheckSystemDeps:
    def test_all_present(self, tmp_path):
        result = _run(tmp_path, "openvino", VULKAN_TOOLS)
        assert result.returncode == 0, result.stdout
        assert "All build requirements for 'openvino' are present" in result.stdout

    def test_usage_error(self, tmp_path):
        result = _run(tmp_path, "tpu", BASE_TOOLS)
        assert result.returncode == 2

    def test_cpu_needs_no_vulkan(self, tmp_path):
        result = _run(tmp_path, "cpu", BASE_TOOLS, cxx_missing="vulkan spirv", icd=False)
        assert result.returncode == 0, result.stdout

    @pytest.mark.parametrize(
        "pm, command",
        [
            ("apt", "sudo apt-get install -y glslc spirv-headers mesa-vulkan-drivers"),
            ("pacman", "sudo pacman -S --needed shaderc spirv-headers vulkan-intel"),
            ("dnf", "sudo dnf install -y glslc spirv-headers-devel mesa-vulkan-drivers"),
        ],
    )
    def test_missing_vulkan_pieces_give_one_command(self, tmp_path, pm, command):
        result = _run(tmp_path, "openvino", BASE_TOOLS, cxx_missing="spirv", pm=pm, icd=False)
        assert result.returncode == 1
        assert "glslc (Vulkan shader compiler)" in result.stdout
        assert "SPIR-V headers" in result.stdout
        assert "Vulkan driver for the GPU" in result.stdout
        assert command in result.stdout
        assert "./install.sh cpu" in result.stdout

    def test_one_package_for_two_items(self, tmp_path):
        # vulkan.hpp and libvulkan both come from libvulkan-dev on apt.
        result = _run(tmp_path, "openvino", VULKAN_TOOLS, cxx_missing="vulkan", pm="apt")
        assert result.returncode == 1
        assert "sudo apt-get install -y libvulkan-dev\n" in result.stdout

    def test_missing_compiler_and_tools(self, tmp_path):
        result = _run(tmp_path, "cpu", ["git", "curl", "ffmpeg", "ffprobe"], pm="pacman")
        assert result.returncode == 1
        assert "sudo pacman -S --needed cmake base-devel just" in result.stdout
        assert "./install.sh cpu" not in result.stdout.split("again.")[1]

    def test_vendor_driver_is_a_hint(self, tmp_path):
        result = _run(tmp_path, "cuda", BASE_TOOLS, pm="apt")
        assert result.returncode == 1
        assert "sudo apt-get install -y nvidia-cuda-toolkit" in result.stdout
        assert "GPU vendor's repository" in result.stdout
        assert "NVIDIA driver (nvidia-smi)" in result.stdout

    def test_unknown_package_manager_lists_items(self, tmp_path):
        result = _run(tmp_path, "openvino", BASE_TOOLS, pm="unknown")
        assert result.returncode == 1
        assert "your distribution's package manager" in result.stdout
        assert "sudo" not in result.stdout

    def test_missing_subtitle_font_is_a_warning(self, tmp_path):
        result = _run(tmp_path, "cpu", BASE_TOOLS, fc_list=1, pm="apt")
        assert result.returncode == 0, result.stdout
        assert "Nimbus Mono PS" in result.stdout
        assert "sudo apt-get install -y fonts-urw-base35" in result.stdout

    @pytest.mark.parametrize(
        "pm, pkg",
        [("apt", "fonts-urw-base35"), ("pacman", "gsfonts"), ("dnf", "urw-base35-fonts")],
    )
    def test_missing_subtitle_font_package_per_manager(self, tmp_path, pm, pkg):
        result = _run(tmp_path, "cpu", BASE_TOOLS, fc_list=1, pm=pm)
        assert result.returncode == 0, result.stdout
        assert pkg in result.stdout

    def test_subtitle_font_present_is_silent(self, tmp_path):
        result = _run(tmp_path, "cpu", BASE_TOOLS, fc_list=0, pm="apt")
        assert result.returncode == 0, result.stdout
        assert "Nimbus Mono PS" not in result.stdout

    def test_no_fc_list_is_silent(self, tmp_path):
        result = _run(tmp_path, "cpu", BASE_TOOLS)
        assert result.returncode == 0, result.stdout
        assert "Nimbus Mono PS" not in result.stdout
