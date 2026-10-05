"""
TASK-P1-06: GPU Backend Integration Tests

Integration tests validating GPU backend detection and GPUConfig
correctness across cpu/cuda/openvino/rocm.

Uses mocks to simulate hardware environments. Set GPU_TEST_BACKEND
env var to run only tests for a specific backend (cpu, cuda, openvino).
"""

import os
from unittest.mock import MagicMock, patch

import pytest

import src.gpu_utils as gpu_utils_module
from src.gpu_utils import (
    GPUUtilsError,
    build_gpu_config,
    detect_gpu_backend,
    get_auto_gpu_config,
    get_gpu_info,
)

TARGET_BACKEND = os.environ.get("GPU_TEST_BACKEND", "").lower()


def _should_run(backend: str) -> bool:
    if not TARGET_BACKEND:
        return True
    return TARGET_BACKEND == backend


@pytest.fixture(autouse=True)
def reset_gpu_cache():
    gpu_utils_module._cached_config = None
    yield
    gpu_utils_module._cached_config = None


class TestCPUBackend:
    @pytest.fixture(autouse=True)
    def skip_if_not_target(self):
        if not _should_run("cpu"):
            pytest.skip("GPU_TEST_BACKEND set to non-cpu backend")

    def test_detect_cpu_when_no_gpu_tools(self):
        with patch("src.gpu_utils._run_subprocess", side_effect=GPUUtilsError("not found")):
            assert detect_gpu_backend() == "cpu"

    def test_gpu_info_cpu(self):
        info = get_gpu_info("cpu")
        assert info["name"] == "CPU"
        assert info["vram_total_mb"] == 0

    def test_build_config_cpu(self):
        info = {"name": "CPU", "vram_total_mb": 0}
        config = build_gpu_config("cpu", info)
        assert config.backend == "cpu"
        assert config.whisper_device == "cpu"
        assert config.whisper_compute_type == "int8"
        assert config.vram_total_mb == 0
        assert config.gpu_name == "CPU"

    def test_auto_gpu_config_cpu(self):
        with (
            patch.object(gpu_utils_module, "detect_gpu_backend", return_value="cpu"),
            patch.object(
                gpu_utils_module, "get_gpu_info", return_value={"name": "CPU", "vram_total_mb": 0}
            ),
        ):
            config = get_auto_gpu_config()
            assert config.backend == "cpu"


class TestCUDABackend:
    @pytest.fixture(autouse=True)
    def skip_if_not_target(self):
        if not _should_run("cuda"):
            pytest.skip("GPU_TEST_BACKEND set to non-cuda backend")

    def test_detect_cuda_via_nvidia_smi(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:
            mock_run.return_value = "NVIDIA GeForce RTX 3060, 12288 MiB"
            assert detect_gpu_backend() == "cuda"
            mock_run.assert_called_once()

    def test_gpu_info_cuda_parses_name_and_vram(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:
            mock_run.return_value = "NVIDIA GeForce RTX 3060, 12288 MiB"
            info = get_gpu_info("cuda")
            assert info["name"] == "NVIDIA GeForce RTX 3060"
            assert info["vram_total_mb"] == 12288

    def test_gpu_info_cuda_handles_missing_vram(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:
            mock_run.return_value = "NVIDIA GeForce RTX 3060"
            info = get_gpu_info("cuda")
            assert info["name"] == "NVIDIA GeForce RTX 3060"
            assert info["vram_total_mb"] is None

    def test_build_config_cuda_high_vram(self):
        info = {"name": "NVIDIA RTX 4090", "vram_total_mb": 24576}
        config = build_gpu_config("cuda", info)
        assert config.backend == "cuda"
        assert config.whisper_device == "cuda"
        assert config.whisper_compute_type == "float16"

    def test_build_config_cuda_low_vram(self):
        info = {"name": "NVIDIA GTX 1050", "vram_total_mb": 4096}
        config = build_gpu_config("cuda", info)
        assert config.whisper_compute_type == "int8"
        assert config.whisper_device == "cuda"

    def test_build_config_cuda_boundary_vram(self):
        info = {"name": "NVIDIA RTX 3060", "vram_total_mb": 8192}
        config = build_gpu_config("cuda", info)
        assert config.whisper_compute_type == "int8"

    def test_auto_gpu_config_cuda(self):
        with (
            patch.object(gpu_utils_module, "detect_gpu_backend", return_value="cuda"),
            patch.object(
                gpu_utils_module,
                "get_gpu_info",
                return_value={"name": "NVIDIA RTX 3060", "vram_total_mb": 12288},
            ),
        ):
            config = get_auto_gpu_config()
            assert config.backend == "cuda"
            assert config.whisper_device == "cuda"
            assert config.whisper_compute_type == "float16"


class TestOpenVINOBackend:
    @pytest.fixture(autouse=True)
    def skip_if_not_target(self):
        if not _should_run("openvino"):
            pytest.skip("GPU_TEST_BACKEND set to non-openvino backend")

    def test_detect_openvino_via_clinfo(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    raise GPUUtilsError("not found")
                if "clinfo" in cmd:
                    return "Platform #0: Intel(R) OpenCL Graphics"
                return ""

            mock_run.side_effect = side_effect
            assert detect_gpu_backend() == "openvino"

    def test_gpu_info_openvino(self):
        with patch("src.gpu_utils.probe_ggml_devices") as probe:
            probe.return_value = [
                {"kind": 1, "total": 12811800576, "name": "Intel(R) Arc(tm) A770 Graphics"}
            ]
            info = get_gpu_info("openvino")
            assert "Intel" in info["name"]
            assert info["vram_total_mb"] == 12218

    def test_build_config_openvino(self):
        info = {"name": "Intel Arc A770", "vram_total_mb": None}
        config = build_gpu_config("openvino", info)
        assert config.backend == "openvino"
        assert config.whisper_device == "cpu"
        assert config.whisper_compute_type == "int8"

    def test_auto_gpu_config_openvino(self):
        with (
            patch.object(gpu_utils_module, "detect_gpu_backend", return_value="openvino"),
            patch.object(
                gpu_utils_module,
                "get_gpu_info",
                return_value={"name": "Intel Arc A770", "vram_total_mb": None},
            ),
        ):
            config = get_auto_gpu_config()
            assert config.backend == "openvino"
            assert config.whisper_device == "cpu"


class TestROCmBackend:
    @pytest.fixture(autouse=True)
    def skip_if_not_target(self):
        if not _should_run("rocm"):
            pytest.skip("GPU_TEST_BACKEND set to non-rocm backend (rocm not in CI matrix)")

    def test_detect_rocm_via_clinfo(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    raise GPUUtilsError("not found")
                if "clinfo" in cmd:
                    return "Platform #0: AMD Accelerated Parallel Processing"
                return ""

            mock_run.side_effect = side_effect
            assert detect_gpu_backend() == "rocm"

    def test_build_config_rocm(self):
        info = {"name": "AMD Radeon RX 7900 XTX", "vram_total_mb": None}
        config = build_gpu_config("rocm", info)
        assert config.backend == "rocm"
        assert config.whisper_device == "cpu"
        assert config.whisper_compute_type == "int8"


class TestBackendFallbackChain:
    def test_nvidia_smi_fails_clinfo_intel_succeeds(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    raise GPUUtilsError("not found")
                if "clinfo" in cmd:
                    return "Platform #0: Intel(R) UHD Graphics"
                return ""

            mock_run.side_effect = side_effect
            assert detect_gpu_backend() == "openvino"

    def test_nvidia_smi_fails_clinfo_amd_succeeds(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    raise GPUUtilsError("not found")
                if "clinfo" in cmd:
                    return "Platform #0: AMD Radeon"
                return ""

            mock_run.side_effect = side_effect
            assert detect_gpu_backend() == "rocm"

    def test_all_detection_fails_fallback_to_cpu(self):
        with patch("src.gpu_utils._run_subprocess", side_effect=GPUUtilsError("not found")):
            assert detect_gpu_backend() == "cpu"

    def test_lspci_nvidia_fallback(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    raise GPUUtilsError("not found")
                if "clinfo" in cmd:
                    raise GPUUtilsError("not found")
                if "lspci" in str(cmd):
                    return "01:00.0 VGA compatible controller: NVIDIA Corporation Device"
                return ""

            mock_run.side_effect = side_effect
            assert detect_gpu_backend() == "cuda"

    def test_lspci_intel_fallback(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    raise GPUUtilsError("not found")
                if "clinfo" in cmd:
                    raise GPUUtilsError("not found")
                if "lspci" in str(cmd):
                    return "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics"
                return ""

            mock_run.side_effect = side_effect
            assert detect_gpu_backend() == "openvino"

    def test_lspci_amd_fallback(self):
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    raise GPUUtilsError("not found")
                if "clinfo" in cmd:
                    raise GPUUtilsError("not found")
                if "lspci" in str(cmd):
                    return "03:00.0 VGA compatible controller: Advanced Micro Devices [AMD/ATI]"
                return ""

            mock_run.side_effect = side_effect
            assert detect_gpu_backend() == "rocm"


class TestCleanupFunctions:
    def test_cpu_cleanup_callable(self):
        info = {"name": "CPU", "vram_total_mb": 0}
        config = build_gpu_config("cpu", info)
        config.cleanup_fn()

    def test_openvino_cleanup_callable(self):
        info = {"name": "Intel Arc", "vram_total_mb": None}
        config = build_gpu_config("openvino", info)
        config.cleanup_fn()

    def test_cuda_cleanup_callable(self):
        info = {"name": "NVIDIA RTX 3060", "vram_total_mb": 12288}
        config = build_gpu_config("cuda", info)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            config.cleanup_fn()
            mock_torch.cuda.empty_cache.assert_called_once()


class TestGPUConfigImmutability:
    def test_config_is_frozen(self):
        info = {"name": "CPU", "vram_total_mb": 0}
        config = build_gpu_config("cpu", info)
        with pytest.raises(AttributeError):
            config.backend = "cuda"

    def test_config_caching_returns_same_instance(self):
        with (
            patch.object(gpu_utils_module, "detect_gpu_backend", return_value="cpu"),
            patch.object(
                gpu_utils_module, "get_gpu_info", return_value={"name": "CPU", "vram_total_mb": 0}
            ),
        ):
            c1 = get_auto_gpu_config()
            c2 = get_auto_gpu_config()
            assert c1 is c2
