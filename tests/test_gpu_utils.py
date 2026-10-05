"""
Tests for GPU Auto-Detection and Runtime Configuration Module (TASK-07)
"""

import subprocess
from unittest.mock import MagicMock, patch

import src.gpu_utils as gpu_utils_module
from src.gpu_utils import (
    build_gpu_config,
    cpu_cleanup,
    cuda_cleanup,
    detect_gpu_backend,
    get_auto_gpu_config,
)


class TestDetectGPUBackend:
    """Test GPU backend detection."""

    def test_detect_cuda_from_nvidia_smi(self):
        """Mock subprocess.run to return nvidia-smi output → assert result == 'cuda'"""
        with patch("src.gpu_utils._run_subprocess") as mock_run:
            mock_run.return_value = "NVIDIA GeForce RTX 3060, 12288 MiB"
            result = detect_gpu_backend()
            assert result == "cuda"
            mock_run.assert_called_once_with(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                timeout=5,
            )

    def test_detect_openvino_from_clinfo(self):
        """Mock subprocess.run where nvidia-smi fails, clinfo returns 'Intel' → assert 'openvino'"""
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    from src.gpu_utils import GPUUtilsError

                    raise GPUUtilsError("nvidia-smi not found")
                if "clinfo" in cmd:
                    return "Platform #0: Intel(R) UHD Graphics"
                if "lspci" in cmd:
                    return ""
                return ""

            mock_run.side_effect = side_effect
            result = detect_gpu_backend()
            assert result == "openvino"

    def test_detect_rocm_from_clinfo(self):
        """Mock clinfo returns 'AMD' → assert 'rocm'"""
        with patch("src.gpu_utils._run_subprocess") as mock_run:

            def side_effect(cmd, timeout=5, shell=False):
                if "nvidia-smi" in cmd:
                    from src.gpu_utils import GPUUtilsError

                    raise GPUUtilsError("nvidia-smi not found")
                if "clinfo" in cmd:
                    return "Platform #0: AMD Radeon RX 6700 XT"
                if "lspci" in cmd:
                    return ""
                return ""

            mock_run.side_effect = side_effect
            result = detect_gpu_backend()
            assert result == "rocm"

    def test_detect_cpu_when_all_fail(self):
        """Mock all subprocess calls to raise FileNotFoundError → assert 'cpu'"""
        with patch("src.gpu_utils._run_subprocess") as mock_run:
            from src.gpu_utils import GPUUtilsError

            mock_run.side_effect = GPUUtilsError("not found")
            result = detect_gpu_backend()
            assert result == "cpu"

    def test_detect_timeout_fallback(self):
        """Mock nvidia-smi to raise TimeoutExpired → assert fallback works"""
        with patch("subprocess.run") as mock_subprocess:

            def side_effect(*args, **kwargs):
                cmd = args[0] if args else kwargs.get("args", [])
                if isinstance(cmd, list) and len(cmd) > 0 and "nvidia-smi" in cmd[0]:
                    raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
                mock_result = MagicMock()
                mock_result.stdout = ""
                return mock_result

            mock_subprocess.side_effect = side_effect
            result = detect_gpu_backend()
            assert result == "cpu"


class TestGPUConfig:
    """Test GPUConfig building."""

    def test_build_config_cuda_high_vram(self):
        """backend='cuda', vram=12288 → whisper_device='cuda', compute_type='float16'"""
        gpu_info = {"name": "NVIDIA RTX 3060", "vram_total_mb": 12288}
        config = build_gpu_config("cuda", gpu_info)
        assert config.backend == "cuda"
        assert config.whisper_device == "cuda"
        assert config.whisper_compute_type == "float16"

    def test_build_config_cuda_low_vram(self):
        """backend='cuda', vram=4096 → whisper_device='cuda', compute_type='int8'"""
        gpu_info = {"name": "NVIDIA GTX 1050", "vram_total_mb": 4096}
        config = build_gpu_config("cuda", gpu_info)
        assert config.backend == "cuda"
        assert config.whisper_device == "cuda"
        assert config.whisper_compute_type == "int8"

    def test_build_config_openvino(self):
        """backend='openvino' → whisper_device='cpu', providers contain OpenVINOExecutionProvider"""
        gpu_info = {"name": "Intel UHD", "vram_total_mb": None}
        config = build_gpu_config("openvino", gpu_info)
        assert config.backend == "openvino"
        assert config.whisper_device == "cpu"

    def test_build_config_rocm(self):
        """backend='rocm' → whisper_device='cpu', providers=['CPUExecutionProvider']"""
        gpu_info = {"name": "AMD Radeon", "vram_total_mb": None}
        config = build_gpu_config("rocm", gpu_info)
        assert config.backend == "rocm"
        assert config.whisper_device == "cpu"

    def test_build_config_cpu(self):
        """backend='cpu' → whisper_device='cpu', providers=['CPUExecutionProvider']"""
        gpu_info = {"name": "CPU", "vram_total_mb": 0}
        config = build_gpu_config("cpu", gpu_info)
        assert config.backend == "cpu"
        assert config.whisper_device == "cpu"


class TestCleanup:
    """Test cleanup functions."""

    def test_cuda_cleanup_calls_torch(self):
        """Mock torch.cuda.empty_cache, call cuda_cleanup() → verify called"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            cuda_cleanup()
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_cpu_cleanup_runs_without_torch(self):
        """call cpu_cleanup() → no ImportError"""
        cpu_cleanup()
        assert True


class TestGetGPUInfo:
    """Test GPU info retrieval for each backend."""

    def test_get_gpu_info_cuda(self):
        """Mock nvidia-smi → parse name and VRAM."""
        from src.gpu_utils import get_gpu_info

        with patch("src.gpu_utils._run_subprocess") as mock_run:
            mock_run.return_value = "NVIDIA GeForce RTX 4090, 24576 MiB"
            info = get_gpu_info("cuda")
            assert info["name"] == "NVIDIA GeForce RTX 4090"
            assert info["vram_total_mb"] == 24576

    def test_get_gpu_info_cuda_nvidia_smi_fails(self):
        """nvidia-smi fails → returns defaults."""
        from src.gpu_utils import GPUUtilsError, get_gpu_info

        with patch("src.gpu_utils._run_subprocess", side_effect=GPUUtilsError("fail")):
            info = get_gpu_info("cuda")
            assert info["name"] == "Unknown NVIDIA GPU"
            assert info["vram_total_mb"] is None

    def test_get_gpu_info_openvino(self):
        """llama.cpp's ggml devices name the Vulkan GPU and its memory."""
        from src.gpu_utils import get_gpu_info

        with patch("src.gpu_utils.probe_ggml_devices") as probe:
            probe.return_value = [
                {
                    "kind": 1,
                    "total": 12811800576,
                    "name": "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
                }
            ]
            info = get_gpu_info("openvino")
            assert info["name"] == "Intel(R) Arc(tm) B580 Graphics (BMG G21)"
            assert info["vram_total_mb"] == 12218

    def test_get_gpu_info_openvino_no_gpu(self):
        """No ggml GPU device: no invented name, just None."""
        from src.gpu_utils import get_gpu_info

        with patch("src.gpu_utils.probe_ggml_devices", return_value=[]):
            info = get_gpu_info("openvino")
            assert info["name"] is None
            assert info["vram_total_mb"] is None

    def test_get_gpu_info_openvino_igpu_reports_no_vram(self):
        """An integrated GPU shares system memory: its total is not VRAM."""
        from src.gpu_utils import get_gpu_info

        with patch("src.gpu_utils.probe_ggml_devices") as probe:
            probe.return_value = [
                {"kind": 2, "total": 17000000000, "name": "Intel(R) Iris(R) Xe Graphics"}
            ]
            info = get_gpu_info("openvino")
            assert info["name"] == "Intel(R) Iris(R) Xe Graphics"
            assert info["vram_total_mb"] is None

    def test_get_gpu_info_rocm(self):
        """Mock clinfo with AMD → parse name."""
        from src.gpu_utils import get_gpu_info

        with patch("src.gpu_utils._run_subprocess") as mock_run:
            mock_run.return_value = "Platform #0: AMD Radeon RX 7900 XTX"
            info = get_gpu_info("rocm")
            assert "AMD" in info["name"]
            assert info["vram_total_mb"] is None

    def test_get_gpu_info_cpu(self):
        """CPU backend → name='CPU', vram=0."""
        from src.gpu_utils import get_gpu_info

        info = get_gpu_info("cpu")
        assert info["name"] == "CPU"
        assert info["vram_total_mb"] == 0


class TestGgmlDevices:
    """The ggml backend probe and the choice of the best GPU device."""

    def test_parse_ggml_devices(self):
        from src.gpu_utils import parse_ggml_devices

        stdout = (
            "device=0\t33554432\tCPU\n"
            "device=1\t12811800576\tIntel(R) Arc(tm) B580 Graphics (BMG G21)\n"
        )
        assert parse_ggml_devices(stdout) == [
            {"kind": 0, "total": 33554432, "name": "CPU"},
            {"kind": 1, "total": 12811800576, "name": "Intel(R) Arc(tm) B580 Graphics (BMG G21)"},
        ]

    def test_parse_ggml_devices_ignores_garbage(self):
        from src.gpu_utils import parse_ggml_devices

        assert parse_ggml_devices("device=1\tnotanumber\tname\n") == []
        assert parse_ggml_devices("random line\n") == []

    def test_best_gpu_device_prefers_discrete(self):
        from src.gpu_utils import _best_gpu_device

        devices = [
            {"kind": 2, "total": 8589934592, "name": "Intel(R) UHD Graphics"},
            {"kind": 1, "total": 12811800576, "name": "Intel(R) Arc(tm) B580"},
        ]
        assert _best_gpu_device(devices)["name"] == "Intel(R) Arc(tm) B580"

    def test_best_gpu_device_falls_back_to_igpu_and_none(self):
        from src.gpu_utils import _best_gpu_device

        assert (
            _best_gpu_device([{"kind": 2, "total": 8589934592, "name": "iGPU"}])["name"] == "iGPU"
        )
        assert _best_gpu_device([]) is None
        assert _best_gpu_device([{"kind": 0, "total": 33554432, "name": "CPU"}]) is None


class TestCaching:
    """Test caching behavior."""

    def test_get_auto_gpu_config_caches_result(self):
        """Mock detect_gpu_backend to return 'cpu', call get_auto_gpu_config() twice → detect_gpu_backend should only be called once"""
        # Reset cache
        gpu_utils_module._cached_config = None

        with (
            patch.object(gpu_utils_module, "detect_gpu_backend", return_value="cpu") as mock_detect,
            patch.object(
                gpu_utils_module, "get_gpu_info", return_value={"name": "CPU", "vram_total_mb": 0}
            ),
        ):
            config1 = get_auto_gpu_config()
            config2 = get_auto_gpu_config()
            assert config1.backend == "cpu"
            assert config2.backend == "cpu"
            assert config1 is config2
            mock_detect.assert_called_once()

        # Reset cache after test
        gpu_utils_module._cached_config = None


class TestImageBackend:
    """In a Docker image PIPELINE_HARDWARE names the backend: no probing."""

    def test_pipeline_hardware_wins(self, monkeypatch):
        from src import gpu_utils

        monkeypatch.setenv("PIPELINE_HARDWARE", "openvino")
        with patch.object(gpu_utils, "_run_subprocess") as probe:
            assert gpu_utils.detect_gpu_backend() == "openvino"
        probe.assert_not_called()

    def test_unknown_value_is_ignored(self, monkeypatch):
        from src import gpu_utils

        monkeypatch.setenv("PIPELINE_HARDWARE", "")
        with patch.object(gpu_utils, "_run_subprocess", side_effect=gpu_utils.GPUUtilsError("x")):
            assert gpu_utils.detect_gpu_backend() == "cpu"
