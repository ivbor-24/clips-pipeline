"""Tests for scripts/check_gpu.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_gpu.py"

VULKAN_STDERR = (
    "ggml_vulkan: Found 1 Vulkan devices:\n"
    "ggml_vulkan: 0 = Intel(R) Arc(tm) B580 Graphics (BMG G21) (Intel open-source Mesa driver)"
    " | uma: 0 | fp16: 1\n"
)
CUDA_STDERR = (
    "ggml_cuda_init: found 1 CUDA devices:\n"
    "  Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes\n"
)


@pytest.fixture
def check_gpu():
    spec = importlib.util.spec_from_file_location("check_gpu_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestParseLlamaDevices:
    def test_vulkan_device(self, check_gpu):
        assert check_gpu.parse_llama_devices("openvino", VULKAN_STDERR) == [
            "Intel(R) Arc(tm) B580 Graphics"
        ]

    def test_cuda_device(self, check_gpu):
        assert check_gpu.parse_llama_devices("cuda", CUDA_STDERR) == ["NVIDIA GeForce RTX 3060"]

    def test_device_of_another_backend_is_ignored(self, check_gpu):
        assert check_gpu.parse_llama_devices("cuda", VULKAN_STDERR) == []

    def test_cpu_has_no_devices(self, check_gpu):
        assert check_gpu.parse_llama_devices("cpu", VULKAN_STDERR) == []


VULKAN_LIBS = "libs=libllama.so,libggml-vulkan.so,libggml-cpu.so\n"
CPU_LIBS = "libs=libllama.so,libggml-cpu.so\n"


class TestCheckLlama:
    def test_gpu_build_with_device(self, check_gpu):
        stdout = VULKAN_LIBS + "gpu_offload=True"
        with patch.object(
            check_gpu.subprocess, "run", return_value=_completed(stdout, VULKAN_STDERR)
        ):
            result = check_gpu.check_llama("openvino")
        assert result.ok is True
        assert "Vulkan: Intel(R) Arc(tm) B580 Graphics" in result.detail

    def test_cpu_only_build_fails_on_gpu_backend(self, check_gpu):
        with patch.object(
            check_gpu.subprocess, "run", return_value=_completed(CPU_LIBS + "gpu_offload=False")
        ):
            result = check_gpu.check_llama("openvino")
        assert result.ok is False
        assert "without Vulkan" in result.detail
        assert "LLAMA_FORCE_REBUILD=1" in result.hint

    def test_gpu_build_without_driver_blames_the_driver(self, check_gpu):
        # A Vulkan build with no visible device reports gpu_offload=False too.
        with patch.object(
            check_gpu.subprocess, "run", return_value=_completed(VULKAN_LIBS + "gpu_offload=False")
        ):
            result = check_gpu.check_llama("openvino")
        assert result.ok is False
        assert "no Vulkan device" in result.detail
        assert "vulkaninfo" in result.hint
        assert "Rebuild" not in result.hint

    def test_build_for_another_backend(self, check_gpu):
        stdout = VULKAN_LIBS + "gpu_offload=True"
        with patch.object(
            check_gpu.subprocess, "run", return_value=_completed(stdout, VULKAN_STDERR)
        ):
            result = check_gpu.check_llama("cuda")
        assert result.ok is False
        assert "without CUDA" in result.detail

    def test_cpu_backend_is_fine(self, check_gpu):
        with patch.object(
            check_gpu.subprocess, "run", return_value=_completed("gpu_offload=False")
        ):
            result = check_gpu.check_llama("cpu")
        assert result.ok is True

    def test_import_failure(self, check_gpu):
        with patch.object(
            check_gpu.subprocess, "run", return_value=_completed(returncode=1, stderr="Error")
        ):
            result = check_gpu.check_llama("openvino")
        assert result.ok is False
        assert "does not import" in result.detail


class TestCheckWhisperCpp:
    def test_not_used_outside_intel_arc(self, check_gpu, tmp_path):
        result = check_gpu.check_whisper_cpp("cuda", tmp_path / "bin", tmp_path / "model")
        assert result.ok is None

    def test_missing_binary_fails(self, check_gpu, tmp_path):
        result = check_gpu.check_whisper_cpp("openvino", tmp_path / "bin", tmp_path / "model")
        assert result.ok is False
        assert "install_whisper_cpp.sh" in result.hint

    def test_missing_model_is_not_checked(self, check_gpu, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.touch()
        result = check_gpu.check_whisper_cpp("openvino", binary, tmp_path / "model.bin")
        assert result.ok is None
        assert "prefetch-models" in result.hint

    @pytest.mark.parametrize(
        "stderr, ok",
        [
            ("whisper_backend_init_gpu: using Vulkan0 backend\n", True),
            ("whisper_backend_init_gpu: no GPU found\n", False),
            ("", False),
        ],
    )
    def test_reads_backend_from_test_run(self, check_gpu, tmp_path, stderr, ok):
        binary = tmp_path / "whisper-cli"
        model = tmp_path / "model.bin"
        binary.touch()
        model.touch()
        with patch.object(check_gpu.subprocess, "run", return_value=_completed(stderr=stderr)):
            result = check_gpu.check_whisper_cpp("openvino", binary, model)
        assert result.ok is ok


class TestDetectBackend:
    """Without --backend: the GPU library in the llama.cpp build tells."""

    def detect(self, check_gpu, tmp_path, libs):
        lib = tmp_path / "lib"
        lib.mkdir()
        for name in libs:
            (lib / name).write_text("")
        with patch.object(check_gpu, "package_file", return_value=lib):
            return check_gpu.detect_backend()

    def test_vulkan_build_is_intel(self, check_gpu, tmp_path):
        assert self.detect(check_gpu, tmp_path, ["libggml-vulkan.so", "libllama.so"]) == "openvino"

    def test_cuda_build(self, check_gpu, tmp_path):
        assert self.detect(check_gpu, tmp_path, ["libggml-cuda.so"]) == "cuda"

    def test_cpu_build(self, check_gpu, tmp_path):
        assert self.detect(check_gpu, tmp_path, ["libggml-cpu.so"]) == "cpu"


DEVICE_LINES = (
    "libs=libggml-vulkan.so\ngpu_offload=True\n"
    "device=1\t12809404416\t9954131968\tIntel(R) Arc(tm) B580 Graphics (BMG G21)\n"
    "device=0\t16678649856\t16678649856\tIntel(R) Xeon(R) CPU E5-2630 v2\n"
)
GIB = 2**30


class TestParseDevices:
    def test_devices_with_memory(self, check_gpu):
        devices = check_gpu.parse_devices(DEVICE_LINES)
        assert [(d.kind, d.name) for d in devices] == [
            ("gpu", "Intel(R) Arc(tm) B580 Graphics (BMG G21)"),
            ("cpu", "Intel(R) Xeon(R) CPU E5-2630 v2"),
        ]
        assert devices[0].total == 12809404416
        assert devices[0].free == 9954131968

    def test_build_without_device_registry(self, check_gpu):
        assert check_gpu.parse_devices("libs=libggml-cpu.so\ngpu_offload=False\n") == []


class TestCheckLlmMemory:
    def device(self, check_gpu, kind, gib, name="GPU"):
        return check_gpu.Device(kind, int(gib * GIB), int(gib * GIB), name)

    def test_model_fits(self, check_gpu):
        devices = [self.device(check_gpu, "gpu", 11.9, "Arc B580")]
        result = check_gpu.check_llm_memory("openvino", devices, int(9.9 * GIB), "qwen.gguf")
        assert result.ok is True
        assert not result.warning
        assert "qwen.gguf needs ~9.9 GiB, Arc B580 has 11.9 GiB" in result.detail

    def test_too_little_memory_is_a_warning(self, check_gpu):
        devices = [self.device(check_gpu, "gpu", 7.9, "RTX 3070")]
        result = check_gpu.check_llm_memory("cuda", devices, int(9.9 * GIB), "qwen.gguf")
        assert result.ok is None
        assert result.warning
        assert "runs on the CPU" in result.detail
        assert "API provider" in result.hint

    def test_largest_gpu_counts(self, check_gpu):
        devices = [
            self.device(check_gpu, "gpu", 4, "small"),
            self.device(check_gpu, "gpu", 24, "big"),
        ]
        result = check_gpu.check_llm_memory("cuda", devices, int(9.9 * GIB), "qwen.gguf")
        assert result.ok is True
        assert "big" in result.detail

    def test_cpu_backend_is_a_warning(self, check_gpu):
        result = check_gpu.check_llm_memory("cpu", [], int(9.9 * GIB), "qwen.gguf")
        assert result.warning
        assert "no GPU" in result.detail

    def test_integrated_gpu_is_a_warning(self, check_gpu):
        devices = [self.device(check_gpu, "igpu", 16, "UHD Graphics 770")]
        result = check_gpu.check_llm_memory("openvino", devices, int(9.9 * GIB), "qwen.gguf")
        assert result.warning
        assert "integrated GPU (UHD Graphics 770)" in result.detail

    def test_no_memory_information(self, check_gpu):
        result = check_gpu.check_llm_memory("openvino", [], int(9.9 * GIB), "qwen.gguf")
        assert result.ok is None
        assert not result.warning

    def test_no_local_llm(self, check_gpu):
        result = check_gpu.check_llm_memory("openvino", [], None, "not needed: scoring uses openai")
        assert result.ok is None
        assert not result.warning
        assert result.detail == "not needed: scoring uses openai"


class TestLlmMemoryNeed:
    def write_config(self, tmp_path, llm: str) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text("scoring:\n  llm:\n" + llm)
        return path

    def test_model_on_disk(self, check_gpu, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"x" * 1000)
        config = self.write_config(
            tmp_path, f"    enabled: true\n    provider: llama_cpp\n    model_path: {model}\n"
        )
        need, name = check_gpu.llm_memory_need(config)
        assert need == 1000 + check_gpu.LLM_OVERHEAD_BYTES
        assert name == "model.gguf"

    def test_pinned_size_before_download(self, check_gpu, tmp_path):
        config = self.write_config(
            tmp_path,
            "    enabled: true\n    provider: llama_cpp\n"
            f"    model_path: {tmp_path / 'missing.gguf'}\n"
            "    model_repo: bartowski/Qwen_Qwen3-14B-GGUF\n"
            "    model_file: Qwen_Qwen3-14B-Q4_K_M.gguf\n",
        )
        need, _ = check_gpu.llm_memory_need(config)
        assert need == 9001753632 + check_gpu.LLM_OVERHEAD_BYTES

    def test_api_provider_needs_no_gpu_memory(self, check_gpu, tmp_path):
        config = self.write_config(tmp_path, "    enabled: true\n    provider: openai\n")
        need, reason = check_gpu.llm_memory_need(config)
        assert need is None
        assert "openai" in reason

    def test_default_config_is_qwen(self, check_gpu):
        need, name = check_gpu.llm_memory_need()
        assert name == "Qwen_Qwen3-14B-Q4_K_M.gguf"
        assert need > 9 * GIB


class TestRunChecks:
    def test_llm_memory_uses_the_devices_of_the_llama_probe(self, check_gpu, tmp_path):
        probe = _completed(DEVICE_LINES, VULKAN_STDERR)
        with (
            patch.object(check_gpu, "run_llama_probe", return_value=probe),
            patch.object(check_gpu, "llm_memory_need", return_value=(int(9.9 * GIB), "q.gguf")),
        ):
            results = check_gpu.run_checks("openvino", tmp_path / "bin", tmp_path / "model")
        by_name = {r.component: r for r in results}
        assert by_name["llama.cpp"].ok is True
        assert by_name["LLM memory"].ok is True

    def test_probe_timeout(self, check_gpu, tmp_path):
        timeout = subprocess.TimeoutExpired("python", 120)
        with (
            patch.object(check_gpu, "run_llama_probe", side_effect=timeout),
            patch.object(check_gpu, "llm_memory_need", return_value=(int(9.9 * GIB), "q.gguf")),
        ):
            results = check_gpu.run_checks("openvino", tmp_path / "bin", tmp_path / "model")
        assert results[0].ok is False
        assert "timed out" in results[0].detail


class TestCheckVideoEncoder:
    """Video encoding: the encoder jobs will render with."""

    def _check(self, check_gpu, tmp_path, backend, found, rendering=None):
        import yaml

        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"rendering": rendering or {}}))
        with patch("src.rendering.find_gpu_encoder", return_value=found) as find:
            result = check_gpu.check_video_encoder(backend, config)
        return result, find

    def test_gpu_encoder_works(self, check_gpu, tmp_path):
        from src.rendering import VideoEncoder

        found = (VideoEncoder("h264_qsv", "qsv"), {})
        result, find = self._check(check_gpu, tmp_path, "openvino", found)
        assert result.ok is True and not result.warning
        assert result.detail == "h264_qsv (Quick Sync)"
        assert find.call_args.args[1:] == ("openvino", 1080, 1920)

    def test_vaapi_names_its_device(self, check_gpu, tmp_path):
        from src.rendering import VideoEncoder

        found = (VideoEncoder("h264_vaapi", "vaapi", "/dev/dri/renderD129"), {"h264_qsv": "x"})
        result, _ = self._check(check_gpu, tmp_path, "openvino", found)
        assert result.detail == "h264_vaapi (VAAPI) on /dev/dri/renderD129"

    def test_gpu_machine_on_cpu_is_a_warning(self, check_gpu, tmp_path):
        found = (None, {"h264_qsv": "unsupported"})
        result, _ = self._check(check_gpu, tmp_path, "openvino", found)
        assert result.ok is None and result.warning is True
        assert "libx264 on the CPU" in result.detail and "h264_qsv: unsupported" in result.detail
        assert "intel-media-va-driver" in result.hint

    def test_cpu_machine_is_fine(self, check_gpu, tmp_path):
        result, _ = self._check(check_gpu, tmp_path, "cpu", (None, {}))
        assert result.ok is None and not result.warning
        assert result.detail == "libx264 on the CPU (no GPU encoder)"

    def test_software_setting(self, check_gpu, tmp_path):
        result, _ = self._check(
            check_gpu, tmp_path, "cuda", (None, {}), {"video_encoder": "software"}
        )
        assert not result.warning
        assert "rendering.video_encoder: software" in result.detail

    def test_explicit_encoder_that_fails(self, check_gpu, tmp_path):
        found = (None, {"h264_nvenc": "no device"})
        result, _ = self._check(check_gpu, tmp_path, "cuda", found, {"video_encoder": "nvenc"})
        assert result.ok is False
        assert "nvenc does not work: h264_nvenc: no device" in result.detail


LDD_OK = "\tlibcublas.so.12 => /venv/nvidia/cublas/lib/libcublas.so.12 (0x1)\n"
LDD_CUDA13 = (
    "\tlibcublas.so.13 => not found\n\tlibcudart.so.13 => not found\n"
    "\tlibcuda.so.1 => not found\n"
)


class TestCheckCudaLibraries:
    @pytest.fixture
    def libs(self, check_gpu, tmp_path, monkeypatch):
        """An installed CUDA build of llama.cpp; ldd output per test."""
        files = {"llama_cpp": tmp_path / "libggml-cuda.so"}
        for path in files.values():
            path.write_text("")
        monkeypatch.setattr(check_gpu, "package_file", lambda package, _: files[package])
        return files

    def run(self, check_gpu, ldd, loadable=True, ignore_driver=False):
        def cdll(name):
            if not loadable:
                raise OSError(f"{name}: cannot open shared object file")

        with (
            patch.object(check_gpu.subprocess, "run", return_value=_completed(ldd)),
            patch.object(check_gpu.ctypes, "CDLL", side_effect=cdll),
        ):
            return check_gpu.check_cuda_libraries("cuda", ignore_driver=ignore_driver)

    def test_all_found(self, check_gpu, libs):
        result = self.run(check_gpu, LDD_OK)
        assert result.ok is True

    def test_libraries_of_another_cuda(self, check_gpu, libs):
        # A llama.cpp built with CUDA 13 next to torch's CUDA 12 libraries.
        result = self.run(check_gpu, LDD_CUDA13)
        assert result.ok is False
        assert "llama.cpp needs libcublas.so.13, libcuda.so.1, libcudart.so.13" in result.detail
        assert "uv sync" in result.hint

    def test_driver_is_ignored_without_a_gpu(self, check_gpu, libs):
        result = self.run(check_gpu, "\tlibcuda.so.1 => not found\n", ignore_driver=True)
        assert result.ok is True

    def test_ctranslate2_libraries_missing(self, check_gpu, libs):
        result = self.run(check_gpu, LDD_OK, loadable=False)
        assert result.ok is False
        assert "faster-whisper needs libcublas.so.12" in result.detail
        assert "libcudnn.so.9" in result.detail

    def test_cpu_build_of_llama(self, check_gpu, libs):
        libs["llama_cpp"].unlink()
        result = self.run(check_gpu, LDD_OK)
        assert result.ok is False
        assert "llama.cpp: libggml-cuda.so not found" in result.detail

    def test_other_backends(self, check_gpu):
        assert check_gpu.check_cuda_libraries("openvino").ok is None

    def test_libraries_only_runs_without_a_gpu(self, check_gpu, capsys):
        ok = check_gpu.CheckResult("CUDA libraries", True, "fine")
        with (
            patch.object(check_gpu, "check_cuda_libraries", return_value=ok) as libraries,
            patch.object(check_gpu, "run_checks") as full,
            patch.object(sys, "argv", ["check_gpu.py", "--backend", "cuda", "--libraries-only"]),
        ):
            assert check_gpu.main() == 0
        libraries.assert_called_once_with("cuda", ignore_driver=True)
        full.assert_not_called()


class TestFormatAndExit:
    def test_failure_shows_hint_and_exits_1(self, check_gpu, capsys):
        results = [
            check_gpu.CheckResult("llama.cpp", True, "Vulkan: Arc"),
            check_gpu.CheckResult("whisper.cpp", False, "runs on CPU", "rebuild it"),
            check_gpu.CheckResult("faster-whisper", None, "not used"),
        ]
        with (
            patch.object(check_gpu, "run_checks", return_value=results),
            patch.object(sys, "argv", ["check_gpu.py", "--backend", "openvino"]),
        ):
            assert check_gpu.main() == 1
        out = capsys.readouterr().out
        assert "✓ llama.cpp" in out
        assert "✗ whisper.cpp" in out
        assert "rebuild it" in out
        assert "- faster-whisper" in out

    def test_warning_is_marked_and_does_not_fail(self, check_gpu, capsys):
        results = [
            check_gpu.CheckResult("LLM memory", None, "too small", "use an API", warning=True)
        ]
        with (
            patch.object(check_gpu, "run_checks", return_value=results),
            patch.object(sys, "argv", ["check_gpu.py", "--backend", "cuda"]),
        ):
            assert check_gpu.main() == 0
        out = capsys.readouterr().out
        # setup.sh looks for this marker.
        assert "  ! LLM memory" in out
        assert "use an API" in out

    def test_not_checked_does_not_fail(self, check_gpu):
        results = [check_gpu.CheckResult("whisper.cpp", None, "not checked", "prefetch")]
        with (
            patch.object(check_gpu, "run_checks", return_value=results),
            patch.object(sys, "argv", ["check_gpu.py", "--backend", "openvino"]),
        ):
            assert check_gpu.main() == 0


class TestDefaultWhisperBinary:
    """install.sh builds whisper-cli into artifacts/; the Docker image has it in PATH."""

    def test_install_sh_location_first(self, check_gpu, tmp_path, monkeypatch):
        built = tmp_path / "whisper-cli"
        built.write_text("")
        monkeypatch.setattr(check_gpu, "DEFAULT_WHISPER_CPP_BINARY", built)
        assert check_gpu.default_whisper_binary() == built

    def test_falls_back_to_path(self, check_gpu, tmp_path, monkeypatch):
        monkeypatch.setattr(check_gpu, "DEFAULT_WHISPER_CPP_BINARY", tmp_path / "missing")
        monkeypatch.setattr(check_gpu.shutil, "which", lambda name: "/opt/whisper.cpp/bin/" + name)
        assert check_gpu.default_whisper_binary() == Path("/opt/whisper.cpp/bin/whisper-cli")
