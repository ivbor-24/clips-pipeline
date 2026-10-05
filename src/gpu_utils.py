"""
TASK-07: GPU Auto-Detection and Runtime Configuration Module

Detects available GPU backends and builds optimal runtime configuration
for ML models (Whisper, the LLM) based on detected hardware.

Outputs:
- GPUConfig dataclass with backend-specific settings
"""

import gc
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Literal, Optional, Union

import structlog

logger = structlog.get_logger("gpu_utils")


class GPUUtilsError(Exception):
    """Custom exception for GPU utility errors."""

    pass


@dataclass(frozen=True)
class GPUConfig:
    """Runtime GPU configuration for ML pipeline.

    Attributes:
        backend: Detected GPU backend ("cuda", "openvino", "rocm", "cpu").
        whisper_device: Device string for faster-whisper ("cuda" or "cpu").
        whisper_compute_type: Compute type for faster-whisper.
        cleanup_fn: Callable to invoke after model usage to free memory.
        vram_total_mb: Total VRAM in MB, if applicable.
        vram_available_mb: Available VRAM in MB, if applicable.
        gpu_name: Human-readable GPU name.
    """

    backend: str
    whisper_device: str
    whisper_compute_type: str
    cleanup_fn: Callable[[], None]
    vram_total_mb: Optional[int] = None
    vram_available_mb: Optional[int] = None
    gpu_name: Optional[str] = None


_cached_config: Optional[GPUConfig] = None


def _run_subprocess(cmd: List[str], timeout: int = 5, shell: bool = False) -> str:
    """Run a subprocess command and return stdout.

    Args:
        cmd: Command and arguments list.
        timeout: Timeout in seconds.
        shell: Whether to run through shell.

    Returns:
        Stripped stdout string.

    Raises:
        GPUUtilsError: If the command fails or is not found.
    """
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout, shell=shell
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.warning("subprocess_failed", cmd=cmd, stderr=e.stderr, returncode=e.returncode)
        raise GPUUtilsError(f"Command failed: {' '.join(cmd)}") from e
    except subprocess.TimeoutExpired:
        logger.warning("subprocess_timeout", cmd=cmd, timeout=timeout)
        raise GPUUtilsError(f"Command timed out: {' '.join(cmd)}")
    except FileNotFoundError:
        logger.warning("binary_not_found", cmd=cmd)
        raise GPUUtilsError(f"Binary not found: {cmd[0]}")


def detect_gpu_backend() -> Literal["cuda", "openvino", "rocm", "cpu"]:
    """Detect the best available GPU backend.

    Detection order:
        0. PIPELINE_HARDWARE: the backend a Docker image was built for (the
           container has no lspci or clinfo, and probing them only filled
           job.log with warnings and a wrong "backend=cpu")
        1. nvidia-smi (CUDA)
        2. clinfo (OpenVINO / ROCm)
        3. lspci (Intel / AMD / NVIDIA)
        4. CPU fallback

    Returns:
        Detected backend string.
    """
    image_backend = os.environ.get("PIPELINE_HARDWARE", "").strip().lower()
    if image_backend in ("cuda", "openvino", "rocm", "cpu"):
        logger.info("gpu_detected", backend=image_backend, method="PIPELINE_HARDWARE")
        return image_backend  # type: ignore[return-value]

    # 1. Try nvidia-smi for CUDA
    try:
        output = _run_subprocess(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=5,
        )
        if output:
            logger.info("gpu_detected", backend="cuda", method="nvidia-smi")
            return "cuda"
    except GPUUtilsError:
        pass

    # 2. Try clinfo for OpenCL platforms
    try:
        output = _run_subprocess(["clinfo", "-l"], timeout=5)
        if output:
            lower_output = output.lower()
            if "intel" in lower_output:
                logger.info("gpu_detected", backend="openvino", method="clinfo")
                return "openvino"
            if "amd" in lower_output or "advanced micro devices" in lower_output:
                logger.info("gpu_detected", backend="rocm", method="clinfo")
                return "rocm"
    except GPUUtilsError:
        pass

    # 3. Fallback to lspci
    try:
        output = _run_subprocess(
            ['lspci | grep -i "VGA\\|Display"'],
            timeout=5,
            shell=True,
        )
        if output:
            lower_output = output.lower()
            if "nvidia" in lower_output:
                logger.info("gpu_detected", backend="cuda", method="lspci")
                return "cuda"
            if "intel" in lower_output:
                logger.info("gpu_detected", backend="openvino", method="lspci")
                return "openvino"
            if "amd" in lower_output or "advanced micro devices" in lower_output:
                logger.info("gpu_detected", backend="rocm", method="lspci")
                return "rocm"
    except GPUUtilsError:
        pass

    logger.info("gpu_detected", backend="cpu", method="fallback")
    return "cpu"


# The device list llama.cpp's ggml backends report when they initialise. This
# is the same source of truth as scripts/check_gpu.py (keep the two probes in
# sync): for Intel Arc it names the Vulkan device (e.g. "Intel(R) Arc(tm) B580
# Graphics (BMG G21)") and its memory, which clinfo (OpenCL) does not see.
# Importing llama_cpp in this process could crash the job, so the probe runs
# in its own subprocess and prints one "device=<kind>\t<total>\t<name>" line
# per registered device (kind: 0 cpu, 1 gpu, 2 igpu, 3 accel; total in bytes).
_GGML_DEVICE_PROBE = r"""
import ctypes, os
import llama_cpp
lib = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
llama_cpp.llama_backend_init()
g = ctypes.CDLL(os.path.join(lib, "libggml.so"))
g.ggml_backend_dev_count.restype = ctypes.c_size_t
g.ggml_backend_dev_get.restype = ctypes.c_void_p
g.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
g.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
g.ggml_backend_dev_description.restype = ctypes.c_char_p
g.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
g.ggml_backend_dev_memory.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
]
for i in range(g.ggml_backend_dev_count()):
    dev = g.ggml_backend_dev_get(i)
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    g.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
    name = g.ggml_backend_dev_description(dev).decode(errors="replace")
    print("device=%d\t%d\t%s" % (g.ggml_backend_dev_type(dev), total.value, name))
"""


def parse_ggml_devices(stdout: str) -> List[Dict[str, Union[str, int]]]:
    """Devices from the ``device=<kind>\t<total>\t<name>`` lines of the probe."""
    devices: List[Dict[str, Union[str, int]]] = []
    for line in stdout.splitlines():
        if not line.startswith("device="):
            continue
        parts = line[len("device=") :].split("\t", 2)
        if len(parts) != 3:
            continue
        try:
            kind, total = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        devices.append({"kind": kind, "total": total, "name": parts[2]})
    return devices


def probe_ggml_devices() -> List[Dict[str, Union[str, int]]]:
    """GPU devices llama.cpp's ggml backends report, from a subprocess."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _GGML_DEVICE_PROBE],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_ggml_devices(proc.stdout)


def _best_gpu_device(
    devices: List[Dict[str, Union[str, int]]],
) -> Optional[Dict[str, Union[str, int]]]:
    """The best GPU ggml reported: a discrete GPU, else an integrated one."""
    gpus = [d for d in devices if d["kind"] == 1]
    if not gpus:
        gpus = [d for d in devices if d["kind"] == 2]
    if not gpus:
        return None
    return max(gpus, key=lambda d: int(d["total"]))


def get_gpu_info(backend: str) -> Dict[str, Union[str, int, None]]:
    """Retrieve GPU information for the given backend.

    Args:
        backend: Detected backend string.

    Returns:
        Dictionary with "name" and "vram_total_mb" keys.
    """
    if backend == "cuda":
        try:
            output = _run_subprocess(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                timeout=5,
            )
            if not output:
                return {"name": "Unknown NVIDIA GPU", "vram_total_mb": None}

            parts = [p.strip() for p in output.split(",")]
            name = parts[0] if parts else "Unknown NVIDIA GPU"
            vram_mb = None
            if len(parts) > 1:
                vram_str = parts[1].replace("MiB", "").replace("MB", "").strip()
                try:
                    vram_mb = int(vram_str)
                except ValueError:
                    logger.warning("failed_to_parse_vram", vram_str=vram_str)
            logger.info("gpu_info_parsed", backend=backend, name=name, vram_total_mb=vram_mb)
            return {"name": name, "vram_total_mb": vram_mb}
        except GPUUtilsError as e:
            logger.warning("gpu_info_failed", backend=backend, error=str(e))
            return {"name": "Unknown NVIDIA GPU", "vram_total_mb": None}

    if backend == "openvino":
        # Intel Arc runs through Vulkan (llama.cpp / whisper.cpp); clinfo
        # (OpenCL) does not see it in the image, so the name and VRAM come
        # from llama.cpp's ggml backend devices, as check_gpu.py does.
        gpu = _best_gpu_device(probe_ggml_devices())
        if gpu is None:
            logger.warning("gpu_info_unavailable", backend=backend)
            logger.info("gpu_info_parsed", backend=backend, name=None, vram_total_mb=None)
            return {"name": None, "vram_total_mb": None}
        name = gpu["name"] or None
        if int(gpu["kind"]) == 2:
            # An integrated GPU shares system memory: its "total" is not
            # dedicated VRAM, so it must not be reported as such (the LLM needs
            # real VRAM; check_gpu.py warns about this the same way).
            logger.warning(
                "gpu_info_shared_memory", backend=backend, name=name, total_bytes=int(gpu["total"])
            )
            logger.info("gpu_info_parsed", backend=backend, name=name, vram_total_mb=None)
            return {"name": name, "vram_total_mb": None}
        vram_mb = int(gpu["total"]) // (1024 * 1024) if int(gpu["total"]) else None
        logger.info("gpu_info_parsed", backend=backend, name=name, vram_total_mb=vram_mb)
        return {"name": name, "vram_total_mb": vram_mb}

    if backend == "rocm":
        try:
            output = _run_subprocess(["clinfo", "-l"], timeout=5)
            name = "Unknown AMD GPU"
            for line in output.splitlines():
                if "AMD" in line or "Advanced Micro Devices" in line:
                    name = line.strip()
                    break
            logger.info("gpu_info_parsed", backend=backend, name=name, vram_total_mb=None)
            return {"name": name, "vram_total_mb": None}
        except GPUUtilsError as e:
            logger.warning("gpu_info_failed", backend=backend, error=str(e))
            return {"name": "Unknown AMD GPU", "vram_total_mb": None}

    # cpu fallback
    logger.info("gpu_info_parsed", backend=backend, name="CPU", vram_total_mb=0)
    return {"name": "CPU", "vram_total_mb": 0}


def cuda_cleanup() -> None:
    """Clean up CUDA memory: garbage collect and empty CUDA cache."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("cuda_cleanup_complete")
    except ImportError:
        logger.debug("torch_not_available_cuda_cleanup_skipped")


def cpu_cleanup() -> None:
    """Clean up CPU memory: garbage collect."""
    gc.collect()
    logger.debug("cpu_cleanup_complete")


def build_gpu_config(backend: str, gpu_info: Dict[str, Union[str, int, None]]) -> GPUConfig:
    """Build GPUConfig based on detected backend and GPU info.

    Args:
        backend: Detected backend string.
        gpu_info: Dictionary from get_gpu_info().

    Returns:
        Configured GPUConfig instance.
    """
    vram_total_mb = gpu_info.get("vram_total_mb")
    gpu_name = gpu_info.get("name")

    if backend == "cuda":
        if isinstance(vram_total_mb, int) and vram_total_mb > 8192:
            whisper_device = "cuda"
            compute_type = "float16"
        else:
            whisper_device = "cuda"
            compute_type = "int8"
        cleanup = cuda_cleanup
    else:
        # Intel (openvino): the GPU is used by llama.cpp and whisper.cpp
        # (Vulkan); faster-whisper (CTranslate2) has no Intel GPU backend.
        whisper_device = "cpu"
        compute_type = "int8"
        cleanup = cpu_cleanup

    config = GPUConfig(
        backend=backend,
        whisper_device=whisper_device,
        whisper_compute_type=compute_type,
        cleanup_fn=cleanup,
        vram_total_mb=vram_total_mb,
        vram_available_mb=None,
        gpu_name=gpu_name,
    )
    logger.info(
        "gpu_config_built",
        backend=backend,
        whisper_device=whisper_device,
        compute_type=compute_type,
        vram_total_mb=vram_total_mb,
    )
    return config


def _apply_gpu_overrides(config: GPUConfig, gpu_settings) -> GPUConfig:
    """Apply explicit GPU overrides from GPUSettings to a GPUConfig."""
    kwargs = {}
    if gpu_settings.whisper_device_override:
        kwargs["whisper_device"] = gpu_settings.whisper_device_override
    if gpu_settings.whisper_compute_override:
        kwargs["whisper_compute_type"] = gpu_settings.whisper_compute_override
    return replace(config, **kwargs) if kwargs else config


def get_auto_gpu_config(gpu_settings=None) -> GPUConfig:
    """Get (cached) auto-detected GPU configuration.

    Args:
        gpu_settings: Optional GPUSettings (from Config.gpu) to override
            auto-detection. A non-auto backend or explicit device/compute
            overrides bypass the cache.

    Returns:
        GPUConfig with detected backend and optimal settings.
    """
    global _cached_config

    has_overrides = gpu_settings is not None and (
        gpu_settings.backend != "auto"
        or gpu_settings.whisper_device_override
        or gpu_settings.whisper_compute_override
    )

    if has_overrides:
        backend = gpu_settings.backend if gpu_settings.backend != "auto" else detect_gpu_backend()
        gpu_info = get_gpu_info(backend)
        config = _apply_gpu_overrides(build_gpu_config(backend, gpu_info), gpu_settings)
        logger.info(
            "gpu_config_with_overrides",
            backend=config.backend,
            whisper_device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )
        return config

    if _cached_config is not None:
        logger.info("gpu_cache_hit", backend=_cached_config.backend)
        return _cached_config

    backend = detect_gpu_backend()
    gpu_info = get_gpu_info(backend)
    config = build_gpu_config(backend, gpu_info)
    _cached_config = config
    return config


def get_vram_gb() -> float:
    """Detect total VRAM in gigabytes.

    Returns:
        Total VRAM in GB, or 0.0 if no GPU VRAM detected.
    """
    gpu_config = get_auto_gpu_config()
    vram_mb = gpu_config.vram_total_mb
    if vram_mb is None:
        return 0.0
    return round(vram_mb / 1024.0, 2)


def suggest_profile(vram_gb: float) -> Optional[str]:
    """Suggest a configuration profile based on available VRAM.

    Args:
        vram_gb: Total VRAM in gigabytes.

    Returns:
        Profile name string or None if no special profile needed.
    """
    if vram_gb < 10.0:
        logger.info("profile_suggested", vram_gb=vram_gb, profile="low_vram")
        return "low_vram"
    return None
