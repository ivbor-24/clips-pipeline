#!/usr/bin/env python3
"""
Check that the installed pipeline really uses the GPU.

A build that silently fell back to the CPU still runs, only several times
slower, and someone who is not a specialist has no way to notice. This check
asks each component which device it would use, without loading the LLM:

- llama.cpp (scoring): was it built with a GPU backend, which devices it sees;
- whisper.cpp (transcription on Intel Arc): which backend whisper-cli picks
  on one second of silence (needs the ggml model, see `just prefetch-models`);
- faster-whisper (transcription on NVIDIA): does CTranslate2 see a CUDA device;
- CUDA libraries (NVIDIA): do llama.cpp and CTranslate2 find the CUDA and
  cuDNN libraries of the version they were built for;
- LLM memory: does the GPU have room for the configured local LLM (Qwen3-14B
  needs ~10 GiB). Too little is a warning, not an error: the
  LLM still runs, partly on the CPU and much slower.
- Video encoding: which encoder renders the clips: a GPU one (Quick Sync,
  VAAPI, NVENC) after a test encode, or libx264 on the CPU. The CPU on a GPU
  machine is a warning: rendering works, only slower.

Usage:
    python3 scripts/check_gpu.py [--backend cpu|cuda|openvino|rocm]
    python3 scripts/check_gpu.py --backend cuda --libraries-only   # no GPU needed (CI)
    python3 scripts/check_gpu.py --json   # one JSON line (the worker, for Diagnostics)
    just check-gpu [backend]

Without --backend the backend is taken from the llama.cpp build.
Exit code 1 if a component that should run on the GPU does not. Warnings
are marked "!" (setup.sh reads the "LLM memory" one).
"""

import argparse
import ctypes
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Where install.sh builds whisper-cli (config/profiles/intel_arc.yaml); the
# Docker image has it in PATH instead (config/hardware/openvino.yaml).
DEFAULT_WHISPER_CPP_BINARY = ROOT / "artifacts" / "cache" / "whisper-cpp-bin" / "whisper-cli"
DEFAULT_WHISPER_CPP_MODEL = ROOT / "artifacts" / "cache" / "models" / "ggml-large-v3-turbo.bin"

BACKENDS = ("cpu", "cuda", "openvino", "rocm")
GPU_NAMES = {"cuda": "CUDA", "openvino": "Vulkan", "rocm": "ROCm"}

LLAMA_PROBE = """
import ctypes, os, llama_cpp
lib = os.path.join(os.path.dirname(llama_cpp.__file__), 'lib')
print('libs=%s' % ','.join(os.listdir(lib)))
llama_cpp.llama_backend_init()
print('gpu_offload=%s' % llama_cpp.llama_supports_gpu_offload())
# Every device ggml registered: type, total and free memory in bytes, name.
try:
    g = ctypes.CDLL(os.path.join(lib, 'libggml.so'))
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
        name = g.ggml_backend_dev_description(dev).decode(errors='replace')
        kind = g.ggml_backend_dev_type(dev)
        print('device=%d\t%d\t%d\t%s' % (kind, total.value, free.value, name))
except (OSError, AttributeError):
    pass
"""
# The GPU backend library shipped in llama_cpp/lib shows how it was built.
# llama_supports_gpu_offload() alone cannot tell a CPU-only build from a GPU
# build that finds no device (missing driver): it is False in both cases.
LLAMA_BACKEND_LIBS = {"openvino": "ggml-vulkan", "cuda": "ggml-cuda", "rocm": "ggml-hip"}
# ggml prints the devices it finds while the backend initialises, e.g.
#   ggml_vulkan: 0 = Intel(R) Arc(tm) B580 Graphics (BMG G21) (Intel open-source Mesa driver) | ...
#   ggml_cuda_init: found 1 CUDA devices:
#     Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes
# (the HIP build of ggml reuses the CUDA code and prints the same lines).
VULKAN_DEVICE = re.compile(r"ggml_vulkan: \d+ = (.+?)(?: \(|\s*\||$)", re.MULTILINE)
CUDA_DEVICE = re.compile(r"^\s*Device \d+: ([^,\n]+)", re.MULTILINE)
LLAMA_DEVICE_PATTERNS = {"openvino": VULKAN_DEVICE, "cuda": CUDA_DEVICE, "rocm": CUDA_DEVICE}
WHISPER_BACKEND_PATTERN = re.compile(r"whisper_backend_init_gpu: using (\S+) backend")

# enum ggml_backend_dev_type
DEVICE_KINDS = {0: "cpu", 1: "gpu", 2: "igpu", 3: "accel"}
GIB = 2**30
# GPU memory the LLM needs on top of its weights: KV cache and compute buffers
# at the default n_ctx of 8192 (Qwen3-14B: ~1.3 GiB of KV cache).
LLM_OVERHEAD_BYTES = int(1.5 * GIB)
# Libraries of the NVIDIA driver: the host provides them (NVIDIA Container
# Toolkit), so a build checked without a GPU (--libraries-only) lacks them.
DRIVER_LIBRARIES = {"libcuda.so.1", "libnvidia-ml.so.1"}
# CTranslate2 (faster-whisper) loads cuBLAS and cuDNN by name when it starts
# on the GPU, instead of linking them.
CTRANSLATE2_CUDA_LIBRARIES = ("libcublas.so.12", "libcudnn.so.9")
API_PROVIDER_HINT = (
    "For fast scoring use an API provider: OPENAI_API_KEY or ANTHROPIC_API_KEY in .env "
    "and scoring.llm.provider in the config (Settings page of the web UI)"
)


@dataclass
class CheckResult:
    """Outcome of one component check. ``ok`` is None when it was not checked.

    A warning (``ok`` None, ``warning`` True) does not fail the check.
    """

    component: str
    ok: Optional[bool]
    detail: str
    hint: str = ""
    warning: bool = False


@dataclass
class Device:
    """A device ggml registered: kind is gpu, igpu (integrated), cpu or accel."""

    kind: str
    total: int
    free: int
    name: str


def detect_backend() -> str:
    """Guess the backend from the GPU library the llama.cpp build ships."""
    lib = package_file("llama_cpp", "lib")
    names = " ".join(p.name for p in lib.iterdir()) if lib is not None and lib.is_dir() else ""
    for backend, name in LLAMA_BACKEND_LIBS.items():
        if name in names:
            return backend
    return "cpu"


def parse_llama_devices(backend: str, stderr: str) -> List[str]:
    """Devices that the backend's ggml GPU backend reported while initialising."""
    pattern = LLAMA_DEVICE_PATTERNS.get(backend)
    if pattern is None:
        return []
    return [name.strip() for name in pattern.findall(stderr)]


def parse_devices(stdout: str) -> List[Device]:
    """Devices with their memory, from the ``device=`` lines of LLAMA_PROBE."""
    devices = []
    for line in stdout.splitlines():
        if not line.startswith("device="):
            continue
        kind, total, free, name = line[len("device=") :].split("\t", 3)
        devices.append(Device(DEVICE_KINDS.get(int(kind), "other"), int(total), int(free), name))
    return devices


def run_llama_probe() -> subprocess.CompletedProcess:
    """Initialise llama.cpp's backends in a subprocess (a crash stays there)."""
    return subprocess.run(
        [sys.executable, "-c", LLAMA_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )


def check_llama(backend: str, proc: Optional[subprocess.CompletedProcess] = None) -> CheckResult:
    """Ask llama.cpp in a subprocess whether it was built for, and sees, a GPU."""
    rebuild = f"LLAMA_FORCE_REBUILD=1 scripts/install_llama_cpp_backend.sh {backend}"
    if proc is None:
        try:
            proc = run_llama_probe()
        except subprocess.TimeoutExpired:
            return CheckResult("llama.cpp", False, "backend initialisation timed out")
    if proc.returncode != 0:
        return CheckResult(
            "llama.cpp", False, "llama-cpp-python does not import", f"Reinstall it: {rebuild}"
        )

    gpu_offload = "gpu_offload=True" in proc.stdout
    devices = parse_llama_devices(backend, proc.stderr)
    if backend == "cpu":
        return CheckResult("llama.cpp", True, "CPU build" if not gpu_offload else "GPU build")
    gpu = GPU_NAMES[backend]
    libs = next((line for line in proc.stdout.splitlines() if line.startswith("libs=")), "")
    if LLAMA_BACKEND_LIBS[backend] not in libs:
        return CheckResult(
            "llama.cpp", False, f"built without {gpu} support (runs on CPU)", f"Rebuild: {rebuild}"
        )
    if not gpu_offload or not devices:
        return CheckResult(
            "llama.cpp",
            False,
            f"built with {gpu}, but no {gpu} device found (GPU driver missing or not loaded)",
            "Check the GPU driver: vulkaninfo --summary (Intel, AMD) or nvidia-smi (NVIDIA)",
        )
    return CheckResult("llama.cpp", True, f"{gpu}: {', '.join(devices)}")


def check_whisper_cpp(backend: str, binary: Path, model: Path) -> CheckResult:
    """Run whisper-cli on one second of silence and read which backend it picked."""
    if backend != "openvino":
        return CheckResult("whisper.cpp", None, "not used on this backend")
    if not binary.exists():
        return CheckResult(
            "whisper.cpp",
            False,
            f"{binary} not found",
            "Build it: scripts/install_whisper_cpp.sh openvino",
        )
    if not model.exists():
        return CheckResult(
            "whisper.cpp",
            None,
            "not checked: the ggml model is not downloaded yet",
            "Download it: just prefetch-models (Docker: scripts/prefetch_models.py), "
            "then check again",
        )

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "silence.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=16000:cl=mono",
                    "-t",
                    "1",
                    str(wav),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            proc = subprocess.run(
                # No -np: it also silences the backend lines read below.
                [str(binary), "-m", str(model), "-f", str(wav), "-l", "en"],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (subprocess.SubprocessError, OSError) as e:
            return CheckResult("whisper.cpp", False, f"test run failed: {e}")

    match = WHISPER_BACKEND_PATTERN.search(proc.stderr)
    if not match or match.group(1).upper().startswith("CPU"):
        return CheckResult(
            "whisper.cpp",
            False,
            "runs on CPU: no GPU backend picked",
            "Check the GPU driver (vulkaninfo --summary), then rebuild: "
            "scripts/install_whisper_cpp.sh openvino --force",
        )
    return CheckResult("whisper.cpp", True, match.group(1))


def check_faster_whisper(backend: str) -> CheckResult:
    """On NVIDIA, faster-whisper transcribes through CTranslate2 with CUDA."""
    if backend != "cuda":
        return CheckResult("faster-whisper", None, "not used on the GPU with this backend")
    try:
        import ctranslate2
    except ImportError:
        return CheckResult("faster-whisper", False, "ctranslate2 not installed")
    count = ctranslate2.get_cuda_device_count()
    if count == 0:
        return CheckResult(
            "faster-whisper", False, "CTranslate2 sees no CUDA device", "Check nvidia-smi"
        )
    return CheckResult("faster-whisper", True, f"CUDA devices: {count}")


DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def llm_memory_need(config_path: Path = DEFAULT_CONFIG) -> Tuple[Optional[int], str]:
    """GPU memory in bytes the configured local LLM needs, and the model file name.

    Returns (None, reason) when no local LLM runs (off, or an API provider) or
    its size is unknown.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from src.config import Config
        from src.model_registry import find_pinned_file

        llm = Config.load(str(config_path)).with_hardware_profile().scoring.llm
    except Exception as e:
        return None, f"not checked: cannot read the config ({e})"
    if not llm.enabled:
        return None, "not needed: the LLM is off (scoring.llm.enabled)"
    if llm.provider != "llama_cpp":
        return None, f"not needed: scoring uses {llm.provider}"
    path = Path(llm.model_path) if llm.model_path else None
    if path is not None and not path.is_absolute():
        path = ROOT / path
    name = path.name if path is not None else (llm.model_file or "the LLM")
    if path is not None and path.exists():
        size = path.stat().st_size
    else:
        pin = find_pinned_file(llm.model_repo, llm.model_file) if llm.model_repo else None
        if pin is None:
            return None, f"not checked: size of {name} unknown"
        size = pin.size
    return size + LLM_OVERHEAD_BYTES, name


def check_llm_memory(
    backend: str, devices: List[Device], need: Optional[int], model: str
) -> CheckResult:
    """Does the GPU have room for the LLM? Too little is a warning."""
    component = "LLM memory"
    if need is None:
        return CheckResult(component, None, model)
    if backend == "cpu":
        return CheckResult(
            component,
            None,
            f"no GPU: {model} runs on the CPU, choosing clips takes hours per video",
            API_PROVIDER_HINT,
            warning=True,
        )
    gpus = [d for d in devices if d.kind == "gpu"]
    if not gpus:
        igpus = [d for d in devices if d.kind == "igpu"]
        if igpus:
            return CheckResult(
                component,
                None,
                f"integrated GPU ({igpus[0].name}) shares system memory: {model} runs slowly",
                API_PROVIDER_HINT,
                warning=True,
            )
        return CheckResult(component, None, "not checked: no GPU memory information")
    best = max(gpus, key=lambda d: d.total)
    detail = f"{model} needs ~{need / GIB:.1f} GiB, {best.name} has {best.total / GIB:.1f} GiB"
    if best.total >= need:
        return CheckResult(component, True, detail)
    return CheckResult(
        component,
        None,
        f"{detail}: part of it runs on the CPU, choosing clips takes hours per video",
        API_PROVIDER_HINT,
        warning=True,
    )


def package_file(package: str, relative: str) -> Optional[Path]:
    """A file inside an installed package, found without importing it (an
    import can fail on exactly the missing library that is being checked)."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]) / relative


def missing_libraries(path: Path) -> List[str]:
    """Shared libraries that ``path`` needs and the dynamic loader cannot find."""
    try:
        proc = subprocess.run(["ldd", str(path)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted({line.split()[0] for line in proc.stdout.splitlines() if "not found" in line})


def check_cuda_libraries(backend: str, ignore_driver: bool = False) -> CheckResult:
    """Do the CUDA builds find the CUDA and cuDNN libraries of their version?"""
    component = "CUDA libraries"
    if backend != "cuda":
        return CheckResult(component, None, "not used on this backend")
    problems = []
    lib = package_file("llama_cpp", "lib/libggml-cuda.so")
    if lib is None or not lib.exists():
        problems.append(f"llama.cpp: {lib.name if lib else 'package'} not found")
    else:
        missing = [
            m for m in missing_libraries(lib) if not (ignore_driver and m in DRIVER_LIBRARIES)
        ]
        if missing:
            problems.append(f"llama.cpp needs {', '.join(missing)}")
    for soname in CTRANSLATE2_CUDA_LIBRARIES:
        try:
            ctypes.CDLL(soname)
        except OSError:
            problems.append(f"faster-whisper needs {soname}")
    if problems:
        return CheckResult(
            component,
            False,
            "; ".join(problems),
            "The CUDA 12 libraries come with torch (nvidia-* packages): "
            "uv sync --frozen --inexact --extra cuda (Docker: ./setup.sh rebuilds the image)",
        )
    return CheckResult(component, True, "llama.cpp and CTranslate2 find CUDA 12")


ENCODER_NAMES = {"qsv": "Quick Sync", "vaapi": "VAAPI", "nvenc": "NVENC"}


def check_video_encoder(backend: str, config_path: Path = DEFAULT_CONFIG) -> CheckResult:
    """Which encoder renders the clips, by the same test encode as a job."""
    component = "Video encoding"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from src.config import Config
        from src.rendering import _ENCODER_HINT, find_gpu_encoder, software_encoder

        config = Config.load(str(config_path)).with_hardware_profile()
    except Exception as e:
        return CheckResult(component, None, f"not checked: cannot read the config ({e})")
    rendering = config.rendering
    try:
        encoder, errors = find_gpu_encoder(
            rendering, backend, config.cropping.output_width, config.cropping.output_height
        )
    except Exception as e:  # ffmpeg missing: rendering fails anyway, say why
        return CheckResult(component, False, str(e))
    if encoder is not None:
        where = f" on {encoder.device}" if encoder.device else ""
        return CheckResult(
            component, True, f"{encoder.name} ({ENCODER_NAMES[encoder.kind]}){where}"
        )
    software = software_encoder(rendering).name
    tried = "; ".join(f"{name}: {error}" for name, error in errors.items())
    if rendering.video_encoder in ENCODER_NAMES:
        return CheckResult(
            component,
            False,
            f"{rendering.video_encoder} does not work: {tried or 'no GPU render node'}",
            _ENCODER_HINT + " Or set rendering.video_encoder to auto.",
        )
    if rendering.video_encoder == "software" or backend == "cpu" or not errors:
        reason = (
            "rendering.video_encoder: software"
            if rendering.video_encoder == "software"
            else "no GPU encoder"
        )
        return CheckResult(component, None, f"{software} on the CPU ({reason})")
    return CheckResult(
        component,
        None,
        f"{software} on the CPU, slower: the GPU encoder did not start ({tried})",
        _ENCODER_HINT,
        warning=True,
    )


def run_checks(backend: str, whisper_binary: Path, whisper_model: Path) -> List[CheckResult]:
    try:
        probe = run_llama_probe()
    except subprocess.TimeoutExpired:
        probe = None
    if probe is None:
        llama = CheckResult("llama.cpp", False, "backend initialisation timed out")
        devices = []
    else:
        llama = check_llama(backend, probe)
        devices = parse_devices(probe.stdout)
    return [
        llama,
        check_llm_memory(backend, devices, *llm_memory_need()),
        check_whisper_cpp(backend, whisper_binary, whisper_model),
        check_faster_whisper(backend),
        check_cuda_libraries(backend),
        check_video_encoder(backend),
    ]


def format_results(backend: str, results: List[CheckResult]) -> str:
    marks = {True: "✓", False: "✗", None: "-"}
    lines = [f"GPU self-check (backend: {backend})"]
    for r in results:
        mark = "!" if r.warning else marks[r.ok]
        lines.append(f"  {mark} {r.component:<15} {r.detail}")
        if r.hint and r.ok is not True:
            lines.append(f"    {'':<15} {r.hint}")
    return "\n".join(lines)


def default_whisper_binary() -> Path:
    """install.sh's whisper-cli if it is there, else the one in PATH (Docker image)."""
    if DEFAULT_WHISPER_CPP_BINARY.exists():
        return DEFAULT_WHISPER_CPP_BINARY
    found = shutil.which("whisper-cli")
    return Path(found) if found else DEFAULT_WHISPER_CPP_BINARY


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=BACKENDS, help="default: detect from the llama.cpp build"
    )
    parser.add_argument("--whisper-binary", type=Path, default=None)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_CPP_MODEL)
    parser.add_argument(
        "--libraries-only",
        action="store_true",
        help="only check that the GPU libraries are complete; needs no GPU or driver (CI)",
    )
    parser.add_argument("--json", action="store_true", help="print the results as one JSON line")
    args = parser.parse_args()

    backend = args.backend or detect_backend()
    if args.libraries_only:
        results = [check_cuda_libraries(backend, ignore_driver=True)]
    else:
        binary = args.whisper_binary or default_whisper_binary()
        results = run_checks(backend, binary, args.whisper_model)
    failed = any(r.ok is False for r in results)
    if args.json:
        print(
            json.dumps(
                {"backend": backend, "ok": not failed, "results": [asdict(r) for r in results]}
            )
        )
    else:
        print(format_results(backend, results))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
