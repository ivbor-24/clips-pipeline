"""Tests for setup.sh, the one-command Docker installation.

setup.sh runs in a copy of the project files it needs, with a fake `docker`
(tests/fake_docker.py) and fake GPU tools first in PATH, and without a
terminal, so every question takes its default answer.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tests.fake_docker import add_command, free_port, make_fake_docker, stop_fake_web

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")
PROJECT_FILES = ["setup.sh", "scripts/docker_common.sh", "scripts/stack.sh", ".env.example"]
LOW_MEMORY_CHECK = (
    "GPU self-check (backend: cpu)\n"
    "  ! LLM memory      no GPU: Qwen_Qwen3-14B-Q4_K_M.gguf runs on the CPU\n"
)
ENOUGH_MEMORY_CHECK = (
    "GPU self-check (backend: openvino)\n"
    "  ✓ LLM memory      Qwen_Qwen3-14B-Q4_K_M.gguf needs ~9.9 GiB, B580 has 11.9 GiB\n"
)


@pytest.fixture
def env(tmp_path):
    """The project copy, the fake system and the environment to run setup.sh in."""
    project = tmp_path / "project"
    for name in PROJECT_FILES:
        (project / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, project / name)
    (project / "Dockerfile.backend").write_text("FROM python:3.11-slim AS base\n")
    bin_dir, state = make_fake_docker(tmp_path)
    # No NVIDIA driver and no lspci unless a test adds them.
    add_command(bin_dir, "nvidia-smi", "exit 1\n")
    add_command(bin_dir, "lspci", "exit 1\n")
    drm = tmp_path / "drm"
    drm.mkdir()
    dri = tmp_path / "dri"
    dri.mkdir()
    variables = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path),
        "FAKE_STATE": str(state),
        "SETUP_DRM_DIR": str(drm),
        "SETUP_DEV_DRI": str(dri),
        "SETUP_NEED_GB_IMAGES": "0",
        "SETUP_NEED_GB_MODELS": "0",
    }
    return {
        "project": project,
        "bin": bin_dir,
        "state": state,
        "drm": drm,
        "dri": dri,
        "vars": variables,
    }


def run_setup(env, *args, **extra_vars):
    variables = {**env["vars"], **extra_vars}
    return subprocess.run(
        [BASH, str(env["project"] / "setup.sh"), *args],
        cwd=env["project"],
        env=variables,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def run_setup_tty(env, answers, *args, **extra_vars):
    """Run setup.sh with a terminal as stdin, typing the answers in order.

    The web UI port is set to a free one first, so no question about a taken
    port comes between the expected ones.
    """
    env_file = env["project"] / ".env"
    if not env_file.exists():
        env_file.write_text(f"WEB_PORT={free_port()}\n")
    variables = {**env["vars"], **extra_vars}
    master, slave = os.openpty()
    try:
        os.write(master, "".join(answer + "\n" for answer in answers).encode())
        return subprocess.run(
            [BASH, str(env["project"] / "setup.sh"), *args],
            cwd=env["project"],
            env=variables,
            stdin=slave,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        os.close(master)
        os.close(slave)


def read_env(env) -> dict:
    values = {}
    for line in (env["project"] / ".env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            assert key not in values, f"{key} appears twice in .env"
            values[key] = value
    return values


def docker_calls(env) -> str:
    return (env["state"] / "log").read_text()


def add_gpu(env, vendor: str):
    device = env["drm"] / "renderD128" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor + "\n")
    (device / "device").write_text("0xe20b\n")
    (env["dri"] / "renderD128").write_text("")


QUICK = ("--yes", "--no-models", "--no-start")
# With the models, without questions or the start.
WITH_MODELS = ("--backend", "cpu", "--yes", "--no-start")
# Interactive runs: the questions before the LLM step are only "open the web
# UI to the network?" (answered with Enter: no).
TTY = ("--backend", "cpu", "--models-dir", "./models", "--no-start")
LLM_STATUS = (
    "LLM_ENABLED={enabled}\nLLM_PROVIDER={provider}\nLLM_MODEL={model}\n"
    "LLM_API_BASE=\nLLM_READY={ready}\n"
)


def set_llm_status(env, provider="llama_cpp", model="Qwen/Qwen3-14B", ready="no", enabled="yes"):
    """What scripts/llm_provider.py status reports in the worker container."""
    (env["state"] / "llm_status").write_text(
        LLM_STATUS.format(provider=provider, model=model, ready=ready, enabled=enabled)
    )


NVIDIA_RUNTIME = '{"nvidia":{},"runc":{}}'


def add_nvidia(env, driver="570.86.15", compute_caps="8.9"):
    """A working NVIDIA driver with an RTX 4070 (or the given GPUs)."""
    caps = compute_caps.replace("\n", "\\n")
    add_command(
        env["bin"],
        "nvidia-smi",
        f"""case "$*" in
    *driver_version*) echo "{driver}" ;;
    *compute_cap*) printf "{caps}\\n" ;;
    *) echo "NVIDIA GeForce RTX 4070" ;;
esac
""",
    )


class TestFreshInstall:
    def test_cpu_install_writes_env_and_builds(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        values = read_env(env)
        assert values["GPU_BACKEND"] == "cpu"
        assert values["COMPOSE_PROFILES"] == "cpu"
        assert values["HOST_UID"] == str(os.getuid())
        assert values["HOST_GID"] == str(os.getgid())
        assert values["BIND_ADDRESS"] == "127.0.0.1"
        assert values["MODELS_DIR"] == "./models"
        for name in ("data", "jobs", "uploads", "cache/models", "models"):
            assert (env["project"] / name).is_dir(), name
        calls = docker_calls(env)
        assert "compose build" in calls
        assert "worker-cpu python scripts/check_gpu.py --backend cpu" in calls
        assert "prefetch-models" not in calls
        assert "compose up" not in calls

    def test_downloads_models_before_the_gpu_check_and_the_llm_after(self, env):
        (env["state"] / "check_gpu").write_text(ENOUGH_MEMORY_CHECK)

        result = run_setup(env, "--backend", "cpu", "--yes", "--no-start")

        assert result.returncode == 0, result.stdout + result.stderr
        calls = docker_calls(env).splitlines()
        first = next(i for i, c in enumerate(calls) if "--skip-llm" in c)
        check = next(i for i, c in enumerate(calls) if "check_gpu.py" in c)
        llm = next(i for i, c in enumerate(calls) if "--skip-whisper" in c)
        assert first < check < llm

    def test_gpu_check_failure_stops(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK, FAKE_CHECK_GPU_RC="1")

        assert result.returncode == 1
        assert "does not use the GPU" in result.stderr

    def test_start_runs_the_stack(self, env):
        port = free_port()
        (env["project"] / ".env").write_text(f"WEB_PORT={port}\n")
        try:
            result = run_setup(
                env, "--backend", "cpu", "--yes", "--no-models", FAKE_WEB_PORT=str(port)
            )
        finally:
            stop_fake_web(env["state"])

        assert result.returncode == 0, result.stdout + result.stderr
        assert "compose up -d" in docker_calls(env)
        assert f"Running: http://127.0.0.1:{port} (version 9.9.9" in result.stdout


class TestGpuDetection:
    def test_intel_gpu_selects_openvino_and_the_render_group(self, env):
        add_gpu(env, "0x8086")

        result = run_setup(env, *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        values = read_env(env)
        assert values["GPU_BACKEND"] == "openvino"
        assert values["COMPOSE_PROFILES"] == "openvino"
        assert values["RENDER_GID"] == str((env["dri"] / "renderD128").stat().st_gid)
        assert "worker-openvino python scripts/check_gpu.py" in docker_calls(env)

    def test_nvidia_without_container_toolkit_stops(self, env):
        add_nvidia(env)

        result = run_setup(env, *QUICK)

        assert result.returncode == 1
        assert "NVIDIA Container Toolkit" in result.stderr

    def test_nvidia_with_container_toolkit(self, env):
        add_nvidia(env)

        result = run_setup(env, *QUICK, FAKE_RUNTIMES=NVIDIA_RUNTIME)

        assert result.returncode == 0, result.stdout + result.stderr
        values = read_env(env)
        assert values["GPU_BACKEND"] == "cuda"
        # llama.cpp is compiled for this GPU only.
        assert values["CUDA_ARCHITECTURES"] == "89"

    def test_nvidia_several_gpus(self, env):
        add_nvidia(env, compute_caps="8.6\n8.9\n8.6")

        result = run_setup(env, *QUICK, FAKE_RUNTIMES=NVIDIA_RUNTIME)

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["CUDA_ARCHITECTURES"] == "'86;89'"

    def test_old_nvidia_driver_stops(self, env):
        add_nvidia(env, driver="470.256.02")

        result = run_setup(env, *QUICK, FAKE_RUNTIMES=NVIDIA_RUNTIME)

        assert result.returncode == 1
        assert "too old for CUDA 12" in result.stderr

    def test_amd_gpu_falls_back_to_cpu_with_a_note(self, env):
        add_gpu(env, "0x1002")

        result = run_setup(env, *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["GPU_BACKEND"] == "cpu"
        assert "AMD GPUs are not supported" in result.stdout

    def test_openvino_without_intel_gpu_stops(self, env):
        result = run_setup(env, "--backend", "openvino", *QUICK)

        assert result.returncode == 1
        assert "needs an Intel GPU" in result.stderr


class TestRerun:
    def test_keeps_values_and_does_not_duplicate_keys(self, env, tmp_path):
        models = tmp_path / "my models"
        models.mkdir()
        run_setup(env, "--backend", "cpu", "--models-dir", str(models), *QUICK)
        env_file = env["project"] / ".env"
        env_file.write_text(
            env_file.read_text().replace("API_LOG_LEVEL=INFO", "API_LOG_LEVEL=DEBUG")
        )

        result = run_setup(env, *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        values = read_env(env)
        assert values["GPU_BACKEND"] == "cpu"
        assert values["MODELS_DIR"] == f"'{models}'"
        assert values["API_LOG_LEVEL"] == "DEBUG"

    def test_env_without_final_newline_gets_new_keys_on_their_own_line(self, env):
        (env["project"] / ".env").write_text("API_LOG_LEVEL=DEBUG")

        result = run_setup(env, "--backend", "cpu", *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        values = read_env(env)
        assert values["API_LOG_LEVEL"] == "DEBUG"
        assert values["GPU_BACKEND"] == "cpu"

    def test_suggests_the_models_of_a_native_install(self, env):
        native = env["project"] / "artifacts" / "cache" / "models"
        native.mkdir(parents=True)
        (native / "Qwen_Qwen3-14B-Q4_K_M.gguf").write_text("")

        result = run_setup(env, "--backend", "cpu", *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["MODELS_DIR"] == "./artifacts/cache/models"
        assert "models already there" in result.stdout

    def test_models_dir_without_models_says_it_will_download(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["MODELS_DIR"] == "./models"
        assert "they will be downloaded" in result.stdout


class TestNetworkAccess:
    def test_lan_without_password_stops(self, env):
        (env["project"] / ".env").write_text("BIND_ADDRESS=0.0.0.0\n")

        result = run_setup(env, "--backend", "cpu", *QUICK)

        assert result.returncode == 1
        assert "needs a password" in result.stderr

    def test_lan_with_password_generates_the_secret_once(self, env):
        (env["project"] / ".env").write_text("BIND_ADDRESS=0.0.0.0\nAPI_PASSWORD='pa$$ word'\n")

        result = run_setup(env, "--backend", "cpu", *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        values = read_env(env)
        assert values["BIND_ADDRESS"] == "0.0.0.0"
        assert values["API_PASSWORD"] == "'pa$$ word'"
        secret = values["API_JWT_SECRET"]
        assert len(secret) >= 16
        mode = stat.S_IMODE((env["project"] / ".env").stat().st_mode)
        assert mode == 0o600

        run_setup(env, *QUICK)
        assert read_env(env)["API_JWT_SECRET"] == secret


class TestDockerProblems:
    def test_docker_missing(self, env):
        (env["bin"] / "docker").unlink()

        result = run_setup(env, *QUICK)

        assert result.returncode == 1
        assert "Docker is not installed" in result.stderr

    def test_no_permission_for_docker(self, env):
        result = run_setup(
            env, *QUICK, FAKE_DOCKER_INFO_ERR="permission denied while trying to connect"
        )

        assert result.returncode == 1
        assert "docker group" in result.stderr

    def test_old_compose(self, env):
        result = run_setup(env, *QUICK, FAKE_COMPOSE_VERSION="2.12.2")

        assert result.returncode == 1
        assert "too old" in result.stderr

    def test_no_buildx(self, env):
        result = run_setup(env, *QUICK, FAKE_NO_BUILDX="1")

        assert result.returncode == 1
        assert "buildx" in result.stderr

    def test_no_internet_in_containers_uses_the_host_network(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK, FAKE_NET_BRIDGE="1")

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["SETUP_NETWORK"] == "host"
        assert "Jobs from a URL" in result.stdout

    def test_no_internet_at_all_stops(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK, FAKE_NET_BRIDGE="1", FAKE_NET_HOST="1")

        assert result.returncode == 1
        assert "cannot reach the internet" in result.stderr
        assert "compose build" not in docker_calls(env)


class TestBuildOutOfMemory:
    def test_oom_build_is_retried_with_lower_jobs(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK, FAKE_BUILD_137="1")

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["BUILD_JOBS"] == "2"
        assert "ran out of memory" in result.stdout
        assert docker_calls(env).count("compose build") == 2

    def test_oom_build_keeps_an_existing_build_jobs_value(self, env):
        (env["project"] / ".env").write_text("BUILD_JOBS=1\n")

        result = run_setup(env, "--backend", "cpu", *QUICK, FAKE_BUILD_137="1")

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["BUILD_JOBS"] == "1"

    def test_oom_build_with_build_jobs_1_does_not_suggest_lowering_again(self, env):
        (env["project"] / ".env").write_text("BUILD_JOBS=1\n")

        result = run_setup(env, "--backend", "cpu", *QUICK, FAKE_BUILD_137="2")

        assert result.returncode == 1
        assert "even with BUILD_JOBS=1" in result.stderr
        assert "set BUILD_JOBS=1" not in result.stderr

    def test_oom_build_that_fails_twice_stops_with_a_hint(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK, FAKE_BUILD_137="2")

        assert result.returncode == 1
        assert "BUILD_JOBS=1" in result.stderr

    def test_other_build_failures_keep_the_message(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK, FAKE_BUILD_FAIL="1")

        assert result.returncode == 1
        assert "The build failed" in result.stderr
        assert read_env(env)["BUILD_JOBS"] == ""
        assert docker_calls(env).count("compose build") == 1


class TestLLM:
    """The LLM is required: Qwen3-14B downloaded, or an API provider."""

    def test_without_questions_downloads_qwen(self, env):
        result = run_setup(env, *WITH_MODELS)

        assert result.returncode == 0, result.stdout + result.stderr
        calls = docker_calls(env)
        assert "--skip-whisper" in calls
        assert "llm_provider.py use-" not in calls

    def test_low_gpu_memory_still_downloads_qwen_without_questions(self, env):
        (env["state"] / "check_gpu").write_text(LOW_MEMORY_CHECK)

        result = run_setup(env, *WITH_MODELS)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "--skip-whisper" in docker_calls(env)
        assert "hours per video" in result.stdout
        assert "put OPENAI_API_KEY or ANTHROPIC_API_KEY into .env" in result.stdout

    def test_api_key_in_env_chooses_that_provider(self, env):
        (env["project"] / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-test\n")

        result = run_setup(env, *WITH_MODELS)

        assert result.returncode == 0, result.stdout + result.stderr
        calls = docker_calls(env)
        assert "llm_provider.py use-api --provider anthropic --model claude-opus-5" in calls
        assert "llm_provider.py check" in calls
        assert "--skip-whisper" not in calls
        assert "the model answered" in result.stdout

    def test_failed_api_check_stops_and_takes_the_choice_back(self, env):
        (env["project"] / ".env").write_text("OPENAI_API_KEY=sk-bad\n")

        result = run_setup(env, *WITH_MODELS, FAKE_LLM_CHECK_FAIL="Error code: 401")

        assert result.returncode == 1
        assert "Error code: 401" in result.stderr
        assert "Fix the key in .env" in result.stderr
        calls = docker_calls(env)
        assert calls.index("llm_provider.py use-api") < calls.index("llm_provider.py use-local")
        assert "--skip-whisper" not in calls

    def test_ready_api_provider_is_kept(self, env):
        set_llm_status(env, provider="openai", model="gpt-test", ready="yes")

        result = run_setup(env, *WITH_MODELS)

        assert result.returncode == 0, result.stdout + result.stderr
        calls = docker_calls(env)
        assert "llm_provider.py use-" not in calls
        assert "llm_provider.py check" not in calls
        assert "--skip-whisper" not in calls
        assert "the openai API, model gpt-test" in result.stdout

    def test_downloaded_qwen_is_checked_without_questions(self, env):
        set_llm_status(env, ready="yes")

        result = run_setup_tty(env, [""], *TTY)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "--skip-whisper" in docker_calls(env)
        assert "already downloaded" in result.stdout
        assert "Your choice" not in result.stderr

    def test_llm_turned_off_in_the_settings_is_turned_on(self, env):
        set_llm_status(env, enabled="no")

        result = run_setup(env, *WITH_MODELS)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "turned off in the settings" in result.stdout
        calls = docker_calls(env)
        assert calls.index("llm_provider.py use-local") < calls.index("--skip-whisper")

    def test_no_models_skips_the_llm(self, env):
        result = run_setup(env, "--backend", "cpu", *QUICK)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "llm_provider.py" not in docker_calls(env)

    def test_choice_1_downloads_qwen(self, env):
        result = run_setup_tty(env, ["", "1"], *TTY)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "--skip-whisper" in docker_calls(env)

    def test_choice_3_stops(self, env):
        result = run_setup_tty(env, ["", "3"], *TTY)

        assert result.returncode == 1
        assert "the LLM is required" in result.stderr
        assert "--skip-whisper" not in docker_calls(env)

    def test_low_gpu_memory_suggests_the_api_provider(self, env):
        (env["state"] / "check_gpu").write_text(LOW_MEMORY_CHECK)

        result = run_setup_tty(env, ["", "3"], *TTY)

        assert result.returncode == 1
        assert "Your choice [2]" in result.stderr

    def test_anthropic_key_goes_into_env_and_the_model_into_the_settings(self, env):
        result = run_setup_tty(env, ["", "2", "2", "sk-ant-test", ""], *TTY)

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["ANTHROPIC_API_KEY"] == "sk-ant-test"
        assert stat.S_IMODE((env["project"] / ".env").stat().st_mode) == 0o600
        calls = docker_calls(env)
        assert "llm_provider.py use-api --provider anthropic --model claude-opus-5\n" in calls
        assert "llm_provider.py check" in calls
        assert "--skip-whisper" not in calls
        # The key is typed without echo and never printed.
        assert "sk-ant-test" not in result.stdout + result.stderr

    def test_openai_compatible_service_asks_for_its_address_and_model(self, env):
        answers = ["", "2", "3", "https://api.example.com/v1", "sk-x", "some-model"]

        result = run_setup_tty(env, answers, *TTY)

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["OPENAI_API_KEY"] == "sk-x"
        assert (
            "use-api --provider openai --model some-model --api-base https://api.example.com/v1"
            in docker_calls(env)
        )

    def test_enter_keeps_the_key_already_in_env(self, env):
        (env["project"] / ".env").write_text(f"WEB_PORT={free_port()}\nOPENAI_API_KEY=sk-old\n")

        result = run_setup_tty(env, ["", "2", "1", "", "gpt-test"], *TTY)

        assert result.returncode == 0, result.stdout + result.stderr
        assert read_env(env)["OPENAI_API_KEY"] == "sk-old"
        assert "use-api --provider openai --model gpt-test" in docker_calls(env)

    def test_failed_check_shows_the_menu_again(self, env):
        answers = ["", "2", "1", "sk-bad", "", "1"]

        result = run_setup_tty(env, answers, *TTY, FAKE_LLM_CHECK_FAIL="Error code: 401")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Error code: 401" in result.stderr
        assert result.stdout.count("so one is required") == 2
        calls = docker_calls(env)
        assert calls.index("use-api") < calls.index("use-local") < calls.index("--skip-whisper")

    def test_failed_check_without_docker_internet_names_the_network(self, env):
        (env["project"] / ".env").write_text(f"WEB_PORT={free_port()}\nSETUP_NETWORK=host\n")
        answers = ["", "2", "1", "sk-x", "", "3"]

        result = run_setup_tty(env, answers, *TTY, FAKE_LLM_CHECK_FAIL="Connection error.")

        assert result.returncode == 1
        assert "cannot reach API providers" in result.stdout
