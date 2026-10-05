"""Tests for scripts/stack.sh: just up / just stop / just status.

The script runs in a copy of the project files it needs, with the fake
`docker` of tests/fake_docker.py; a small HTTP server stands in for the web UI.
"""

import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fake_docker import free_port, make_fake_docker, start_fake_web, stop_fake_web

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")
ALL_RUNNING = "backend\nfrontend\nworker-cpu\n"


@pytest.fixture
def env(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    for name in ("scripts/stack.sh", "scripts/docker_common.sh"):
        shutil.copy2(REPO / name, project / name)
    port = free_port()
    (project / ".env").write_text(f"GPU_BACKEND=cpu\nCOMPOSE_PROFILES=cpu\nWEB_PORT={port}\n")
    bin_dir, state = make_fake_docker(tmp_path)
    variables = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path),
        "FAKE_STATE": str(state),
        "FAKE_WEB_PORT": str(port),
        "STACK_START_TIMEOUT": "4",
    }
    yield {"project": project, "state": state, "port": port, "vars": variables}
    stop_fake_web(state)


def run_stack(env, *args, **extra_vars):
    return subprocess.run(
        [BASH, str(env["project"] / "scripts" / "stack.sh"), *args],
        cwd=env["project"],
        env={**env["vars"], **extra_vars},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def docker_calls(env) -> str:
    return (env["state"] / "log").read_text()


class TestUp:
    def test_starts_and_waits_for_the_web_ui(self, env):
        result = run_stack(env, "up")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "compose up -d" in docker_calls(env)
        assert f"Running: http://127.0.0.1:{env['port']} (version 9.9.9" in result.stdout

    def test_failed_start_shows_the_error(self, env):
        error = (
            "src.api.migrate.MigrationError: The job database is at revision 9999, which "
            "this version of the code does not know\n"
        )
        (env["state"] / "logs").write_text(
            "INFO:     Started server process\n"
            + error
            + "    raise MigrationError(\n"
            + error  # the worker fails on the same
            + "ERROR:    Application startup failed. Exiting.\n"
        )
        (env["state"] / "restarting").write_text("0123456789ab\n")
        result = run_stack(env, "up", FAKE_WEB_PORT="", STACK_START_TIMEOUT="2")

        assert result.returncode == 1
        assert "Last errors in the logs:" in result.stderr
        assert result.stderr.count("revision 9999") == 1
        assert "raise MigrationError" not in result.stderr
        assert "does not answer" in result.stderr
        assert "kept failing at start; they are stopped" in result.stdout
        assert "compose --profile cpu --profile cuda --profile openvino stop" in docker_calls(env)

    def test_already_running(self, env):
        (env["state"] / "running").write_text(ALL_RUNNING)
        web = start_fake_web(env["state"], env["port"])
        try:
            result = run_stack(env, "up")
        finally:
            web.terminate()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Already running" in result.stdout
        assert "version 9.9.9" in result.stdout
        assert "Restarted" not in result.stdout

    def test_running_on_an_older_image_restarts(self, env):
        # ./setup.sh after `git pull` rebuilt the image while the stack ran:
        # compose recreates the containers.
        (env["state"] / "running").write_text(ALL_RUNNING)

        result = run_stack(env, "up", FAKE_RECREATE="1")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Restarted with the new image or settings: backend frontend worker-cpu" in (
            result.stdout
        )
        assert f"Running: http://127.0.0.1:{env['port']} (version 9.9.9" in result.stdout

    def test_without_env_asks_for_setup(self, env):
        (env["project"] / ".env").unlink()

        result = run_stack(env, "up")

        assert result.returncode == 1
        assert "./setup.sh" in result.stderr

    def test_image_not_built(self, env):
        result = run_stack(env, "up", FAKE_IMAGE_MISSING="1")

        assert result.returncode == 1
        assert "not built yet" in result.stderr

    def test_port_taken_by_another_program(self, env):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", env["port"]))
            listener.listen()
            result = run_stack(env, "up")

        assert result.returncode == 1
        assert f"Port {env['port']} is taken by another program" in result.stderr
        assert "WEB_PORT" in result.stderr
        assert "compose up" not in docker_calls(env)

    def test_another_copy_is_not_stopped_without_asking(self, env):
        (env["state"] / "other_copies").write_text("other\t/home/me/other-copy\n")

        result = run_stack(env, "up", "--yes")

        assert result.returncode == 1
        assert "/home/me/other-copy" in result.stdout
        assert "Stop the other copy first" in result.stderr
        assert "compose up" not in docker_calls(env)

    def test_worker_of_this_copy_outside_docker(self, env):
        # Its command line is that of `python3 -m src.worker` (`just worker`).
        dev = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", "-m", "src.worker"],
            cwd=env["project"],
        )
        try:
            result = run_stack(env, "up")
        finally:
            dev.kill()
            dev.wait()

        assert result.returncode == 1
        assert "runs outside Docker" in result.stderr
        assert str(dev.pid) in result.stderr

    def test_a_command_line_that_only_mentions_the_worker_is_not_one(self, env):
        # For example the shell that runs `just up && python -m src.worker`.
        shell = subprocess.Popen(
            [BASH, "-c", "sleep 30; echo python -m src.worker"], cwd=env["project"]
        )
        try:
            result = run_stack(env, "up")
        finally:
            shell.kill()
            shell.wait()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "outside Docker" not in result.stdout + result.stderr

    def test_worker_of_another_backend_is_stopped(self, env):
        (env["state"] / "running").write_text("backend\nfrontend\nworker-openvino\n")

        result = run_stack(env, "up")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "stop worker-openvino" in docker_calls(env)


class TestStop:
    def test_stops_every_profile(self, env):
        (env["state"] / "running").write_text(ALL_RUNNING)

        result = run_stack(env, "stop")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "--profile cpu --profile cuda --profile openvino stop" in docker_calls(env)
        assert "Stopped" in result.stdout

    def test_not_running(self, env):
        result = run_stack(env, "stop")

        assert result.returncode == 0
        assert "not running" in result.stdout
        assert " stop" not in docker_calls(env)

    def test_reports_another_copy(self, env):
        (env["state"] / "other_copies").write_text("other\t/home/me/other-copy\n")

        result = run_stack(env, "stop")

        assert "/home/me/other-copy" in result.stdout


class TestStatus:
    def test_running(self, env):
        (env["state"] / "running").write_text(ALL_RUNNING)
        web = start_fake_web(env["state"], env["port"])
        try:
            result = run_stack(env, "status")
        finally:
            web.terminate()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "worker-cpu" in result.stdout
        assert f"http://127.0.0.1:{env['port']} (version 9.9.9)" in result.stdout

    def test_nothing_yet(self, env):
        result = run_stack(env, "status")

        assert result.returncode == 0
        assert "no containers yet" in result.stdout
        assert "not running" in result.stdout


def test_unknown_command_prints_usage(env):
    result = run_stack(env, "restart")

    assert result.returncode == 2
    assert "Usage" in result.stderr
