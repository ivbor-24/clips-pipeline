"""Tests for scripts/dev.sh (just dev / serve / worker / dev-web) and for
`just stop` stopping the dev processes.

The services are small stand-ins (DEV_*_CMD) that start a child of their own,
so the tests see whether whole process groups stop, as uvicorn's reloader and
server or npm and vite must.
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.fake_docker import free_port, make_fake_docker

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")
SERVICES = ("api", "worker", "web")


def service_cmd(state: Path, name: str) -> str:
    """A service with a child process; both PIDs are written to the state dir."""
    return (
        f"bash -c 'sleep 300 & echo $! > {state}/{name}.child; echo $$ > {state}/{name}.pid; wait'"
    )


@pytest.fixture
def env(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "web").mkdir()
    for name in ("scripts/dev.sh", "scripts/stack.sh", "scripts/docker_common.sh"):
        shutil.copy2(REPO / name, project / name)
    bin_dir, state = make_fake_docker(tmp_path)
    variables = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path),
        "FAKE_STATE": str(state),
        "DEV_API_PORT": str(free_port()),
        "DEV_WEB_PORT": str(free_port()),
        **{f"DEV_{s.upper()}_CMD": service_cmd(state, s) for s in SERVICES},
    }
    started = []
    yield {"project": project, "state": state, "bin": bin_dir, "vars": variables, "procs": started}
    for proc in started:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    for pgid_file in (project / ".run").glob("*.pgid"):
        try:
            os.killpg(int(pgid_file.read_text()), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass


def start_dev(env, *services, **extra_vars) -> subprocess.Popen:
    """dev.sh in its own session, as in a terminal; waits until all services run."""
    proc = subprocess.Popen(
        [BASH, str(env["project"] / "scripts" / "dev.sh"), *services],
        cwd=env["project"],
        env={**env["vars"], **extra_vars},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    env["procs"].append(proc)
    wanted = services or SERVICES
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if all((env["state"] / f"{s}.child").exists() for s in wanted):
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    raise AssertionError(f"services did not start: {proc.communicate(timeout=5)[0]}")


def run_script(env, script, *args, **extra_vars):
    return subprocess.run(
        [BASH, str(env["project"] / "scripts" / script), *args],
        cwd=env["project"],
        env={**env["vars"], **extra_vars},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def pids(env, services=SERVICES):
    found = []
    for s in services:
        for suffix in ("pid", "child"):
            found.append(int((env["state"] / f"{s}.{suffix}").read_text()))
    return found


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie is dead too; it only waits for its parent.
    status = Path(f"/proc/{pid}/status")
    return status.exists() and "zombie" not in status.read_text()


def wait_gone(pid_list, timeout=10.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        left = [p for p in pid_list if alive(p)]
        if not left:
            return []
        time.sleep(0.1)
    return [p for p in pid_list if alive(p)]


class TestDev:
    def test_ctrl_c_stops_every_service_with_its_children(self, env):
        dev = start_dev(env)
        running = pids(env)
        assert all(alive(p) for p in running)
        assert sorted(p.name for p in (env["project"] / ".run").iterdir()) == [
            "api.pgid",
            "web.pgid",
            "worker.pgid",
        ]

        os.kill(dev.pid, signal.SIGINT)
        out, _ = dev.communicate(timeout=30)

        assert dev.returncode == 130, out
        assert wait_gone(running) == []
        assert list((env["project"] / ".run").glob("*.pgid")) == []
        assert "Stopped." in out

    def test_one_service_exiting_stops_the_others(self, env):
        state = env["state"]
        dev = start_dev(
            env,
            DEV_WEB_CMD=f"bash -c 'echo $$ > {state}/web.pid; : > {state}/web.child; sleep 1; exit 3'",
        )
        others = pids(env, ("api", "worker"))

        out, _ = dev.communicate(timeout=30)

        assert dev.returncode == 3, out
        assert "web exited (code 3)" in out
        assert wait_gone(others) == []

    def test_just_stop_from_another_terminal(self, env):
        dev = start_dev(env)
        running = pids(env)

        result = run_script(env, "stack.sh", "stop")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Stopping worker (dev mode" in result.stdout
        assert wait_gone(running) == []
        dev.communicate(timeout=30)
        assert dev.returncode is not None

    def test_one_service(self, env):
        dev = start_dev(env, "worker")
        assert [p.name for p in (env["project"] / ".run").iterdir()] == ["worker.pgid"]
        os.kill(dev.pid, signal.SIGINT)
        dev.communicate(timeout=30)
        assert wait_gone(pids(env, ("worker",))) == []


class TestRefusals:
    def test_a_second_worker(self, env):
        # `python3 -m src.worker` of this copy, started in another terminal.
        other = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", "-m", "src.worker"],
            cwd=env["project"],
        )
        try:
            result = run_script(env, "dev.sh", "worker")
        finally:
            other.kill()
            other.wait()

        assert result.returncode == 1
        assert "The worker of this copy already runs" in result.stderr
        assert str(other.pid) in result.stderr

    def test_worker_while_the_docker_installation_runs(self, env):
        (env["state"] / "running").write_text("backend\nfrontend\nworker-cpu\n")

        result = run_script(env, "dev.sh")

        assert result.returncode == 1
        assert "Docker installation runs" in result.stderr
        assert not (env["project"] / ".run").exists()

    def test_taken_port(self, env):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", int(env["vars"]["DEV_API_PORT"])))
            listener.listen()
            result = run_script(env, "dev.sh", "api")

        assert result.returncode == 1
        assert f"Port {env['vars']['DEV_API_PORT']} (api) is taken" in result.stderr


class TestStaleRecords:
    def test_a_recorded_group_of_someone_else_is_left_alone(self, env, tmp_path):
        # After a reboot the PGID in .run/ may belong to an unrelated program.
        stranger = subprocess.Popen(["sleep", "60"], cwd=tmp_path, start_new_session=True)
        run_dir = env["project"] / ".run"
        run_dir.mkdir()
        (run_dir / "worker.pgid").write_text(str(stranger.pid))
        try:
            stop = run_script(env, "stack.sh", "stop")
            assert alive(stranger.pid)
            assert not (run_dir / "worker.pgid").exists()

            dev = start_dev(env, "worker")
            os.kill(dev.pid, signal.SIGINT)
            dev.communicate(timeout=30)
            assert alive(stranger.pid)
        finally:
            stranger.kill()
            stranger.wait()
        assert stop.returncode == 0, stop.stdout + stop.stderr


def test_just_stop_without_docker(env):
    (env["bin"] / "docker").unlink()
    dev = start_dev(env, "api")
    running = pids(env, ("api",))

    result = run_script(env, "stack.sh", "stop")

    assert result.returncode == 0, result.stdout + result.stderr
    assert wait_gone(running) == []
    dev.communicate(timeout=30)
