"""Tests for the shutdown from the web UI and after idle time."""

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api import shutdown
from src.api.models.job import Job, JobStatus
from src.api.models.worker import WORKER_STATUS_ID, WorkerStatus
from src.worker import Worker, make_session_factory, publish_status

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """A new API process each test: no shutdown yet, a page just called."""
    monkeypatch.setattr(shutdown, "state", shutdown._State())
    monkeypatch.setattr(shutdown, "EXIT_DELAY_SEC", 0)


# ---- worker ----------------------------------------------------------------


@pytest.fixture
def session_factory(tmp_path):
    return make_session_factory(f"sqlite:///{tmp_path / 'api.db'}")


def add_job(session_factory, tmp_path, status=JobStatus.QUEUED) -> int:
    with session_factory() as session:
        job = Job(
            user_id=1,
            status=status,
            input_source="/videos/lecture.mp4",
            work_dir=str(tmp_path / f"job_{time.monotonic_ns()}"),
        )
        session.add(job)
        session.commit()
        return job.id


def job_status(session_factory, job_id) -> JobStatus:
    with session_factory() as session:
        return session.get(Job, job_id).status


def worker_status(session_factory) -> WorkerStatus:
    with session_factory() as session:
        return session.get(WorkerStatus, WORKER_STATUS_ID)


def spawn_sleep(seconds: float):
    def spawn(job_id):
        code = f"import time; time.sleep({seconds})"
        return subprocess.Popen([sys.executable, "-c", code], start_new_session=True)

    return spawn


def start_worker(session_factory, spawn) -> tuple:
    worker = Worker(
        session_factory,
        spawn=spawn,
        poll_interval=0.05,
        heartbeat_interval=0.1,
        gpu_check=lambda: '{"ok": true, "results": []}',
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    return worker, thread


def wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def request(session_factory, mode):
    with session_factory() as session:
        publish_status(
            session, shutdown_requested_at=datetime.now(timezone.utc), shutdown_mode=mode
        )


class TestWorkerShutdown:
    @pytest.mark.parametrize(
        "mode, expected", [("requeue", JobStatus.QUEUED), ("cancel", JobStatus.CANCELLED)]
    )
    def test_running_job(self, session_factory, tmp_path, mode, expected):
        job_id = add_job(session_factory, tmp_path)
        worker, thread = start_worker(session_factory, spawn_sleep(60))
        assert wait_for(lambda: job_status(session_factory, job_id) == JobStatus.RUNNING)

        request(session_factory, mode)

        thread.join(timeout=20)
        assert not thread.is_alive()
        assert job_status(session_factory, job_id) == expected
        assert worker_status(session_factory).stopped_at is not None

    def test_wait_lets_the_job_finish(self, session_factory, tmp_path):
        job_id = add_job(session_factory, tmp_path)
        queued = add_job(session_factory, tmp_path)
        worker, thread = start_worker(session_factory, spawn_sleep(1.0))
        assert wait_for(lambda: job_status(session_factory, job_id) == JobStatus.RUNNING)

        request(session_factory, "wait")

        thread.join(timeout=20)
        assert not thread.is_alive()
        # The child ran to its natural end (it records no result, so "failed").
        with session_factory() as session:
            job = session.get(Job, job_id)
        assert job.status == JobStatus.FAILED
        assert "exit code 0" in job.error_message
        # The next job waits for the next start.
        assert job_status(session_factory, queued) == JobStatus.QUEUED

    def test_old_request_does_not_stop_a_new_worker(self, session_factory):
        request(session_factory, "requeue")  # before this worker's start (`just up` after it)
        worker, thread = start_worker(session_factory, spawn_sleep(60))
        time.sleep(0.5)
        assert thread.is_alive()
        worker.request_stop()
        thread.join(timeout=10)


class TestWorkerStopped:
    def since(self):
        return datetime.now(timezone.utc) - timedelta(seconds=30)

    def test_no_worker(self):
        assert shutdown.worker_stopped(None, self.since())

    def test_stopped_after_the_request(self):
        status = WorkerStatus(
            heartbeat_at=datetime.now(timezone.utc), stopped_at=datetime.now(timezone.utc)
        )
        assert shutdown.worker_stopped(status, self.since())

    def test_still_running(self):
        status = WorkerStatus(heartbeat_at=datetime.now(timezone.utc))
        assert not shutdown.worker_stopped(status, self.since())

    def test_silent_worker_counts_as_stopped(self):
        status = WorkerStatus(heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=5))
        assert shutdown.worker_stopped(status, self.since())


# ---- API -------------------------------------------------------------------


@pytest_asyncio.fixture
async def api(tmp_path):
    from src.api.app import app
    from src.api.database import Base, get_db

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with patch("src.api.routes.system.async_session", factory):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, factory
    app.dependency_overrides.clear()
    # A shutdown the test started keeps polling the database for the worker:
    # stop it before the engine goes. Otherwise the event loop's teardown could
    # wait forever on its open aiosqlite connection (CI hung here for 25 min).
    task = shutdown.state.finish_task
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await engine.dispose()


class TestApi:
    @pytest.mark.asyncio
    async def test_shutdown_request(self, api):
        client, factory = api
        assert (await client.get("/api/v1/health")).json()["status"] == "ok"
        status = (await client.get("/api/v1/system/status")).json()
        assert status["state"] == "running"
        assert status["auto_shutdown_min"] == 60

        response = await client.post("/api/v1/system/shutdown", json={"mode": "cancel"})
        assert response.status_code == 202
        requested = response.json()["requested_at"]

        assert (await client.get("/api/v1/health")).json()["status"] == "shutting_down"
        status = (await client.get("/api/v1/system/status")).json()
        assert status["state"] == "shutting_down"
        assert status["shutdown"]["mode"] == "cancel"
        assert status["shutdown"]["reason"] == "user"
        async with factory() as db:
            row = await db.get(WorkerStatus, WORKER_STATUS_ID)
        assert row.shutdown_mode == "cancel"

        # A second click changes nothing.
        again = await client.post("/api/v1/system/shutdown", json={"mode": "wait"})
        assert again.json()["requested_at"] == requested

    @pytest.mark.asyncio
    async def test_unknown_mode(self, api):
        client, _ = api
        response = await client.post("/api/v1/system/shutdown", json={"mode": "now"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_api_exits_once_the_worker_stopped(self, api):
        client, _ = api
        with patch("src.api.serve.request_exit", return_value=True) as exit_:
            await client.post("/api/v1/system/shutdown", json={"mode": "requeue"})
            for _ in range(50):  # no worker here: it counts as stopped
                if exit_.called:
                    break
                await asyncio.sleep(0.05)
        exit_.assert_called_once()

    @pytest.mark.asyncio
    async def test_page_requests_count_as_activity_health_checks_do_not(self, api):
        client, _ = api
        shutdown.state.last_activity -= 3600
        await client.get("/api/v1/health")
        assert shutdown.idle_minutes() > 59
        await client.get("/api/v1/system/status")
        assert shutdown.idle_minutes() < 1


class TestIdleWatchdog:
    async def run_watchdog(self, factory, minutes=60):
        with patch.object(shutdown, "IDLE_CHECK_SEC", 0.01):
            task = asyncio.create_task(shutdown.idle_watchdog(factory, minutes))
            await asyncio.sleep(0.2)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_idle_without_jobs_shuts_down(self, api):
        _, factory = api
        shutdown.state.last_activity -= 61 * 60
        await self.run_watchdog(factory)
        assert shutdown.state.requested_at is not None
        async with factory() as db:
            row = await db.get(WorkerStatus, WORKER_STATUS_ID)
        assert row.shutdown_reason == "idle"

    @pytest.mark.asyncio
    async def test_queued_job_keeps_it_on(self, api):
        _, factory = api
        async with factory() as db:
            db.add(Job(user_id=1, status=JobStatus.QUEUED, input_source="x", work_dir="w"))
            await db.commit()
        shutdown.state.last_activity -= 61 * 60
        await self.run_watchdog(factory)
        assert shutdown.state.requested_at is None

    @pytest.mark.asyncio
    async def test_recent_page_keeps_it_on(self, api):
        _, factory = api
        await self.run_watchdog(factory)
        assert shutdown.state.requested_at is None

    @pytest.mark.asyncio
    async def test_zero_turns_it_off(self, api):
        _, factory = api
        shutdown.state.last_activity -= 600 * 60
        await self.run_watchdog(factory, minutes=0)
        assert shutdown.state.requested_at is None


def test_serve_request_exit():
    from src.api import serve

    assert serve.request_exit() is False  # not started through serve.main()

    class FakeServer:
        should_exit = False

    with patch.object(serve, "_server", FakeServer()):
        assert serve.request_exit() is True
        assert serve._server.should_exit is True


def test_api_exits_with_code_0_after_shutdown(tmp_path):
    """The real thing: python -m src.api.serve, a shutdown request, exit code 0
    (uvicorn stopped by SIGTERM would give 143 and Docker would restart it)."""
    import json
    import socket
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {
        **os.environ,
        "API_HOST": "127.0.0.1",
        "API_PORT": str(port),
        "API_DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'api.db'}",
        "API_PASSWORD": "",
        "API_IDLE_SHUTDOWN_MIN": "0",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.api.serve"],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{port}/api/v1"
        assert wait_for(lambda: _get(f"{url}/health") is not None, timeout=30)
        req = urllib.request.Request(
            f"{url}/system/shutdown",
            data=json.dumps({"mode": "requeue"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        assert "shutting_down" in _get(f"{url}/health")
        # No worker: it counts as stopped; then EXIT_DELAY_SEC and exit.
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def _get(url):
    import urllib.request

    try:
        return urllib.request.urlopen(url, timeout=2).read().decode()
    except OSError:
        return None


# ---- the web UI container (scripts/web_entrypoint.sh) ------------------------

FAKE_NGINX = """#!/bin/sh
# `nginx -g ...` runs until `nginx -s quit`.
if [ "$1" = "-s" ]; then
    echo quit >> "$STATE/nginx.log"
    kill "$(cat "$STATE/nginx.pid")"
    exit 0
fi
echo $$ > "$STATE/nginx.pid"
exec sleep 300
"""

# Answers like the API's health endpoint: from the file "health", nothing if absent.
FAKE_WGET = """#!/bin/sh
[ -f "$STATE/health" ] || exit 1
cat "$STATE/health"
"""


@pytest.fixture
def web(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in (("nginx", FAKE_NGINX), ("wget", FAKE_WGET)):
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    for tool in ("sh", "cat", "kill", "sleep", "echo"):
        real = shutil.which(tool)
        if real and not (bin_dir / tool).exists():
            (bin_dir / tool).symlink_to(real)
    state = tmp_path / "state"
    state.mkdir()
    env = {"PATH": str(bin_dir), "STATE": str(state), "WATCH_INTERVAL_SEC": "0.1"}
    proc = subprocess.Popen(
        [shutil.which("sh"), str(REPO / "scripts" / "web_entrypoint.sh")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    yield proc, state
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)


class TestWebEntrypoint:
    def test_stops_after_the_api_shut_down(self, web):
        proc, state = web
        (state / "health").write_text('{"status": "shutting_down"}')
        time.sleep(0.4)
        (state / "health").unlink()  # the API exited
        assert proc.wait(timeout=10) == 0
        assert (state / "nginx.log").read_text() == "quit\n"

    def test_an_api_restart_does_not_stop_the_web_ui(self, web):
        proc, state = web
        (state / "health").write_text('{"status": "ok"}')
        time.sleep(0.3)
        (state / "health").unlink()  # restarted for an update, or not up yet
        time.sleep(0.8)
        assert proc.poll() is None
        assert not (state / "nginx.log").exists()
