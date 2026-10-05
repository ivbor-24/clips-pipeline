"""Tests for the Diagnostics page: service, API and the worker's part."""

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.models.worker import WORKER_STATUS_ID, WorkerStatus
from src.api.services import diagnostics


def status(**fields) -> WorkerStatus:
    return WorkerStatus(id=WORKER_STATUS_ID, **fields)


def ago(seconds: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


class TestWorkerState:
    def test_never_started(self):
        assert diagnostics.worker_state(None, None)["state"] == "never_started"

    def test_idle_and_busy(self):
        assert diagnostics.worker_state(status(heartbeat_at=ago(3)), None)["state"] == "idle"
        busy = diagnostics.worker_state(status(heartbeat_at=ago(3)), {"id": 7, "stage": "scoring"})
        assert busy["state"] == "busy"
        assert busy["job"] == {"id": 7, "stage": "scoring"}

    def test_old_heartbeat_is_not_responding(self):
        state = diagnostics.worker_state(status(heartbeat_at=ago(600)), {"id": 7, "stage": None})
        assert state["state"] == "not_responding"

    def test_naive_datetimes_from_sqlite_are_utc(self):
        naive = ago(3).replace(tzinfo=None)
        assert diagnostics.worker_state(status(heartbeat_at=naive), None)["state"] == "idle"


class TestGpuCheck:
    def test_result_and_pending(self):
        result = diagnostics.gpu_check(
            status(
                gpu_check=json.dumps({"backend": "openvino", "ok": True, "results": []}),
                gpu_check_at=ago(60),
                gpu_check_requested_at=ago(5),
            )
        )
        assert result["backend"] == "openvino"
        assert result["pending"] is True

    def test_done_request_is_not_pending(self):
        result = diagnostics.gpu_check(
            status(gpu_check="{}", gpu_check_at=ago(5), gpu_check_requested_at=ago(60))
        )
        assert result["pending"] is False


class TestModels:
    def write_config(self, tmp_path, model_path: str) -> str:
        path = tmp_path / "config.yaml"
        path.write_text(
            "scoring:\n  term_extraction_method: statistical\n  llm:\n    enabled: true\n"
            f"    provider: llama_cpp\n    model_path: {model_path}\n"
            "    model_repo: bartowski/Qwen_Qwen3-14B-GGUF\n"
            "    model_file: Qwen_Qwen3-14B-Q4_K_M.gguf\n"
            "transcription:\n  engine: whisper_cpp\n"
            f"  whisper_cpp_model_path: {tmp_path / 'ggml-large-v3-turbo.bin'}\n"
        )
        return str(path)

    def test_missing_and_wrong_size(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ggml-large-v3-turbo.bin").write_bytes(b"short")
        entries = diagnostics.models(self.write_config(tmp_path, str(tmp_path / "q.gguf")))
        by_purpose = {e["purpose"]: e for e in entries}
        assert by_purpose["LLM"]["state"] == "missing"
        whisper = by_purpose["speech recognition (whisper.cpp)"]
        assert whisper["state"] == "wrong_size"
        assert whisper["size"] == 5
        assert by_purpose["face detection"]["state"] == "missing"
        assert "keywords (KeyBERT)" not in by_purpose  # statistical needs no model

    def test_api_provider(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "config.yaml"
        path.write_text(
            "scoring:\n  llm:\n    enabled: true\n    provider: openai\n    model: gpt\n"
        )
        entries = diagnostics.models(str(path))
        assert entries[0] == {"name": "gpt", "purpose": "LLM", "state": "api:openai"}


class TestFfmpeg:
    def test_version_line(self):
        done = subprocess.CompletedProcess([], 0, stdout="ffmpeg version 7.1\nbuilt with gcc\n")
        with patch.object(diagnostics.subprocess, "run", return_value=done):
            assert diagnostics.ffmpeg() == {"ok": True, "version": "ffmpeg version 7.1"}

    def test_not_installed(self):
        with patch.object(diagnostics.subprocess, "run", side_effect=FileNotFoundError("ffmpeg")):
            assert diagnostics.ffmpeg()["ok"] is False


def test_file_state(tmp_path):
    from src.model_registry import PinnedFile, file_state, verify_file

    data = b"model bytes"
    import hashlib

    pin = PinnedFile(
        repo="r",
        file="m.bin",
        revision="x",
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )
    path = tmp_path / "m.bin"
    assert file_state(path, pin) == "missing"
    path.write_bytes(data)
    assert file_state(path, None) == "unpinned"
    assert file_state(path, pin) == "unverified"
    verify_file(path, pin)
    assert file_state(path, pin) == "verified"
    path.write_bytes(b"other")
    assert file_state(path, pin) == "wrong_size"


@pytest_asyncio.fixture
async def client():
    from src.api.app import app
    from src.api.database import Base, get_db

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    local = {"models": [], "models_error": None, "ffmpeg": {"ok": True, "version": "x"}, "disk": []}
    with patch("src.api.routes.diagnostics._local_checks", return_value=local):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
    app.dependency_overrides.clear()
    await engine.dispose()


class TestApi:
    @pytest.mark.asyncio
    async def test_before_the_worker_ran(self, client):
        data = (await client.get("/api/v1/diagnostics/")).json()
        assert data["worker"]["state"] == "never_started"
        assert data["gpu_check"] is None
        assert data["system"]["version"]
        assert data["ffmpeg"]["version"] == "x"

    @pytest.mark.asyncio
    async def test_check_again_is_pending_until_the_worker_runs_it(self, client):
        response = await client.post("/api/v1/diagnostics/gpu-check")
        assert response.status_code == 202
        data = (await client.get("/api/v1/diagnostics/")).json()
        assert data["gpu_check"]["pending"] is True


class TestWorkerPart:
    @pytest.fixture
    def session_factory(self, tmp_path):
        from src.worker import make_session_factory

        return make_session_factory(f"sqlite:///{tmp_path / 'api.db'}")

    def read(self, session_factory) -> WorkerStatus:
        with session_factory() as session:
            return session.get(WorkerStatus, WORKER_STATUS_ID)

    def test_gpu_check_runs_only_when_requested(self, session_factory):
        from src.worker import Worker, publish_status

        calls = []

        def fake_check():
            calls.append(1)
            return json.dumps({"backend": "cpu", "ok": True, "results": []})

        worker = Worker(session_factory, gpu_check=fake_check)
        worker.check_gpu_if_requested()
        assert calls == []

        with session_factory() as session:
            publish_status(session, gpu_check_requested_at=datetime.now(timezone.utc))
        worker.check_gpu_if_requested()
        assert calls == [1]
        assert json.loads(self.read(session_factory).gpu_check)["backend"] == "cpu"

        worker.check_gpu_if_requested()  # the request is done
        assert calls == [1]

    def test_heartbeat_is_throttled(self, session_factory):
        from src.worker import Worker

        worker = Worker(session_factory, status_interval=3600)
        # Shortly after boot monotonic time is small: the first beat must still be written.
        with patch("src.worker.time.monotonic", return_value=5.0):
            worker.heartbeat()
        assert self.read(session_factory).heartbeat_at is not None
        worker = Worker(session_factory, status_interval=3600)
        worker.heartbeat()
        first = self.read(session_factory).heartbeat_at
        worker.heartbeat()
        assert self.read(session_factory).heartbeat_at == first
        worker.heartbeat(force=True)
        assert self.read(session_factory).heartbeat_at >= first

    def test_run_gpu_check_takes_the_last_json_line(self):
        from src import worker

        out = 'Rusticl warning\n{"backend": "openvino", "ok": true, "results": []}\n'
        with patch.object(
            worker.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, out, "")
        ):
            assert json.loads(worker.run_gpu_check())["backend"] == "openvino"
        with patch.object(
            worker.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "boom")
        ):
            result = json.loads(worker.run_gpu_check())
        assert result["ok"] is False and "boom" in result["error"]


def test_check_gpu_json(tmp_path):
    """check_gpu.py --json prints one JSON line the worker can store."""
    import sys

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "check_gpu.py"),
         "--backend", "cuda", "--libraries-only", "--json"],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["backend"] == "cuda"
    assert data["results"][0]["component"] == "CUDA libraries"
