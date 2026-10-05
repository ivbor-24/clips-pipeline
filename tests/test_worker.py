"""Tests for the job worker (src/worker.py).

A file-backed SQLite database stands in for data/api.db. Supervision tests
spawn real short-lived processes instead of the pipeline, so killing a job's
process group is tested for real.
"""

import json
import os
import signal
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest
from sqlalchemy import select

from src.api.models.job import Job, JobEvent, JobStatus
from src.worker import (
    Worker,
    claim_next_job,
    make_session_factory,
    recover_interrupted_jobs,
    run_job,
    sync_database_url,
)


@pytest.fixture
def session_factory(tmp_path):
    return make_session_factory(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")


def add_job(session_factory, tmp_path, status=JobStatus.QUEUED, **fields) -> int:
    with session_factory() as session:
        job = Job(
            user_id=1,
            status=status,
            input_source="/videos/lecture.mp4",
            work_dir=str(tmp_path / f"job_{status.value}_{len(os.listdir(tmp_path))}"),
            **fields,
        )
        session.add(job)
        session.commit()
        return job.id


def get_job(session_factory, job_id) -> Job:
    with session_factory() as session:
        return session.get(Job, job_id)


def events(session_factory, job_id) -> list:
    with session_factory() as session:
        rows = session.scalars(
            select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id)
        ).all()
        return [(e.event_type, json.loads(e.data)) for e in rows]


def spawn_python(code: str):
    """A spawn function that runs `code` in a new process group, like the worker does."""

    def spawn(job_id):
        return subprocess.Popen([sys.executable, "-c", code], start_new_session=True)

    return spawn


def group_is_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


class TestDatabase:
    def test_sync_url(self):
        assert sync_database_url("sqlite+aiosqlite:///./data/api.db") == "sqlite:///./data/api.db"

    def test_uses_wal(self, session_factory):
        with session_factory() as session:
            mode = session.connection().exec_driver_sql("PRAGMA journal_mode").scalar()
        assert mode == "wal"


class TestQueue:
    def test_claims_oldest_queued_job(self, session_factory, tmp_path):
        add_job(session_factory, tmp_path, status=JobStatus.COMPLETED)
        first = add_job(session_factory, tmp_path)
        add_job(session_factory, tmp_path)

        with session_factory() as session:
            assert claim_next_job(session) == first
        job = get_job(session_factory, first)
        assert job.status == JobStatus.RUNNING
        assert job.attempts == 1
        assert job.started_at is not None
        assert job.heartbeat_at is not None

    def test_empty_queue(self, session_factory, tmp_path):
        add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        with session_factory() as session:
            assert claim_next_job(session) is None

    def test_recover_requeues_running_jobs(self, session_factory, tmp_path):
        running = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        cancelling = add_job(
            session_factory, tmp_path, status=JobStatus.RUNNING, cancel_requested=True
        )
        done = add_job(session_factory, tmp_path, status=JobStatus.COMPLETED)

        with session_factory() as session:
            assert recover_interrupted_jobs(session) == 1

        assert get_job(session_factory, running).status == JobStatus.QUEUED
        assert events(session_factory, running) == [
            ("job_requeued", {"job_id": running, "reason": "restart"})
        ]
        assert get_job(session_factory, cancelling).status == JobStatus.CANCELLED
        assert get_job(session_factory, done).status == JobStatus.COMPLETED


class TestSupervise:
    def test_child_that_records_success(self, session_factory, tmp_path):
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)

        def spawn(jid):
            with session_factory() as session:
                session.get(Job, jid).status = JobStatus.COMPLETED
                session.commit()
            return subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)

        worker = Worker(session_factory, spawn=spawn, heartbeat_interval=0.1)
        assert worker.supervise(job_id) == JobStatus.COMPLETED

    def test_child_that_dies_fails_the_job(self, session_factory, tmp_path):
        """A crash (segfault, OOM kill) leaves the job running; the worker fails it."""
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        worker = Worker(
            session_factory, spawn=spawn_python("import sys; sys.exit(3)"), heartbeat_interval=0.1
        )

        assert worker.supervise(job_id) == JobStatus.FAILED
        job = get_job(session_factory, job_id)
        assert "exit code 3" in job.error_message
        assert events(session_factory, job_id)[-1][0] == "job_failed"

    def test_cancel_kills_the_whole_process_group(self, session_factory, tmp_path):
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        # The child starts a grandchild, as the pipeline starts ffmpeg.
        code = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)"
        )
        started = []

        def spawn(jid):
            proc = spawn_python(code)(jid)
            started.append(proc)
            with session_factory() as session:
                session.get(Job, jid).cancel_requested = True
                session.commit()
            return proc

        worker = Worker(session_factory, spawn=spawn, heartbeat_interval=0.1)
        assert worker.supervise(job_id) == JobStatus.CANCELLED

        assert group_is_gone(started[0].pid)
        job = get_job(session_factory, job_id)
        assert job.completed_at is not None
        assert events(session_factory, job_id)[-1][0] == "job_cancelled"

    def test_stop_puts_the_job_back_in_the_queue(self, session_factory, tmp_path):
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        worker = Worker(
            session_factory,
            spawn=spawn_python("import time; time.sleep(60)"),
            heartbeat_interval=0.1,
        )
        worker.stopping = True

        assert worker.supervise(job_id) == JobStatus.QUEUED
        assert events(session_factory, job_id)[-1] == (
            "job_requeued",
            {"job_id": job_id, "reason": "stop"},
        )

    def test_heartbeat_is_updated(self, session_factory, tmp_path):
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        worker = Worker(
            session_factory,
            spawn=spawn_python("import time; time.sleep(0.5)"),
            heartbeat_interval=0.1,
        )
        worker.supervise(job_id)
        assert get_job(session_factory, job_id).heartbeat_at is not None

    def test_run_once_takes_one_job(self, session_factory, tmp_path):
        first = add_job(session_factory, tmp_path)
        second = add_job(session_factory, tmp_path)
        worker = Worker(session_factory, spawn=spawn_python("pass"), heartbeat_interval=0.1)

        assert worker.run_once() is True
        # The child did not record a result, so the job counts as crashed.
        assert get_job(session_factory, first).status == JobStatus.FAILED
        assert get_job(session_factory, second).status == JobStatus.QUEUED

    def test_sigterm_stops_the_worker(self, session_factory):
        worker = Worker(session_factory)
        worker.request_stop(signal.SIGTERM, None)
        assert worker.stopping is True


class TestRunJob:
    def test_success_records_stages(self, session_factory, tmp_path):
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)

        def fake_pipeline(config, input_source, dry_run, resume, force, progress_callback):
            assert resume is True and force is False
            assert input_source == "/videos/lecture.mp4"
            progress_callback("ingestion", "skipped", None)
            progress_callback("transcription", "started", None)
            assert get_job(session_factory, job_id).current_stage == "transcription"
            progress_callback("transcription", "completed", 100.0)

        with patch("src.pipeline.run_pipeline", side_effect=fake_pipeline):
            assert run_job(session_factory, job_id) == JobStatus.COMPLETED

        job = get_job(session_factory, job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert [e[0] for e in events(session_factory, job_id)] == [
            "job_started",
            "stage_skipped",
            "stage_started",
            "stage_completed",
            "job_completed",
        ]
        assert events(session_factory, job_id)[3][1]["progress"] == 100.0

    def test_pipeline_error_fails_the_job(self, session_factory, tmp_path):
        from src.pipeline import PipelineError

        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        with patch("src.pipeline.run_pipeline", side_effect=PipelineError("no audio")):
            assert run_job(session_factory, job_id) == JobStatus.FAILED

        job = get_job(session_factory, job_id)
        assert job.error_message == "no audio"
        assert events(session_factory, job_id)[-1] == (
            "job_failed",
            {"job_id": job_id, "error": "no audio"},
        )
        assert (tmp_path / job.work_dir / "artifacts" / "logs" / "job.log").exists()


class TestOrphanedChild:
    """A worker killed with SIGKILL leaves its child running the job."""

    def _orphan(self, job_id):
        # Same shape of command line as the real child: ... --run-job <id>
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", "--run-job", str(job_id)],
            start_new_session=True,
        )
        # A real orphan is reaped by init; reap this one the same way.
        threading.Thread(target=proc.wait, daemon=True).start()
        return proc

    def test_recovery_stops_the_orphan_before_requeueing(self, session_factory, tmp_path):
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        orphan = self._orphan(job_id)
        with session_factory() as session:
            session.get(Job, job_id).process_group = orphan.pid
            session.commit()

        with session_factory() as session:
            assert recover_interrupted_jobs(session, grace_sec=5) == 1

        assert orphan.wait(timeout=5) is not None
        job = get_job(session_factory, job_id)
        assert job.status == JobStatus.QUEUED
        assert job.process_group is None

    def test_unrelated_process_with_same_id_is_left_alone(self, session_factory, tmp_path):
        """After a reboot the stored group id may belong to another program."""
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        with session_factory() as session:
            session.get(Job, job_id).process_group = unrelated.pid
            session.commit()
        try:
            with session_factory() as session:
                recover_interrupted_jobs(session, grace_sec=1)
            assert unrelated.poll() is None
        finally:
            unrelated.kill()
            unrelated.wait()

    def test_supervise_records_and_clears_the_group(self, session_factory, tmp_path):
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        seen = []

        def spawn(jid):
            proc = spawn_python("import time; time.sleep(0.3)")(jid)
            seen.append(proc.pid)
            return proc

        worker = Worker(session_factory, spawn=spawn, heartbeat_interval=0.1)
        # Read the stored group while the child runs, from the heartbeat loop.
        original_finish = worker._finish

        def finish(*args):
            seen.append(get_job(session_factory, job_id).process_group)
            return original_finish(*args)

        worker._finish = finish
        worker.supervise(job_id)
        assert seen[0] == seen[1]
        assert get_job(session_factory, job_id).process_group is None
