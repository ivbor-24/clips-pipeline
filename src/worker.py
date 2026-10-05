"""
Job worker — runs queued jobs one at a time, each in its own process.

The API only puts jobs in the queue (the ``jobs`` table, status ``queued``).
This worker takes the oldest queued job and runs the pipeline for it in a
child process with its own process group. A separate process means:

- restarting the API (or ``uvicorn --reload``) does not kill running jobs;
- cancelling a job really stops it: the whole process group is killed, and
  the RAM and VRAM it held are freed with it;
- only one job uses the GPU at a time.

The worker and the child talk to the API only through the database:
``jobs.status`` / ``current_stage`` / ``heartbeat_at`` / ``cancel_requested``
and the ``job_events`` table that the API streams over SSE.

When the worker starts, jobs left ``running`` by a previous worker (crash,
reboot) go back to the queue and resume from their last artifact. When the
worker is stopped (SIGTERM, Ctrl+C), the job it was running goes back to the
queue too. Run one worker per database.

Inputs:
- data/api.db (``API_DATABASE_URL``): queued jobs, cancel requests

Outputs:
- job status and ``job_events`` rows
- <work_dir>/artifacts/logs/job.log and the pipeline artifacts of each job
- scratch directories of a job (src.temp_files) removed when its process ends

Usage:
    python -m src.worker                 # run the queue
    python -m src.worker --run-job 12    # run one job in this process (used
                                         # by the worker for its children)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import structlog
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker

from src import __version__
from src.api.db_base import DatabaseSettings, enable_sqlite_concurrency, sync_database_url
from src.api.models.job import Job, JobEvent, JobStatus, JobType
from src.api.models.worker import WORKER_STATUS_ID, WorkerStatus
from src.temp_files import remove_job_temp_files, remove_legacy_tmp_files

logger = structlog.get_logger("worker")

POLL_INTERVAL_SEC = 2.0
HEARTBEAT_INTERVAL_SEC = 2.0
STOP_GRACE_SEC = 10.0
# The worker's own heartbeat in worker_status (the Diagnostics page).
WORKER_HEARTBEAT_SEC = 10.0
# check_gpu.py runs whisper-cli on a second of silence: it loads the 1.6 GB model.
GPU_CHECK_TIMEOUT_SEC = 600
ROOT = Path(__file__).resolve().parent.parent


class WorkerError(Exception):
    """Custom exception for worker errors."""

    pass


def make_session_factory(database_url: Optional[str] = None) -> sessionmaker:
    """Session factory for the job database, after bringing it to the newest schema."""
    from src.api.migrate import upgrade_database

    url = sync_database_url(database_url or DatabaseSettings().database_url)
    upgrade_database(url)
    engine = create_engine(url)
    enable_sqlite_concurrency(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def add_event(session: Session, job_id: int, event_type: str, data: Optional[dict] = None) -> None:
    """Record a job event for the SSE stream (committed by the caller)."""
    session.add(JobEvent(job_id=job_id, event_type=event_type, data=json.dumps(data or {})))


# --------------------------------------------------------------------------
# Queue operations (worker process)
# --------------------------------------------------------------------------


def is_job_process_group(pgid: int, job_id: int) -> bool:
    """True if ``pgid`` is still the process group of this job's child.

    Checks the group leader's command line, so a process group id reused after
    a reboot by an unrelated program is never mistaken for the job. Linux only
    (reads /proc); elsewhere it returns False and nothing is stopped.
    """
    try:
        cmdline = Path(f"/proc/{pgid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return b"--run-job" in cmdline and str(job_id).encode() in cmdline


def stop_process_group(
    pgid: int, grace_sec: float = STOP_GRACE_SEC, leader: Optional[subprocess.Popen] = None
) -> None:
    """SIGTERM a process group, then SIGKILL it if it outlives the grace period.

    Args:
        pgid: Process group id.
        grace_sec: How long to wait after SIGTERM.
        leader: The group leader when it is our own child: it has to be reaped,
            or as a zombie it keeps the group "alive".
    """
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if leader is not None:
            leader.poll()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    logger.warning("job_process_kill", pgid=pgid, grace_sec=grace_sec)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def recover_interrupted_jobs(session: Session, grace_sec: float = STOP_GRACE_SEC) -> int:
    """Put jobs left running by a stopped or crashed worker back in the queue.

    If the previous worker was killed alone (``kill -9``), its child may still
    be running the job: it is stopped first, so the job never runs twice on the
    GPU. A job whose cancellation was requested is cancelled instead. Only safe
    with one worker per database, which is how the system runs.

    Returns:
        Number of jobs put back in the queue.
    """
    jobs = session.scalars(select(Job).where(Job.status == JobStatus.RUNNING)).all()
    requeued = 0
    for job in jobs:
        if job.process_group and is_job_process_group(job.process_group, job.id):
            logger.warning("orphaned_job_process_stopped", job_id=job.id, pgid=job.process_group)
            stop_process_group(job.process_group, grace_sec)
        job.process_group = None
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            job.completed_at = _now()
            add_event(session, job.id, "job_cancelled", {"job_id": job.id})
        else:
            job.status = JobStatus.QUEUED
            add_event(session, job.id, "job_requeued", {"job_id": job.id, "reason": "restart"})
            requeued += 1
    session.commit()
    if jobs:
        logger.info("interrupted_jobs_recovered", requeued=requeued, total=len(jobs))
    return requeued


def publish_status(session: Session, **fields) -> None:
    """Update the worker_status row the Diagnostics page reads."""
    status = session.get(WorkerStatus, WORKER_STATUS_ID)
    if status is None:
        status = WorkerStatus(id=WORKER_STATUS_ID)
        session.add(status)
    for name, value in fields.items():
        setattr(status, name, value)
    session.commit()


def run_gpu_check() -> str:
    """scripts/check_gpu.py --json in a subprocess: its JSON, or a failed result."""
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_gpu.py"), "--json"],
            capture_output=True,
            text=True,
            timeout=GPU_CHECK_TIMEOUT_SEC,
            cwd=ROOT,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return json.dumps({"backend": None, "ok": False, "error": str(e), "results": []})
    # The JSON is the last line; libraries may print before it.
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                json.loads(line)
                return line
            except json.JSONDecodeError:
                break
    error = (proc.stderr or proc.stdout or "no output")[-500:]
    return json.dumps({"backend": None, "ok": False, "error": error, "results": []})


def claim_next_job(session: Session) -> Optional[int]:
    """Mark the oldest queued job as running and return its id.

    The conditional UPDATE makes the claim atomic: if another process took or
    cancelled the job in between, nothing is updated and None is returned.
    """
    job_id = session.scalar(
        select(Job.id).where(Job.status == JobStatus.QUEUED).order_by(Job.id).limit(1)
    )
    if job_id is None:
        return None
    result = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == JobStatus.QUEUED)
        .values(
            status=JobStatus.RUNNING,
            started_at=_now(),
            completed_at=None,
            error_message=None,
            heartbeat_at=_now(),
            attempts=Job.attempts + 1,
        )
    )
    session.commit()
    return job_id if result.rowcount == 1 else None


def _kill_group(proc: subprocess.Popen, grace_sec: float = STOP_GRACE_SEC) -> None:
    """Stop the child and everything it started (ffmpeg, whisper-cli)."""
    stop_process_group(proc.pid, grace_sec, leader=proc)
    proc.wait()


def default_spawn(job_id: int) -> subprocess.Popen:
    """Start ``python -m src.worker --run-job <id>`` in a new process group."""
    return subprocess.Popen(
        [sys.executable, "-m", "src.worker", "--run-job", str(job_id)],
        start_new_session=True,
    )


class Worker:
    """Takes queued jobs one by one and supervises the child that runs each."""

    def __init__(
        self,
        session_factory: sessionmaker,
        spawn: Callable[[int], subprocess.Popen] = default_spawn,
        poll_interval: float = POLL_INTERVAL_SEC,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SEC,
        gpu_check: Callable[[], str] = run_gpu_check,
        status_interval: float = WORKER_HEARTBEAT_SEC,
    ):
        self.session_factory = session_factory
        self.spawn = spawn
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.gpu_check = gpu_check
        self.status_interval = status_interval
        # None: never written. Not 0.0: monotonic time starts near zero at boot.
        self._last_status: Optional[float] = None
        self.started_at: Optional[datetime] = None
        self.stopping = False

    def request_stop(self, *_args) -> None:
        logger.info("worker_stop_requested")
        self.stopping = True

    def heartbeat(self, force: bool = False) -> None:
        """Tell the Diagnostics page the worker is alive (at most every status_interval)."""
        now = time.monotonic()
        if (
            not force
            and self._last_status is not None
            and now - self._last_status < self.status_interval
        ):
            return
        self._last_status = now
        with self.session_factory() as session:
            publish_status(session, heartbeat_at=_now())

    def check_gpu_if_requested(self, force: bool = False) -> None:
        """Run the GPU self-check when the API asked for it. Only between jobs:
        it loads whisper.cpp's model, which would compete with a job for VRAM."""
        with self.session_factory() as session:
            status = session.get(WorkerStatus, WORKER_STATUS_ID)
            requested = (
                status is not None
                and status.gpu_check_requested_at is not None
                and (
                    status.gpu_check_at is None
                    or status.gpu_check_requested_at > status.gpu_check_at
                )
            )
        if not (force or requested):
            return
        logger.info("gpu_check_started")
        result = self.gpu_check()
        with self.session_factory() as session:
            publish_status(session, gpu_check=result, gpu_check_at=_now(), heartbeat_at=_now())
        logger.info("gpu_check_finished", ok=json.loads(result).get("ok"))

    def shutdown_request(self) -> Optional[str]:
        """The mode of a shutdown requested from the web UI since this worker started."""
        with self.session_factory() as session:
            status = session.get(WorkerStatus, WORKER_STATUS_ID)
            if status is None or status.shutdown_requested_at is None or self.started_at is None:
                return None
            requested = status.shutdown_requested_at
            if requested.tzinfo is None:  # SQLite returns naive UTC
                requested = requested.replace(tzinfo=timezone.utc)
            return status.shutdown_mode if requested > self.started_at else None

    def run_forever(self) -> None:
        logger.info("worker_started", pid=os.getpid())
        self.started_at = _now()
        try:
            self._run()
        finally:
            # The API waits for this before it exits (shutdown).
            with self.session_factory() as session:
                publish_status(session, stopped_at=_now())
        logger.info("worker_stopped")

    def _run(self) -> None:
        with self.session_factory() as session:
            publish_status(
                session,
                started_at=self.started_at,
                heartbeat_at=_now(),
                backend=os.environ.get("PIPELINE_HARDWARE") or None,
                version=__version__,
            )
            recover_interrupted_jobs(session)
            # No job runs now: scratch files left by a crash or reboot can go.
            for work_dir in session.scalars(select(Job.work_dir)):
                remove_job_temp_files(work_dir)
        remove_legacy_tmp_files()
        self.check_gpu_if_requested(force=True)
        while not self.stopping:
            self.heartbeat()
            # Between jobs every mode means the same: stop. Queued jobs stay
            # queued and run after the next start.
            mode = self.shutdown_request()
            if mode is not None:
                logger.info("worker_shutdown", mode=mode)
                break
            self.check_gpu_if_requested()
            if not self.run_once():
                time.sleep(self.poll_interval)

    def run_once(self) -> bool:
        """Run the next queued job to its end. Returns False if the queue is empty."""
        with self.session_factory() as session:
            job_id = claim_next_job(session)
        if job_id is None:
            return False
        self.supervise(job_id)
        return True

    def supervise(self, job_id: int) -> JobStatus:
        """Run one job in a child process until it ends, is cancelled or the worker stops."""
        logger.info("job_started", job_id=job_id)
        proc = self.spawn(job_id)
        with self.session_factory() as session:
            # The child leads its own process group (start_new_session).
            session.execute(update(Job).where(Job.id == job_id).values(process_group=proc.pid))
            session.commit()
        outcome = None
        while True:
            try:
                returncode = proc.wait(timeout=self.heartbeat_interval)
                break
            except subprocess.TimeoutExpired:
                pass
            self.heartbeat()
            with self.session_factory() as session:
                job = session.get(Job, job_id)
                job.heartbeat_at = _now()
                session.commit()
                cancel = job.cancel_requested
            # Shutdown from the web UI: "wait" lets the job finish first.
            shutdown = self.shutdown_request()
            if cancel or shutdown == "cancel":
                outcome = JobStatus.CANCELLED
            elif self.stopping or shutdown == "requeue":
                outcome = JobStatus.QUEUED
            if outcome is not None:
                logger.info("job_process_stopping", job_id=job_id, reason=outcome.value)
                _kill_group(proc)
                returncode = proc.returncode
                break
        return self._finish(job_id, returncode, outcome)

    def _finish(self, job_id: int, returncode: int, outcome: Optional[JobStatus]) -> JobStatus:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if outcome == JobStatus.CANCELLED:
                job.status = JobStatus.CANCELLED
                job.completed_at = _now()
                add_event(session, job_id, "job_cancelled", {"job_id": job_id})
            elif outcome == JobStatus.QUEUED:
                # Stopped with the worker: resume from the last artifact next time.
                job.status = JobStatus.QUEUED
                add_event(session, job_id, "job_requeued", {"job_id": job_id, "reason": "stop"})
            elif job.status == JobStatus.RUNNING:
                # The child records completion and pipeline errors itself; a job
                # still marked running means it died (crash, OOM kill, signal).
                error = f"Job process exited unexpectedly (exit code {returncode})"
                job.status = JobStatus.FAILED
                job.error_message = error
                job.completed_at = _now()
                add_event(session, job_id, "job_failed", {"job_id": job_id, "error": error})
            job.current_stage = None
            job.process_group = None
            session.commit()
            status = job.status
            work_dir = job.work_dir
        # Whatever the outcome: a killed job had no chance to clean up, and a
        # requeued one recreates what it needs.
        remove_job_temp_files(work_dir)
        logger.info("job_finished", job_id=job_id, status=status.value, exit_code=returncode)
        return status


# --------------------------------------------------------------------------
# One job (child process)
# --------------------------------------------------------------------------


def make_progress_callback(session_factory: sessionmaker, job_id: int):
    """Pipeline progress callback that records stage events in the database."""

    def callback(stage_name: str, status: str, progress_percent: Optional[float] = None) -> None:
        data = {"stage": stage_name, "status": status}
        if progress_percent is not None:
            data["progress"] = progress_percent
        with session_factory() as session:
            if status == "started":
                session.execute(
                    update(Job).where(Job.id == job_id).values(current_stage=stage_name)
                )
            add_event(session, job_id, f"stage_{status}", data)
            session.commit()

    return callback


def run_job(session_factory: sessionmaker, job_id: int) -> JobStatus:
    """Run the pipeline for one job and record the result.

    Resumes from existing artifacts, so a job that was interrupted continues
    where it stopped. Pipeline errors are recorded as a failed job, not raised.
    """
    from src.api.job_config import build_job_config
    from src.logger import job_log_file, setup_logging
    from src.pipeline import PipelineError, run_chapters_pipeline, run_pipeline

    setup_logging(log_file=None, level="INFO", json_format=False)
    with session_factory() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise WorkerError(f"Job {job_id} not found")
        work_dir, input_source = job.work_dir, job.input_source
        config_overrides, job_type = job.config_overrides, job.job_type
        add_event(session, job_id, "job_started", {"job_id": job_id})
        session.commit()

    job_log = Path(work_dir) / "artifacts" / "logs" / "job.log"
    status, error = JobStatus.COMPLETED, None
    with job_log_file(job_log, job_id):
        try:
            config = build_job_config(config_overrides, work_dir)
            pipeline_fn = run_chapters_pipeline if job_type == JobType.CHAPTERS else run_pipeline
            pipeline_fn(
                config=config,
                input_source=input_source,
                dry_run=False,
                resume=True,
                force=False,
                progress_callback=make_progress_callback(session_factory, job_id),
            )
            logger.info("Job completed", job_id=job_id, job_type=job_type.value)
        except PipelineError as e:
            status, error = JobStatus.FAILED, str(e)
            logger.error("Job failed", job_id=job_id, error=error)
        except Exception as e:
            status, error = JobStatus.FAILED, f"Unexpected error: {e}"
            logger.error("Job failed with unexpected error", job_id=job_id, error=str(e))

    with session_factory() as session:
        job = session.get(Job, job_id)
        job.status = status
        job.error_message = error
        job.completed_at = _now()
        if status == JobStatus.COMPLETED:
            add_event(session, job_id, "job_completed", {"job_id": job_id})
        else:
            add_event(session, job_id, "job_failed", {"job_id": job_id, "error": error})
        session.commit()
    return status


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run queued pipeline jobs.")
    parser.add_argument("--run-job", type=int, help="run one job in this process")
    args = parser.parse_args(argv)

    session_factory = make_session_factory()
    if args.run_job is not None:
        run_job(session_factory, args.run_job)
        return 0

    from src.logger import setup_logging

    setup_logging(log_file=None, level="INFO", json_format=False)
    worker = Worker(session_factory)
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
    worker.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
