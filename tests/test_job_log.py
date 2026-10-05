"""
Tests for per-job log files (src/logger.py job_log_file).
"""

import asyncio
import json

import structlog

from src.logger import job_log_file, setup_logging


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_concurrent_jobs_get_separate_logs(tmp_path):
    """Records from the pipeline thread land in their own job's file only."""
    setup_logging(log_file=None, level="INFO", json_format=True)
    log = structlog.get_logger("test_job_log")

    def pipeline(stage):
        log.info("stage_done", stage=stage)

    async def run_job(job_id):
        with job_log_file(tmp_path / f"job_{job_id}" / "job.log", job_id):
            log.info("job_started")
            await asyncio.to_thread(pipeline, f"scoring_{job_id}")
            await asyncio.sleep(0)
            await asyncio.to_thread(pipeline, f"rendering_{job_id}")

    async def main():
        await asyncio.gather(run_job(1), run_job(2))
        log.info("outside_any_job")

    asyncio.run(main())

    for job_id in (1, 2):
        events = _events(tmp_path / f"job_{job_id}" / "job.log")
        assert [e["event"] for e in events] == ["job_started", "stage_done", "stage_done"]
        assert {e["job_id"] for e in events} == {job_id}
        assert [e["stage"] for e in events[1:]] == [f"scoring_{job_id}", f"rendering_{job_id}"]


def test_context_is_restored_after_job(tmp_path):
    setup_logging(log_file=None, level="INFO", json_format=True)

    with job_log_file(tmp_path / "job.log", 7):
        assert structlog.contextvars.get_contextvars()["job_id"] == 7

    assert "job_id" not in structlog.contextvars.get_contextvars()
    structlog.get_logger("test_job_log").info("after_job")
    assert [e["event"] for e in _events(tmp_path / "job.log")] == []
