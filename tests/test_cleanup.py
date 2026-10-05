"""
Tests for the disk cleanup: source videos of old jobs, orphan
uploads, job deletion, the API endpoints.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.models.job import Job, JobEvent, JobStatus
from src.api.models.review import ClipReview
from src.api.services.cleanup import (
    CleanupError,
    delete_job,
    remove_job_sources,
    remove_orphan_uploads,
    run_cleanup,
)
from src.config import CleanupConfig

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
# _job(): prep.mp4 1000 + raw.wav 100 + upload 2000 are the sources;
# transcript.json 10 + clip 300 stay.
SOURCES = 3100
WHOLE_JOB = 3410


@pytest_asyncio.fixture
async def factory(tmp_path, monkeypatch):
    """An empty job database; jobs/ and uploads/ live in tmp_path (the cwd)."""
    from src.api.database import Base

    monkeypatch.chdir(tmp_path)
    (tmp_path / "jobs").mkdir()
    (tmp_path / "uploads").mkdir()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


async def _job(
    db,
    name: str,
    status=JobStatus.COMPLETED,
    finished_days_ago: float = 40,
    upload: str = None,
) -> Job:
    """A job with a source video copy, audio, clips and an upload."""
    work = Path("jobs") / name
    _write(work / "artifacts" / "video" / "prep.mp4", 1000)
    _write(work / "artifacts" / "audio" / "raw.wav", 100)
    _write(work / "artifacts" / "transcript.json", 10)
    _write(work / "output" / "clips" / "clip_001.mp4", 300)
    upload = upload or f"uploads/1/{name}.mp4"
    if not Path(upload).exists():
        _write(Path(upload), 2000)
    finished = NOW - timedelta(days=finished_days_ago)
    job = Job(
        user_id=1,
        status=status,
        input_source=upload,
        work_dir=str(work),
        created_at=finished,
        completed_at=finished if status != JobStatus.RUNNING else None,
    )
    db.add(job)
    await db.commit()
    return job


class TestRemoveJobSources:
    @pytest.mark.asyncio
    async def test_removes_video_audio_and_upload_keeps_clips(self, factory):
        async with factory() as db:
            job = await _job(db, "job_a")
            freed = await remove_job_sources(db, job)
            work = Path(job.work_dir)
            assert freed == SOURCES
            assert not (work / "artifacts" / "video").exists()
            assert not (work / "artifacts" / "audio").exists()
            assert not Path(job.input_source).exists()
            assert (work / "output" / "clips" / "clip_001.mp4").exists()
            assert (work / "artifacts" / "transcript.json").exists()
            assert job.sources_removed_at is not None

    @pytest.mark.asyncio
    async def test_upload_kept_while_another_job_needs_it(self, factory):
        async with factory() as db:
            old = await _job(db, "job_old", upload="uploads/1/shared.mp4")
            recent = await _job(db, "job_new", upload="uploads/1/shared.mp4")
            await remove_job_sources(db, old)
            assert Path("uploads/1/shared.mp4").exists()
            # Once the second job's sources go too, nobody needs the upload.
            await remove_job_sources(db, recent)
            assert not Path("uploads/1/shared.mp4").exists()

    @pytest.mark.asyncio
    async def test_running_job_is_refused(self, factory):
        async with factory() as db:
            job = await _job(db, "job_r", status=JobStatus.RUNNING)
            with pytest.raises(CleanupError):
                await remove_job_sources(db, job)
            assert (Path(job.work_dir) / "artifacts" / "video").exists()

    @pytest.mark.asyncio
    async def test_work_dir_outside_jobs_is_not_touched(self, factory, tmp_path):
        async with factory() as db:
            job = await _job(db, "job_x")
            outside = _write(tmp_path / "elsewhere" / "artifacts" / "video" / "v.mp4", 5)
            job.work_dir = str(tmp_path / "elsewhere")
            await db.commit()
            await remove_job_sources(db, job)
            assert outside.exists()

    @pytest.mark.asyncio
    async def test_url_source_is_left_alone(self, factory):
        async with factory() as db:
            job = await _job(db, "job_u")
            job.input_source = "https://youtube.com/watch?v=x"
            await db.commit()
            assert await remove_job_sources(db, job) == 1100


class TestRunCleanup:
    @pytest.mark.asyncio
    async def test_only_finished_jobs_older_than_retention(self, factory):
        async with factory() as db:
            old = await _job(db, "job_old", finished_days_ago=31)
            young = await _job(db, "job_young", finished_days_ago=29)
            failed = await _job(db, "job_failed", status=JobStatus.FAILED, finished_days_ago=60)
            running = await _job(db, "job_run", status=JobStatus.RUNNING, finished_days_ago=60)
            result = await run_cleanup(db, CleanupConfig(retention_days=30), now=NOW)
            assert result["jobs_cleaned"] == 2
            assert old.sources_removed_at and failed.sources_removed_at
            assert young.sources_removed_at is None and running.sources_removed_at is None
            assert result["total_bytes_freed"] == 2 * SOURCES

    @pytest.mark.asyncio
    async def test_already_cleaned_job_is_skipped(self, factory):
        async with factory() as db:
            await _job(db, "job_old")
            first = await run_cleanup(db, CleanupConfig(retention_days=30), now=NOW)
            second = await run_cleanup(db, CleanupConfig(retention_days=30), now=NOW)
            assert (first["jobs_cleaned"], second["jobs_cleaned"]) == (1, 0)

    @pytest.mark.asyncio
    async def test_zero_retention_keeps_everything(self, factory):
        async with factory() as db:
            job = await _job(db, "job_old", finished_days_ago=400)
            result = await run_cleanup(db, CleanupConfig(retention_days=0), now=NOW)
            assert result["jobs_cleaned"] == 0 and result["cutoff_date"] is None
            assert job.sources_removed_at is None
            assert Path(job.input_source).exists()

    def test_negative_retention_is_rejected(self):
        with pytest.raises(ValueError):
            CleanupConfig(retention_days=-1)


class TestOrphanUploads:
    @pytest.mark.asyncio
    async def test_unreferenced_old_upload_removed(self, factory):
        async with factory() as db:
            job = await _job(db, "job_a", finished_days_ago=1)
            orphan = _write(Path("uploads/1/never_used.mp4"), 50)
            fresh = _write(Path("uploads/1/just_uploaded.mp4"), 50)
            partial = _write(Path("uploads/1/in_progress.mp4.part"), 50)
            day_ago = time.time() - 25 * 3600
            for path in (orphan, partial, Path(job.input_source)):
                os.utime(path, (day_ago, day_ago))
            result = await remove_orphan_uploads(db)
            assert result == {"files_removed": 1, "bytes_freed": 50}
            assert not orphan.exists()
            assert fresh.exists() and partial.exists() and Path(job.input_source).exists()


class TestDeleteJob:
    @pytest.mark.asyncio
    async def test_removes_directory_upload_and_rows(self, factory):
        async with factory() as db:
            job = await _job(db, "job_d", finished_days_ago=1)
            db.add(JobEvent(job_id=job.id, event_type="stage", data="{}"))
            db.add(ClipReview(job_id=job.id, clip_id="clip_001", status="keep"))
            await db.commit()
            work, upload = Path(job.work_dir), Path(job.input_source)
            assert await delete_job(db, job) == WHOLE_JOB
            assert not work.exists() and not upload.exists()
            for model in (Job, JobEvent, ClipReview):
                assert await db.scalar(select(func.count()).select_from(model)) == 0

    @pytest.mark.asyncio
    async def test_running_job_cannot_be_deleted(self, factory):
        async with factory() as db:
            job = await _job(db, "job_r", status=JobStatus.RUNNING)
            with pytest.raises(CleanupError, match="cancel it first"):
                await delete_job(db, job)
            assert Path(job.work_dir).exists()


@pytest_asyncio.fixture
async def client(factory):
    from src.api.app import app
    from src.api.database import get_db

    async def override_get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, factory
    app.dependency_overrides.clear()


class TestApi:
    @pytest.mark.asyncio
    async def test_delete_endpoint(self, client):
        ac, factory = client
        async with factory() as db:
            job = await _job(db, "job_api", finished_days_ago=1)
        response = await ac.post(f"/api/v1/jobs/{job.id}/delete")
        assert response.status_code == 204
        assert (await ac.get(f"/api/v1/jobs/{job.id}")).status_code == 404
        assert not Path(job.work_dir).exists()

    @pytest.mark.asyncio
    async def test_delete_running_job_is_conflict(self, client):
        ac, factory = client
        async with factory() as db:
            job = await _job(db, "job_run", status=JobStatus.RUNNING)
        response = await ac.post(f"/api/v1/jobs/{job.id}/delete")
        assert response.status_code == 409
        assert "cancel it first" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_resume_refused_after_sources_removed(self, client):
        ac, factory = client
        async with factory() as db:
            job = await _job(db, "job_f", status=JobStatus.FAILED)
            await remove_job_sources(db, job)
        detail = (await ac.get(f"/api/v1/jobs/{job.id}")).json()
        assert detail["sources_removed_at"] is not None
        response = await ac.post(f"/api/v1/jobs/{job.id}/retry")
        assert response.status_code == 409
        assert "source video" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_manual_cleanup_endpoint(self, client, tmp_path):
        ac, factory = client
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "config.yaml").write_text("cleanup:\n  retention_days: 30\n")
        async with factory() as db:
            await _job(db, "job_old", finished_days_ago=400)
        result = (await ac.delete("/api/v1/cleanup/")).json()
        assert result["jobs_cleaned"] == 1
        assert result["total_bytes_freed"] == SOURCES
