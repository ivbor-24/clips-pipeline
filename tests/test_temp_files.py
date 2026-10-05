"""Temporary files of a job: kept in the work directory, removed by the worker."""

import os
import time
from pathlib import Path

import pytest

from src.api.models.job import Job, JobStatus
from src.temp_files import job_temp_dir, remove_job_temp_files, remove_legacy_tmp_files
from src.worker import Worker, make_session_factory
from tests.test_worker import add_job, get_job, spawn_python


def make_job_dir(work_dir: Path) -> None:
    """A work directory with scratch data next to real artifacts."""
    (work_dir / "artifacts" / "tmp" / "face_crop_x").mkdir(parents=True)
    (work_dir / "artifacts" / "tmp" / "face_crop_x" / "f_0001.jpg").write_bytes(b"\0" * 100)
    (work_dir / "artifacts" / "temp_download").mkdir(parents=True)
    (work_dir / "artifacts" / "temp_download" / "downloaded.part").write_bytes(b"\0" * 10)
    (work_dir / "artifacts" / "audio" / "temp_chunks").mkdir(parents=True)
    (work_dir / "artifacts" / "audio" / "temp_chunks" / "chunk_000.wav").write_bytes(b"\0" * 10)
    (work_dir / "artifacts" / "audio" / "raw.wav").write_bytes(b"\0" * 10)
    (work_dir / "artifacts" / "transcript.json").write_text("[]")


def assert_only_scratch_removed(work_dir: Path) -> None:
    assert not (work_dir / "artifacts" / "tmp").exists()
    assert not (work_dir / "artifacts" / "temp_download").exists()
    assert not (work_dir / "artifacts" / "audio" / "temp_chunks").exists()
    assert (work_dir / "artifacts" / "audio" / "raw.wav").exists()
    assert (work_dir / "artifacts" / "transcript.json").exists()


class TestRemoveJobTempFiles:
    def test_removes_scratch_keeps_artifacts(self, tmp_path):
        make_job_dir(tmp_path)
        assert remove_job_temp_files(tmp_path) == 120
        assert_only_scratch_removed(tmp_path)

    def test_nothing_to_remove(self, tmp_path):
        assert remove_job_temp_files(tmp_path / "missing") == 0

    def test_job_temp_dir_is_inside_the_work_dir(self, tmp_path):
        path = job_temp_dir(tmp_path)
        assert path == tmp_path / "artifacts" / "tmp"
        assert path.is_dir()


class TestLegacyTmp:
    def test_removes_only_our_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        (tmp_path / "face_crop_abc").mkdir()
        (tmp_path / "face_crop_abc" / "f.jpg").write_bytes(b"\0" * 5)
        (tmp_path / "someone_else").mkdir()

        assert remove_legacy_tmp_files() == 5
        assert not (tmp_path / "face_crop_abc").exists()
        assert (tmp_path / "someone_else").exists()


@pytest.fixture
def session_factory(tmp_path):
    return make_session_factory(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")


class TestWorkerCleanup:
    @pytest.mark.parametrize(
        "code, cancel, expected",
        [
            ("import sys; sys.exit(3)", False, JobStatus.FAILED),
            ("import time; time.sleep(60)", True, JobStatus.CANCELLED),
        ],
    )
    def test_after_the_job_process_ends(self, session_factory, tmp_path, code, cancel, expected):
        """Killed or crashed, the job never cleans up itself; the worker does."""
        job_id = add_job(session_factory, tmp_path, status=JobStatus.RUNNING)
        work_dir = Path(get_job(session_factory, job_id).work_dir)
        make_job_dir(work_dir)

        def spawn(jid):
            if cancel:
                with session_factory() as session:
                    session.get(Job, jid).cancel_requested = True
                    session.commit()
            return spawn_python(code)(jid)

        worker = Worker(session_factory, spawn=spawn, heartbeat_interval=0.1)
        assert worker.supervise(job_id) == expected
        assert_only_scratch_removed(work_dir)

    def test_at_worker_start(self, session_factory, tmp_path, monkeypatch):
        legacy_tmp = tmp_path / "system_tmp"
        (legacy_tmp / "face_crop_old").mkdir(parents=True)
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(legacy_tmp))
        job_id = add_job(session_factory, tmp_path, status=JobStatus.FAILED)
        work_dir = Path(get_job(session_factory, job_id).work_dir)
        make_job_dir(work_dir)

        worker = Worker(session_factory)
        worker.stopping = True  # recover and clean up, then leave the loop
        worker.run_forever()

        assert_only_scratch_removed(work_dir)
        assert not (legacy_tmp / "face_crop_old").exists()


class TestPartialUploads:
    def test_stale_part_files_removed_fresh_kept(self, tmp_path, monkeypatch):
        from src.api.routes import upload

        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path / "uploads")
        user_dir = tmp_path / "uploads" / "1"
        user_dir.mkdir(parents=True)
        stale = user_dir / "a_lecture.mp4.part"
        fresh = user_dir / "b_lecture.mp4.part"
        done = user_dir / "c_lecture.mp4"
        for path in (stale, fresh, done):
            path.write_bytes(b"\0")
        old = time.time() - 2 * 3600
        os.utime(stale, (old, old))
        os.utime(done, (old, old))

        assert upload.remove_stale_partial_uploads() == 1
        assert not stale.exists()
        assert fresh.exists()
        assert done.exists()
