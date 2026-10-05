"""Tests for scripts/backup.py and scripts/backup.sh: just backup / just restore."""

import importlib.util
import io
import json
import shutil
import sqlite3
import subprocess
import tarfile
from pathlib import Path

import pytest

from tests.fake_docker import make_fake_docker

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sql(db: Path, *statements: str) -> list:
    """Run statements and close the connection, as a stopped API leaves the file."""
    conn = sqlite3.connect(db)
    try:
        rows = [conn.execute(statement).fetchall() for statement in statements]
        conn.commit()
    finally:
        conn.close()
    return rows[-1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project with a job database, settings, one finished job and its upload."""
    root = tmp_path / "project"
    db = root / "data" / "api.db"
    db.parent.mkdir(parents=True)
    _sql(
        db,
        "PRAGMA journal_mode=WAL",
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, status TEXT, work_dir TEXT, input_source TEXT)",
        "INSERT INTO jobs VALUES (1, 'COMPLETED', 'jobs/job_a', 'uploads/1/a.mp4'), "
        "(2, 'RUNNING', 'jobs/job_b', 'uploads/1/b.mp4')",
    )
    _write(root / "config" / "config.yaml", b"cleanup:\n  retention_days: 30\n")
    _write(root / "data" / "settings.yaml", b"scoring:\n  max_clips_per_video: 8\n")
    _write(root / ".env", b"GPU_BACKEND=cpu\nAPI_PASSWORD=secret\n")
    _write(root / "src" / "__init__.py", b'__version__ = "9.9.9"\n')
    job = root / "jobs" / "job_a"
    _write(job / "artifacts" / "video" / "prep.mp4", b"v" * 100)
    _write(job / "artifacts" / "audio" / "raw.wav", b"a" * 10)
    _write(job / "artifacts" / "transcript.json", b"{}")
    _write(job / "artifacts" / "tmp" / "frame.jpg")
    _write(job / "output" / "clips" / "clip_001.mp4", b"c" * 30)
    _write(job / "output" / "clips" / "clip_002.part.mp4", b"half")
    _write(root / "uploads" / "1" / "a.mp4", b"u" * 200)
    monkeypatch.setenv("BACKUP_ROOT", str(root))
    monkeypatch.delenv("API_DATABASE_URL", raising=False)
    spec = importlib.util.spec_from_file_location("backup_under_test", REPO / "scripts/backup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module


def _members(archive: Path) -> dict:
    with tarfile.open(archive) as tar:
        return {m.name: m for m in tar.getmembers()}


def _manifest(archive: Path) -> dict:
    with tarfile.open(archive) as tar:
        return json.load(tar.extractfile("backup.json"))


class TestCreate:
    def test_without_sources(self, project):
        root, backup = project
        archive = backup.create_backup()
        names = _members(archive)

        assert archive.parent == root / "backups"
        assert archive.stat().st_mode & 0o777 == 0o600
        assert {
            "backup.json",
            "data/api.db",
            "data/settings.yaml",
            ".env",
            "jobs/job_a/artifacts/transcript.json",
            "jobs/job_a/output/clips/clip_001.mp4",
        } <= set(names)
        # The tracked config/config.yaml is not part of a backup: user settings
        # (data/settings.yaml) took its place.
        assert "config/config.yaml" not in names
        assert names["data/settings.yaml"].mode == 0o600
        assert names[".env"].mode == 0o600
        skipped = ("video/", "audio/", "artifacts/tmp", ".part.mp4", "uploads/")
        assert not [n for n in names if any(part in n for part in skipped)]
        manifest = _manifest(archive)
        assert manifest["with_sources"] is False and manifest["app_version"] == "9.9.9"
        assert manifest["jobs"] == 2 and manifest["active_jobs"] == ["2"]

    def test_without_settings_file(self, project):
        root, backup = project
        (root / "data" / "settings.yaml").unlink()
        names = _members(backup.create_backup())
        assert "data/settings.yaml" not in names  # optional: only when it exists

    def test_with_sources(self, project):
        _, backup = project
        names = _members(backup.create_backup(with_sources=True))
        assert "jobs/job_a/artifacts/video/prep.mp4" in names
        assert "jobs/job_a/artifacts/audio/raw.wav" in names
        assert "uploads/1/a.mp4" in names
        assert "jobs/job_a/output/clips/clip_002.part.mp4" not in names

    def test_database_snapshot_is_complete(self, project, tmp_path):
        root, backup = project
        archive = backup.create_backup()
        snapshot = tmp_path / "snapshot.db"
        with tarfile.open(archive) as tar:
            snapshot.write_bytes(tar.extractfile("data/api.db").read())
        assert _sql(snapshot, "SELECT count(*) FROM jobs") == [(2,)]

    def test_database_url_from_env_file(self, project):
        root, backup = project
        (root / "data" / "api.db").rename(root / "data" / "other.db")
        with (root / ".env").open("a") as env:
            env.write("API_DATABASE_URL=sqlite+aiosqlite:///./data/other.db\n")
        assert backup.database_path() == root / "data" / "other.db"
        assert "data/api.db" in _members(backup.create_backup())

    def test_no_database(self, project):
        root, backup = project
        (root / "data" / "api.db").unlink()
        with pytest.raises(backup.BackupError, match="nothing to back up"):
            backup.create_backup()


class TestRestore:
    def test_restores_over_changed_data(self, project):
        root, backup = project
        archive = backup.create_backup()
        # After the backup: a job lost, settings changed, a new .env.
        shutil.rmtree(root / "jobs" / "job_a")
        _sql(root / "data" / "api.db", "DELETE FROM jobs")
        (root / "data" / "settings.yaml").write_text("changed\n")
        (root / ".env").write_text("GPU_BACKEND=cuda\n")
        _write(root / "data" / "api.db-wal", b"stale log")

        result = backup.restore_backup(archive, assume_yes=True)

        assert _sql(root / "data" / "api.db", "SELECT count(*) FROM jobs") == [(2,)]
        assert (root / "jobs/job_a/output/clips/clip_001.mp4").read_bytes() == b"c" * 30
        assert "max_clips_per_video" in (root / "data" / "settings.yaml").read_text()
        assert (root / "data" / "settings.yaml").stat().st_mode & 0o777 == 0o600
        # config/config.yaml is not in the backup; whatever is on disk stays.
        assert (root / "config" / "config.yaml").read_text() == "cleanup:\n  retention_days: 30\n"
        assert (root / ".env").read_text() == "GPU_BACKEND=cuda\n"
        env_copy = root / ".env.from-backup"
        assert "API_PASSWORD=secret" in env_copy.read_text()
        assert env_copy.stat().st_mode & 0o777 == 0o600
        assert not (root / "data" / "api.db-wal").exists()
        safety = Path(result["safety_backup"])
        assert safety.name.startswith("before-restore-")
        assert "data/settings.yaml" in _members(safety)

    def test_restores_config_from_old_backup(self, project, tmp_path):
        """Archives made before the Settings page carried config/config.yaml."""
        root, backup = project
        archive = tmp_path / "old.tar"
        manifest = (
            b'{"format": 1, "created_at": "2026-09-01T00:00:00", "app_version": "0.7", '
            b'"with_sources": false, "jobs": 1, "bytes": 0}'
        )
        with tarfile.open(archive, "w") as tar:
            for name, data in (
                ("backup.json", manifest),
                ("data/api.db", b""),
                ("config/config.yaml", b"cleanup:\n  retention_days: 7\n"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

        backup.restore_backup(archive, assume_yes=True)

        assert (root / "config" / "config.yaml").read_text() == "cleanup:\n  retention_days: 7\n"
        # The user settings on disk are not touched by an old backup.
        assert (
            root / "data" / "settings.yaml"
        ).read_text() == "scoring:\n  max_clips_per_video: 8\n"

    def test_answer_no_changes_nothing(self, project, monkeypatch):
        root, backup = project
        archive = backup.create_backup()
        (root / "data" / "settings.yaml").write_text("changed\n")
        monkeypatch.setattr("builtins.input", lambda prompt: "n")
        with pytest.raises(backup.BackupError, match="Cancelled"):
            backup.restore_backup(archive)
        assert (root / "data" / "settings.yaml").read_text() == "changed\n"

    @pytest.mark.parametrize(
        "name, kind",
        [
            ("../evil.sh", tarfile.REGTYPE),
            ("/etc/cron.d/x", tarfile.REGTYPE),
            ("jobs/link", tarfile.SYMTYPE),
            ("src/api/app.py", tarfile.REGTYPE),
        ],
    )
    def test_unsafe_archive_is_refused(self, project, tmp_path, name, kind):
        root, backup = project
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tar:
            for member_name, data in (("backup.json", b'{"format": 1}'), ("data/api.db", b"")):
                info = tarfile.TarInfo(member_name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
            tar.addfile(info, io.BytesIO(b""))
        with pytest.raises(backup.BackupError, match="unexpected"):
            backup.restore_backup(archive, assume_yes=True)
        assert not (root / "backups").exists()  # refused before the safety backup

    def test_not_a_backup(self, project, tmp_path):
        _, backup = project
        archive = tmp_path / "other.tar"
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo("jobs/x.txt")
            tar.addfile(info, io.BytesIO(b""))
        with pytest.raises(backup.BackupError, match="no backup.json"):
            backup.restore_backup(archive, assume_yes=True)


class TestWrapper:
    def _run(self, project_root, tmp_path, *args, running=""):
        scripts = project_root / "scripts"
        scripts.mkdir(exist_ok=True)
        for name in ("backup.sh", "backup.py", "docker_common.sh"):
            shutil.copy2(REPO / "scripts" / name, scripts / name)
        bin_dir, state = make_fake_docker(tmp_path)
        (state / "running").write_text(running)
        return subprocess.run(
            [BASH, str(scripts / "backup.sh"), *args],
            cwd=project_root,
            env={"PATH": str(bin_dir), "HOME": str(tmp_path), "FAKE_STATE": str(state)},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_backup_while_running_is_fine(self, project, tmp_path):
        root, _ = project
        result = self._run(root, tmp_path, "create", running="backend\nworker-cpu\n")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Backup: " in result.stdout
        assert "jobs 2 were queued or running" in result.stderr

    def test_restore_refused_while_running(self, project, tmp_path):
        root, _ = project
        made = self._run(root, tmp_path, "create")
        archive = made.stdout.split("Backup: ")[1].split(" (")[0]
        shutil.rmtree(tmp_path / "bin")
        shutil.rmtree(tmp_path / "state")
        result = self._run(root, tmp_path, "restore", archive, "--yes", running="backend\n")
        assert result.returncode == 1
        assert "Stop them first: just stop" in result.stderr
