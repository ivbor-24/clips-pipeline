"""Tests for the job database migrations (src/api/migrate.py)."""

import sqlite3
import threading
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

import src.api.models  # noqa: F401  (registers every model on Base.metadata)
from src.api.db_base import Base
from src.api.migrate import MigrationError, head_revision, upgrade_database


def _url(path: Path) -> str:
    return f"sqlite:///{path}"


def _rows(path: Path, sql: str) -> list:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _columns(path: Path, table: str) -> set:
    return {row[1] for row in _rows(path, f"PRAGMA table_info({table})")}


class TestSchema:
    def test_migrations_match_the_models(self, tmp_path):
        """A model change without a migration fails here: run just db-revision."""
        db = tmp_path / "api.db"
        upgrade_database(_url(db))
        engine = create_engine(_url(db))
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        engine.dispose()
        assert diff == []

    def test_new_database(self, tmp_path):
        db = tmp_path / "data" / "api.db"
        assert upgrade_database(_url(db)) is None
        tables = {r[0] for r in _rows(db, "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"jobs", "job_events", "clip_reviews", "worker_status"} <= tables
        assert _rows(db, "SELECT version_num FROM alembic_version") == [(head_revision(),)]
        assert not list(tmp_path.glob("data/api.db.before-*"))  # nothing to keep

    def test_up_to_date_database_is_left_alone(self, tmp_path):
        db = tmp_path / "api.db"
        upgrade_database(_url(db))
        assert upgrade_database(_url(db)) == head_revision()
        assert not list(tmp_path.glob("api.db.before-*"))


class TestDatabaseFromBeforeMigrations:
    def test_old_schema_gets_missing_tables_and_columns(self, tmp_path):
        # A database of an early version: jobs without the worker's columns,
        # no clip_reviews and worker_status tables.
        db = tmp_path / "api.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE jobs (
                id INTEGER NOT NULL PRIMARY KEY, user_id INTEGER NOT NULL,
                status VARCHAR(9) NOT NULL, job_type VARCHAR(8) NOT NULL,
                input_source VARCHAR(500) NOT NULL, config_overrides TEXT,
                work_dir VARCHAR(500) NOT NULL, created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
                started_at DATETIME, completed_at DATETIME, error_message TEXT);
            CREATE INDEX ix_jobs_id ON jobs (id);
            CREATE TABLE job_events (
                id INTEGER NOT NULL PRIMARY KEY, job_id INTEGER NOT NULL REFERENCES jobs (id),
                event_type VARCHAR(32) NOT NULL, data TEXT NOT NULL,
                created_at DATETIME DEFAULT (CURRENT_TIMESTAMP));
            INSERT INTO jobs (id, user_id, status, job_type, input_source, work_dir)
                VALUES (7, 1, 'COMPLETED', 'CLIPS', 'uploads/1/a.mp4', 'jobs/job_a');
            """)
        conn.close()

        assert upgrade_database(_url(db)) is None

        assert {"current_stage", "cancel_requested", "attempts", "sources_removed_at"} <= _columns(
            db, "jobs"
        )
        assert _rows(db, "SELECT id, status, cancel_requested, attempts FROM jobs") == [
            (7, "COMPLETED", 0, 0)
        ]
        assert "notes" in _columns(db, "clip_reviews") and "gpu_check" in _columns(
            db, "worker_status"
        )
        indexes = {r[0] for r in _rows(db, "SELECT name FROM sqlite_master WHERE type='index'")}
        assert {"ix_jobs_id", "ix_jobs_user_id", "ix_job_events_job_id"} <= indexes
        assert _rows(db, "SELECT version_num FROM alembic_version") == [(head_revision(),)]
        kept = tmp_path / "api.db.before-migrations"
        assert _rows(kept, "SELECT id FROM jobs") == [(7,)]
        assert "sources_removed_at" not in _columns(kept, "jobs")

    def test_schema_of_v0_8_is_only_stamped(self, tmp_path):
        """The last create_all schema already is the baseline."""
        db = tmp_path / "api.db"
        engine = create_engine(_url(db))
        Base.metadata.create_all(engine)
        engine.dispose()
        upgrade_database(_url(db))
        assert _rows(db, "SELECT version_num FROM alembic_version") == [(head_revision(),)]


class TestRefusals:
    def test_database_of_a_newer_version(self, tmp_path):
        db = tmp_path / "api.db"
        upgrade_database(_url(db))
        conn = sqlite3.connect(db)
        conn.execute("UPDATE alembic_version SET version_num = '9999'")
        conn.commit()
        conn.close()
        with pytest.raises(MigrationError, match="revision 9999.*newer version"):
            upgrade_database(_url(db))


class TestConcurrency:
    def test_api_and_worker_starting_together(self, tmp_path):
        db = tmp_path / "api.db"
        errors = []

        def start():
            try:
                upgrade_database(_url(db))
            except Exception as e:  # noqa: BLE001 - collected for the assertion
                errors.append(e)

        threads = [threading.Thread(target=start) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert errors == []
        assert _rows(db, "SELECT version_num FROM alembic_version") == [(head_revision(),)]


class TestVersion:
    def test_one_version_everywhere(self):
        import json
        import re

        from src import __version__

        root = Path(__file__).resolve().parent.parent
        pyproject = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
        package = json.loads((root / "web" / "package.json").read_text())["version"]
        assert pyproject.group(1) == __version__ == package
