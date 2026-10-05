"""
Job database migrations: Alembic, applied by the API and the worker at start.

- A new database is created by the migrations.
- A database from before migrations (up to v0.8.0) is brought to the
  baseline by revision 0001 itself, which adds only what is missing.
- Before an existing database is upgraded, a copy is kept next to it:
  ``<db>.before-<revision>`` (``before-migrations`` for one from before them).
- A database written by a newer version of the code is refused with a clear
  message instead of being used with a schema this code does not know (an
  older version restored over newer data, for example).
- The API and the worker start together: a lock file next to the SQLite file
  makes the second one wait for the first.

Inputs / Outputs:
- data/api.db (API_DATABASE_URL), src/api/migrations/versions/*.py
"""

import fcntl
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import structlog
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Connection, make_url

logger = structlog.get_logger("migrate")

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class MigrationError(Exception):
    """The database cannot be brought to this code's schema; the message says why."""


def alembic_config(connection: Optional[Connection] = None) -> Config:
    """Alembic settings without alembic.ini (the Docker image has none)."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def head_revision() -> str:
    """The newest revision this code knows."""
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def _sqlite_path(url: str) -> Optional[Path]:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite" or parsed.database in (None, "", ":memory:"):
        return None
    return Path(parsed.database)


@contextmanager
def _migration_lock(path: Optional[Path]) -> Iterator[None]:
    """One migration at a time: the API and the worker start together."""
    if path is None:
        yield
        return
    with open(path.with_name(path.name + ".migrate.lock"), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _keep_copy(path: Path, label: str) -> Path:
    """A consistent copy of the database file before a migration changes it."""
    target = path.with_name(f"{path.name}.before-{label}")
    source = sqlite3.connect(path)
    try:
        copy = sqlite3.connect(target)
        with copy:
            source.backup(copy)
        copy.close()
    finally:
        source.close()
    return target


def upgrade_connection(connection: Connection) -> None:
    """Apply the migrations on an open connection (an in-memory database has no file)."""
    command.upgrade(alembic_config(connection), "head")


def upgrade_database(sync_url: str) -> Optional[str]:
    """Bring the job database to the newest revision.

    Args:
        sync_url: SQLAlchemy URL with a sync driver (src.api.db_base.sync_database_url).

    Returns:
        The revision the database was at; None for a new database or one from
        before migrations.

    Raises:
        MigrationError: The database was written by a newer version of the code.
    """
    path = _sqlite_path(sync_url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    script = ScriptDirectory.from_config(alembic_config())
    head = script.get_current_head()
    known = {revision.revision for revision in script.walk_revisions()}

    with _migration_lock(path):
        engine = create_engine(sync_url)
        try:
            with engine.connect() as conn:
                current = MigrationContext.configure(conn).get_current_revision()
                has_tables = bool(inspect(conn).get_table_names())
            if current == head:
                return current
            if current is not None and current not in known:
                raise MigrationError(
                    f"The job database is at revision {current}, which this version of the "
                    f"code does not know (newest: {head}): it was written by a newer version. "
                    "Update the code (git pull, then ./setup.sh) or restore a backup made by "
                    "this version (just restore)."
                )
            if has_tables and path is not None:
                copy = _keep_copy(path, current or "migrations")
                logger.info("database_copy_before_migration", path=str(copy))
            with engine.begin() as conn:
                upgrade_connection(conn)
            logger.info("database_migrated", from_revision=current, to_revision=head)
            return current
        finally:
            engine.dispose()
