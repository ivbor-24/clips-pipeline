"""Declarative base, schema helpers and database settings shared by the API and the job worker.

Kept apart from ``src.api.database`` so the worker can use the models without
loading the API settings (which fail fast on an unsafe ``API_JWT_SECRET``).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, event
from sqlalchemy.orm import DeclarativeBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatabaseSettings(BaseSettings):
    """Where the job queue lives: ``API_*`` environment variables, then the project's ``.env``.

    The API and the worker read it the same way, so they always open one database.
    Real environment variables win over ``.env``.
    """

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",  # .env also holds GPU_BACKEND, HF_TOKEN and other non-API keys
        hide_input_in_errors=True,  # a startup error must not print API_PASSWORD
    )

    database_url: str = "sqlite+aiosqlite:///./data/api.db"


def sync_database_url(url: str) -> str:
    """The API uses async drivers; the worker and the migrations need their sync counterparts."""
    return url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")


class Base(DeclarativeBase):
    pass


def enable_sqlite_concurrency(sync_engine: Engine) -> None:
    """Let the API and the worker use one SQLite file at the same time.

    WAL lets readers work while the other process writes; busy_timeout makes
    a writer wait for the lock instead of failing with "database is locked".
    """
    if sync_engine.dialect.name != "sqlite":
        return

    @event.listens_for(sync_engine, "connect")
    def _set_pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout=10000")
        if sync_engine.url.database not in (None, "", ":memory:"):
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
