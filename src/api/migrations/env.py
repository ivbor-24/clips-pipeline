"""Alembic environment of the job database.

Two ways in:
- at start, the API and the worker call src.api.migrate.upgrade_database,
  which passes its connection in ``config.attributes["connection"]``;
- a developer runs `just db-revision "message"` (alembic.ini in the project
  root), and the database comes from API_DATABASE_URL / .env as for the API.

``render_as_batch``: SQLite cannot ALTER most of a table, so Alembic rebuilds
it (batch mode); migrations must use ``op.batch_alter_table`` for changes
other than adding a table or a nullable column.
"""

from alembic import context
from sqlalchemy import create_engine

import src.api.models  # noqa: F401  (registers every model on Base.metadata)
from src.api.db_base import Base, DatabaseSettings, sync_database_url

target_metadata = Base.metadata


def _sync_url() -> str:
    return sync_database_url(DatabaseSettings().database_url)


def run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if context.is_offline_mode():
    context.configure(url=_sync_url(), target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
elif connection is not None:
    run_migrations(connection)
else:
    engine = create_engine(_sync_url())
    with engine.connect() as conn:
        run_migrations(conn)
    engine.dispose()
