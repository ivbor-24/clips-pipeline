import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.config import settings
from src.api.db_base import Base, enable_sqlite_concurrency, sync_database_url

__all__ = ["Base", "async_session", "create_tables", "engine", "get_db"]

logger = structlog.get_logger("database")

engine = create_async_engine(settings.database_url, echo=False)
enable_sqlite_concurrency(engine.sync_engine)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def create_tables():
    """Bring the database to the newest schema (migrations, src/api/migrate.py)."""
    import asyncio

    from src.api.migrate import upgrade_connection, upgrade_database
    from src.api.models.job import Job
    from src.api.services.auth import LOCAL_USER_ID

    # The database of this engine (tests replace it), not a fresh read of the settings.
    url = engine.url
    if url.get_backend_name() == "sqlite" and url.database in (None, "", ":memory:"):
        async with engine.begin() as conn:
            await conn.run_sync(upgrade_connection)
    else:
        sync_url = sync_database_url(url.render_as_string(hide_password=False))
        await asyncio.to_thread(upgrade_database, sync_url)
    async with engine.begin() as conn:
        adopted = await conn.execute(
            update(Job).where(Job.user_id != LOCAL_USER_ID).values(user_id=LOCAL_USER_ID)
        )
    if adopted.rowcount:
        # Single-user mode: jobs made under the old per-user
        # accounts belong to the built-in user.
        logger.info("jobs_assigned_to_local_user", jobs=adopted.rowcount)


async def get_db():
    async with async_session() as session:
        yield session
