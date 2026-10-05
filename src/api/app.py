from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.config import settings

logger = structlog.get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.api.database import create_tables
    from src.api.routes.upload import remove_stale_partial_uploads

    # Route structlog through stdlib so each job can mirror its records into
    # <work_dir>/artifacts/logs/job.log (see src.logger.job_log_file).
    from src.logger import setup_logging

    setup_logging(log_file=None, level="INFO", json_format=False)

    await create_tables()
    # Uploads cut off by a previous stop of the API.
    remove_stale_partial_uploads()
    logger.info("api_access", password_required=settings.password_required)
    # Shut down after idle time.
    import asyncio

    from src.api.database import async_session
    from src.api.shutdown import idle_watchdog

    watchdog = asyncio.create_task(idle_watchdog(async_session, settings.idle_shutdown_min))
    logger.info("idle_shutdown", after_minutes=settings.idle_shutdown_min or None)
    # Source videos of old jobs and orphan uploads: now and every 6 h.
    from src.api.services.cleanup import cleanup_loop

    cleanup = asyncio.create_task(cleanup_loop(async_session))
    # Jobs run in the job worker (python -m src.worker), not in this process,
    # so stopping or reloading the API does not touch them.
    yield
    watchdog.cancel()
    cleanup.cancel()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Video Clips Pipeline API",
        version=__version__,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    origins = settings.cors_origin_list

    @app.middleware("http")
    async def _note_activity(request, call_next):
        # Any page calling the API keeps the system on (idle shutdown);
        # the web UI container's health checks do not.
        if request.url.path.startswith("/api/") and request.url.path != "/api/v1/health":
            from src.api.shutdown import note_activity

            note_activity()
        return await call_next(request)

    @app.middleware("http")
    async def _check_host_and_origin(request, call_next):
        # Other sites in the same browser (DNS rebinding, CSRF): src/api/security.py.
        # Added after _note_activity, so it runs first and a refused request is
        # not activity.
        from fastapi.responses import JSONResponse

        from src.api.security import check_request

        reason = check_request(
            request.method,
            request.url.path,
            request.headers.get("host", ""),
            request.headers.get("origin"),
            settings.password_required,
            settings.allowed_host_list,
            settings.cors_origin_list,
        )
        if reason is not None:
            logger.warning(
                "request_refused",
                path=request.url.path,
                host=request.headers.get("host"),
                origin=request.headers.get("origin"),
            )
            return JSONResponse(status_code=403, content={"detail": reason})
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.api.routes import (
        artifacts,
        auth,
        cleanup,
        config,
        diagnostics,
        health,
        jobs,
        review,
        system,
        upload,
    )

    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
    app.include_router(config.router, prefix="/api/v1/config", tags=["config"])
    app.include_router(artifacts.router, prefix="/api/v1/jobs", tags=["artifacts"])
    app.include_router(review.router, prefix="/api/v1/jobs", tags=["review"])
    app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
    app.include_router(cleanup.router, prefix="/api/v1/cleanup", tags=["cleanup"])
    app.include_router(diagnostics.router, prefix="/api/v1/diagnostics", tags=["diagnostics"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])

    return app


app = create_app()
