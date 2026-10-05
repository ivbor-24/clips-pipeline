"""Tests for the job progress stream (SSE).

The job worker writes events to the job_events table; the stream replays them
for the current attempt and ends on the job's final event.
"""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.app import app
from src.api.database import Base, get_db
from src.api.models.job import Job, JobEvent, JobStatus
from tests.api_auth_helpers import login, password_mode  # noqa: F401

pytestmark = pytest.mark.usefixtures("password_mode")


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def create_job(client, token) -> int:
    resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    return resp.json()["id"]


async def add_events(db_session, job_id, *events, status=None):
    """Write events as the worker would, optionally with the job's new status."""
    for event_type, data in events:
        db_session.add(JobEvent(job_id=job_id, event_type=event_type, data=json.dumps(data)))
    if status is not None:
        job = await db_session.get(Job, job_id)
        job.status = status
    await db_session.commit()


def event_types(body: str) -> list:
    return [line[len("event: ") :] for line in body.split("\n") if line.startswith("event: ")]


@pytest.mark.asyncio
async def test_sse_streams_worker_events(client, db_session):
    token = await login(client)
    job_id = await create_job(client, token)
    await add_events(
        db_session,
        job_id,
        ("job_started", {"job_id": job_id}),
        ("stage_started", {"stage": "ingestion", "status": "started"}),
        ("stage_completed", {"stage": "ingestion", "status": "completed", "progress": 100.0}),
        ("job_completed", {"job_id": job_id}),
        status=JobStatus.COMPLETED,
    )

    response = await client.get(
        f"/api/v1/jobs/{job_id}/progress", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert event_types(response.text) == [
        "job_status",
        "job_started",
        "stage_started",
        "stage_completed",
        "job_completed",
    ]
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            assert isinstance(json.loads(line[len("data: ") :]), dict)


@pytest.mark.asyncio
async def test_sse_ends_on_job_failed_with_error(client, db_session):
    token = await login(client)
    job_id = await create_job(client, token)
    await add_events(
        db_session,
        job_id,
        ("job_failed", {"job_id": job_id, "error": "boom"}),
        status=JobStatus.FAILED,
    )

    response = await client.get(
        f"/api/v1/jobs/{job_id}/progress", headers={"Authorization": f"Bearer {token}"}
    )
    assert event_types(response.text)[-1] == "job_failed"
    assert "boom" in response.text


@pytest.mark.asyncio
async def test_sse_ends_on_job_cancelled(client, db_session):
    token = await login(client)
    job_id = await create_job(client, token)
    await client.delete(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})

    response = await client.get(
        f"/api/v1/jobs/{job_id}/progress", headers={"Authorization": f"Bearer {token}"}
    )
    assert event_types(response.text) == ["job_status", "job_cancelled"]


@pytest.mark.asyncio
async def test_sse_replays_only_the_current_attempt(client, db_session):
    """After a retry the old job_failed must not close the new stream."""
    token = await login(client)
    job_id = await create_job(client, token)
    await add_events(
        db_session,
        job_id,
        ("stage_failed", {"stage": "scoring", "status": "failed"}),
        ("job_failed", {"job_id": job_id, "error": "old"}),
        ("job_requeued", {"reason": "retry"}),
        ("job_started", {"job_id": job_id}),
        ("stage_skipped", {"stage": "ingestion", "status": "skipped"}),
        ("job_completed", {"job_id": job_id}),
        status=JobStatus.COMPLETED,
    )

    response = await client.get(
        f"/api/v1/jobs/{job_id}/progress", headers={"Authorization": f"Bearer {token}"}
    )
    assert event_types(response.text) == [
        "job_status",
        "job_started",
        "stage_skipped",
        "job_completed",
    ]
    assert "old" not in response.text


@pytest.mark.asyncio
async def test_sse_finished_job_without_events_still_ends(client, db_session):
    """Jobs from before the worker have no events; the stream must not hang."""
    token = await login(client)
    job_id = await create_job(client, token)
    await add_events(db_session, job_id, status=JobStatus.COMPLETED)

    response = await client.get(
        f"/api/v1/jobs/{job_id}/progress", headers={"Authorization": f"Bearer {token}"}
    )
    assert event_types(response.text) == ["job_status", "job_completed"]


@pytest.mark.asyncio
async def test_sse_unauthorized(client):
    response = await client.get("/api/v1/jobs/1/progress")
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_sse_job_not_found(client):
    token = await login(client)
    response = await client.get(
        "/api/v1/jobs/9999/progress", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404
