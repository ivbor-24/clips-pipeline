from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.app import app
from src.api.database import Base, get_db
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


@pytest.mark.asyncio
async def test_create_job(client):
    token = await login(client)
    response = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["input_source"] == "/path/to/video.mp4"
    assert data["status"] == "queued"
    assert "id" in data
    assert "work_dir" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_job_work_dir_stays_out_of_checkout(client, tmp_path):
    """Test that a queued job's work dir is made under the (temporary) jobs root."""
    checkout_jobs = Path("jobs")
    before = set(checkout_jobs.iterdir()) if checkout_jobs.is_dir() else set()

    token = await login(client)
    response = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )

    assert response.status_code == 201
    work_dir = Path(response.json()["work_dir"])
    assert work_dir.parent == tmp_path / "jobs"
    assert work_dir.is_dir()
    after = set(checkout_jobs.iterdir()) if checkout_jobs.is_dir() else set()
    assert after == before


@pytest.mark.asyncio
async def test_create_job_with_config_overrides(client):
    token = await login(client)
    response = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "input_source": "/path/to/video.mp4",
            "config_overrides": {"scoring": {"max_clips_per_video": 10}},
        },
    )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_create_job_unauthorized(client):
    response = await client.post(
        "/api/v1/jobs/",
        json={"input_source": "/path/to/video.mp4"},
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_list_jobs(client):
    token = await login(client)
    await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video1.mp4"},
    )
    await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video2.mp4"},
    )

    response = await client.get(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["jobs"]) == 2


@pytest.mark.asyncio
async def test_get_job_by_id(client):
    token = await login(client)
    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    response = await client.get(
        f"/api/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == job_id


@pytest.mark.asyncio
async def test_get_job_not_found(client):
    token = await login(client)
    response = await client.get(
        "/api/v1/jobs/9999",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cancel_job(client):
    token = await login(client)
    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    response = await client.delete(
        f"/api/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 204

    get_resp = await client.get(
        f"/api/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_completed_job(client, db_session):
    token = await login(client)
    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    from sqlalchemy import select

    from src.api.models.job import Job, JobStatus

    result = await db_session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one()
    job.status = JobStatus.COMPLETED
    await db_session.commit()

    response = await client.delete(
        f"/api/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_jobs_pagination(client):
    token = await login(client)
    for i in range(5):
        await client.post(
            "/api/v1/jobs/",
            headers={"Authorization": f"Bearer {token}"},
            json={"input_source": f"/path/to/video{i}.mp4"},
        )

    response = await client.get(
        "/api/v1/jobs/?limit=2&offset=0",
        headers={"Authorization": f"Bearer {token}"},
    )
    data = response.json()
    assert data["total"] == 5
    assert len(data["jobs"]) == 2

    response = await client.get(
        "/api/v1/jobs/?limit=2&offset=4",
        headers={"Authorization": f"Bearer {token}"},
    )
    data = response.json()
    assert data["total"] == 5
    assert len(data["jobs"]) == 1


async def _set_status(db_session, job_id, status):
    from src.api.models.job import Job

    job = await db_session.get(Job, job_id)
    job.status = status
    await db_session.commit()
    return job


@pytest.mark.asyncio
async def test_create_job_only_queues_it(client, db_session):
    """The API does not run jobs: the worker (src/worker.py) takes them from the queue."""
    token = await login(client)
    resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    from src.api.models.job import Job

    job = await db_session.get(Job, resp.json()["id"])
    assert job.status.value == "queued"
    assert job.started_at is None
    assert job.attempts == 0


@pytest.mark.asyncio
async def test_cancel_running_job_asks_the_worker(client, db_session):
    from src.api.models.job import JobStatus

    token = await login(client)
    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]
    job = await _set_status(db_session, job_id, JobStatus.RUNNING)

    response = await client.delete(
        f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 204
    await db_session.refresh(job)
    # Stays running until the worker has killed its processes.
    assert job.status == JobStatus.RUNNING
    assert job.cancel_requested is True
    # The web UI shows "Stopping..." from this flag.
    shown = await client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
    assert shown.json()["cancel_requested"] is True


@pytest.mark.asyncio
async def test_retry_failed_job_requeues_it(client, db_session):
    from sqlalchemy import select

    from src.api.models.job import JobEvent, JobStatus

    token = await login(client)
    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]
    job = await _set_status(db_session, job_id, JobStatus.FAILED)
    job.error_message = "boom"
    job.cancel_requested = True
    await db_session.commit()

    response = await client.post(
        f"/api/v1/jobs/{job_id}/retry", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    await db_session.refresh(job)
    assert job.error_message is None
    assert job.cancel_requested is False
    events = (await db_session.scalars(select(JobEvent).where(JobEvent.job_id == job_id))).all()
    assert [e.event_type for e in events] == ["job_requeued"]


@pytest.mark.asyncio
async def test_retry_running_job_is_rejected(client, db_session):
    from src.api.models.job import JobStatus

    token = await login(client)
    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]
    await _set_status(db_session, job_id, JobStatus.RUNNING)

    response = await client.post(
        f"/api/v1/jobs/{job_id}/retry", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 400
