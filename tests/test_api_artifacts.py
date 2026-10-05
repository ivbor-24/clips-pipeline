import json
from unittest.mock import patch

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


async def create_job_with_artifacts(client, token, tmp_path, artifacts=None, manifest=None):
    work_dir = tmp_path / "job_test"
    work_dir.mkdir(parents=True)
    (work_dir / "artifacts").mkdir()
    (work_dir / "output").mkdir()

    with patch("src.api.services.job.create_job") as mock_create:
        from src.api.models.job import Job, JobStatus

        job = Job(
            id=1,
            user_id=1,
            status=JobStatus.COMPLETED,
            input_source="/path/to/video.mp4",
            work_dir=str(work_dir),
        )
        mock_create.return_value = job

        create_resp = await client.post(
            "/api/v1/jobs/",
            headers={"Authorization": f"Bearer {token}"},
            json={"input_source": "/path/to/video.mp4"},
        )

    job_id = create_resp.json()["id"]

    if artifacts:
        for filename, data in artifacts.items():
            (work_dir / "artifacts" / filename).write_text(json.dumps(data))

    if manifest:
        (work_dir / "output" / "manifest.json").write_text(json.dumps(manifest))

    return job_id, work_dir


@pytest.mark.asyncio
async def test_get_transcript(client, tmp_path):
    token = await login(client)
    transcript_data = [{"start": 0.0, "end": 5.0, "text": "Hello world"}]

    work_dir = tmp_path / "job_test"
    work_dir.mkdir(parents=True)
    (work_dir / "artifacts").mkdir()
    (work_dir / "output").mkdir()
    (work_dir / "artifacts" / "transcript.json").write_text(json.dumps(transcript_data))

    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    from sqlalchemy import select

    from src.api.models.job import Job

    async for db in app.dependency_overrides[get_db]():
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        job.work_dir = str(work_dir)
        await db.commit()

    response = await client.get(
        f"/api/v1/jobs/{job_id}/transcript",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["segments"] == transcript_data


@pytest.mark.asyncio
async def test_get_scored_segments(client, tmp_path):
    token = await login(client)
    scored_data = [{"start": 0.0, "end": 5.0, "score": 0.95}]

    work_dir = tmp_path / "job_test"
    work_dir.mkdir(parents=True)
    (work_dir / "artifacts").mkdir()
    (work_dir / "output").mkdir()
    (work_dir / "artifacts" / "scored_segments.json").write_text(json.dumps(scored_data))

    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    from sqlalchemy import select

    from src.api.models.job import Job

    async for db in app.dependency_overrides[get_db]():
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        job.work_dir = str(work_dir)
        await db.commit()

    response = await client.get(
        f"/api/v1/jobs/{job_id}/scored_segments",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["segments"] == scored_data


@pytest.mark.asyncio
async def test_get_crop_params(client, tmp_path):
    token = await login(client)
    crops_data = [{"x": 100, "y": 200, "w": 300, "h": 400}]

    work_dir = tmp_path / "job_test"
    work_dir.mkdir(parents=True)
    (work_dir / "artifacts").mkdir()
    (work_dir / "output").mkdir()
    (work_dir / "artifacts" / "crop_params.json").write_text(json.dumps(crops_data))

    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    from sqlalchemy import select

    from src.api.models.job import Job

    async for db in app.dependency_overrides[get_db]():
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        job.work_dir = str(work_dir)
        await db.commit()

    response = await client.get(
        f"/api/v1/jobs/{job_id}/crop_params",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["crops"] == crops_data


@pytest.mark.asyncio
async def test_get_manifest(client, tmp_path):
    token = await login(client)
    manifest_data = {"clips": [{"path": "clip_001.mp4", "duration": 30}], "total_clips": 1}

    work_dir = tmp_path / "job_test"
    work_dir.mkdir(parents=True)
    (work_dir / "artifacts").mkdir()
    (work_dir / "output").mkdir()
    (work_dir / "output" / "manifest.json").write_text(json.dumps(manifest_data))

    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    from sqlalchemy import select

    from src.api.models.job import Job

    async for db in app.dependency_overrides[get_db]():
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        job.work_dir = str(work_dir)
        await db.commit()

    response = await client.get(
        f"/api/v1/jobs/{job_id}/manifest",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_clips"] == 1
    assert data["clips"] == [{"path": "clip_001.mp4", "duration": 30}]


@pytest.mark.asyncio
async def test_artifact_not_found(client, tmp_path):
    token = await login(client)

    work_dir = tmp_path / "job_test"
    work_dir.mkdir(parents=True)
    (work_dir / "artifacts").mkdir()
    (work_dir / "output").mkdir()

    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    from sqlalchemy import select

    from src.api.models.job import Job

    async for db in app.dependency_overrides[get_db]():
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        job.work_dir = str(work_dir)
        await db.commit()

    response = await client.get(
        f"/api/v1/jobs/{job_id}/transcript",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404
    assert "transcript.json" in response.json()["detail"]


@pytest.mark.asyncio
async def test_manifest_not_found(client, tmp_path):
    token = await login(client)

    work_dir = tmp_path / "job_test"
    work_dir.mkdir(parents=True)
    (work_dir / "artifacts").mkdir()
    (work_dir / "output").mkdir()

    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]

    from sqlalchemy import select

    from src.api.models.job import Job

    async for db in app.dependency_overrides[get_db]():
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        job.work_dir = str(work_dir)
        await db.commit()

    response = await client.get(
        f"/api/v1/jobs/{job_id}/manifest",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404
    assert "Manifest" in response.json()["detail"]


@pytest.mark.asyncio
async def test_job_not_found(client):
    token = await login(client)

    response = await client.get(
        "/api/v1/jobs/9999/transcript",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_unauthorized_access(client):
    response = await client.get("/api/v1/jobs/1/transcript")
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_notices(client, tmp_path):
    from sqlalchemy import select

    from src.api.models.job import Job
    from src.notices import add_notice

    token = await login(client)
    work_dir = tmp_path / "job_notices"
    create_resp = await client.post(
        "/api/v1/jobs/",
        headers={"Authorization": f"Bearer {token}"},
        json={"input_source": "/path/to/video.mp4"},
    )
    job_id = create_resp.json()["id"]
    async for db in app.dependency_overrides[get_db]():
        job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one()
        job.work_dir = str(work_dir)
        await db.commit()

    headers = {"Authorization": f"Bearer {token}"}
    empty = await client.get(f"/api/v1/jobs/{job_id}/notices", headers=headers)
    assert empty.json() == {"notices": []}

    add_notice(work_dir, "scoring", "llm_failed_heuristics_used", "Heuristics.", error="OOM")
    response = await client.get(f"/api/v1/jobs/{job_id}/notices", headers=headers)
    assert response.status_code == 200
    assert response.json()["notices"] == [
        {
            "stage": "scoring",
            "code": "llm_failed_heuristics_used",
            "message": "Heuristics.",
            "error": "OOM",
        }
    ]


async def add_finished_job(db_session, work_dir) -> int:
    """A completed chapters job whose files live in work_dir."""
    from src.api.models.job import Job, JobStatus, JobType
    from src.api.services.auth import LOCAL_USER_ID

    job = Job(
        user_id=LOCAL_USER_ID,
        status=JobStatus.COMPLETED,
        job_type=JobType.CHAPTERS,
        input_source="/path/to/video.mp4",
        work_dir=str(work_dir),
    )
    db_session.add(job)
    await db_session.commit()
    return job.id


class TestChaptersAndBroll:
    """Endpoints behind the chapters and b-roll panels of the job page."""

    @pytest.mark.asyncio
    async def test_chapters(self, client, db_session, tmp_path):
        chapters = [{"timestamp": "00:00", "title": "Intro", "summary": "Why it matters"}]
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "chapters.json").write_text(
            json.dumps({"chapters": chapters, "source": "lecture.mp4"})
        )
        (artifacts / "chapters.txt").write_text("00:00 Intro\n")
        job_id = await add_finished_job(db_session, tmp_path)
        headers = {"Authorization": f"Bearer {await login(client)}"}

        response = await client.get(f"/api/v1/jobs/{job_id}/chapters", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"chapters": chapters, "source": "lecture.mp4"}

        response = await client.get(f"/api/v1/jobs/{job_id}/chapters/text", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"text": "00:00 Intro\n", "format": "youtube"}

    @pytest.mark.asyncio
    async def test_broll(self, client, db_session, tmp_path):
        suggestions = [{"timestamp": "01:30", "suggestion": "Lab bench", "keywords": ["lab"]}]
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "broll_suggestions.json").write_text(
            json.dumps({"suggestions": suggestions, "source": "lecture.mp4"})
        )
        (artifacts / "broll_suggestions.txt").write_text("01:30 Lab bench\n")
        job_id = await add_finished_job(db_session, tmp_path)
        headers = {"Authorization": f"Bearer {await login(client)}"}

        response = await client.get(f"/api/v1/jobs/{job_id}/broll", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"suggestions": suggestions, "source": "lecture.mp4"}

        response = await client.get(f"/api/v1/jobs/{job_id}/broll/text", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"text": "01:30 Lab bench\n"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["chapters", "chapters/text", "broll", "broll/text"])
    async def test_missing_file_is_404(self, client, db_session, tmp_path, path):
        # B-roll is optional, so a finished job may have no suggestions at all.
        job_id = await add_finished_job(db_session, tmp_path)
        headers = {"Authorization": f"Bearer {await login(client)}"}

        response = await client.get(f"/api/v1/jobs/{job_id}/{path}", headers=headers)
        assert response.status_code == 404


class TestJobLog:
    """GET /jobs/{id}/log: the job's log, also once the job has ended."""

    async def make_job(self, client, tmp_path):
        from sqlalchemy import select

        from src.api.models.job import Job

        token = await login(client)
        headers = {"Authorization": f"Bearer {token}"}
        create_resp = await client.post(
            "/api/v1/jobs/", headers=headers, json={"input_source": "/path/to/video.mp4"}
        )
        job_id = create_resp.json()["id"]
        work_dir = tmp_path / "job_log"
        async for db in app.dependency_overrides[get_db]():
            job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one()
            job.work_dir = str(work_dir)
            await db.commit()
        log = work_dir / "artifacts" / "logs" / "job.log"
        log.parent.mkdir(parents=True)
        return job_id, headers, log

    @pytest.mark.asyncio
    async def test_lines_and_incremental_reads(self, client, tmp_path):
        job_id, headers, log = await self.make_job(client, tmp_path)
        log.write_text(
            json.dumps(
                {
                    "event": "llama_cpp_gpu_offload",
                    "offloaded": 41,
                    "layers": 41,
                    "job_id": job_id,
                    "level": "info",
                    "timestamp": "2026-10-01T08:00:00Z",
                }
            )
            + "\n"
            + '{"event": "half a line", "lev'
        )

        first = (await client.get(f"/api/v1/jobs/{job_id}/log", headers=headers)).json()
        assert first["lines"] == [
            {
                "timestamp": "2026-10-01T08:00:00Z",
                "level": "info",
                "event": "llama_cpp_gpu_offload",
                "fields": {"offloaded": 41, "layers": 41},
            }
        ]

        # The half-written line is returned once it is complete.
        with open(log, "a") as f:
            f.write('el": "warning"}\n')
        offset = first["offset"]
        second = (
            await client.get(f"/api/v1/jobs/{job_id}/log?offset={offset}", headers=headers)
        ).json()
        assert [(line["event"], line["level"]) for line in second["lines"]] == [
            ("half a line", "warning")
        ]
        third = (
            await client.get(
                f"/api/v1/jobs/{job_id}/log?offset={second['offset']}", headers=headers
            )
        ).json()
        assert third["lines"] == []

    @pytest.mark.asyncio
    async def test_no_log_yet(self, client, tmp_path):
        job_id, headers, _ = await self.make_job(client, tmp_path)
        response = await client.get(f"/api/v1/jobs/{job_id}/log", headers=headers)
        assert response.json() == {"lines": [], "offset": 0}

    @pytest.mark.asyncio
    async def test_shorter_log_starts_over(self, client, tmp_path):
        job_id, headers, log = await self.make_job(client, tmp_path)
        log.write_text('{"event": "new run"}\n')
        response = await client.get(f"/api/v1/jobs/{job_id}/log?offset=99999", headers=headers)
        assert [line["event"] for line in response.json()["lines"]] == ["new run"]
