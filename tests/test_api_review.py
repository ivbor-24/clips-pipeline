import json

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


async def create_job_with_clips(client, token, tmp_path, clips=None):
    work_dir = tmp_path / "job_test"
    work_dir.mkdir(parents=True)
    (work_dir / "artifacts").mkdir()
    (work_dir / "output" / "clips").mkdir(parents=True)

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

    if clips:
        for clip_id, meta in clips.items():
            meta_file = work_dir / "output" / "clips" / f"{clip_id}.meta.json"
            meta_file.write_text(json.dumps(meta))
            video_file = work_dir / "output" / "clips" / f"{clip_id}.mp4"
            video_file.write_bytes(b"\x00" * 1024)

    return job_id, work_dir


@pytest.mark.asyncio
async def test_list_clips(client, tmp_path):
    token = await login(client)
    clips = {
        "clip_001": {"score": 0.9, "duration": 30.0, "tags": ["intro"], "review_status": "pending"},
        "clip_002": {"score": 0.8, "duration": 25.0, "tags": ["main"], "review_status": "keep"},
    }
    job_id, work_dir = await create_job_with_clips(client, token, tmp_path, clips)

    (work_dir / "output" / "clips" / "clip_001.srt").write_text("subtitle content")

    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["clips"][0]["id"] == "clip_001"
    assert data["clips"][0]["score"] == 0.9
    assert data["clips"][0]["srt_path"] == "output/clips/clip_001.srt"
    assert data["clips"][1]["id"] == "clip_002"
    assert data["clips"][1]["srt_path"] is None


@pytest.mark.asyncio
async def test_list_clips_empty(client, tmp_path):
    token = await login(client)
    job_id, _ = await create_job_with_clips(client, token, tmp_path)

    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["clips"] == []


@pytest.mark.asyncio
async def test_update_clip(client, tmp_path):
    token = await login(client)
    clips = {
        "clip_001": {"score": 0.9, "duration": 30.0, "tags": [], "review_status": "pending"},
    }
    job_id, work_dir = await create_job_with_clips(client, token, tmp_path, clips)

    response = await client.put(
        f"/api/v1/jobs/{job_id}/clips/clip_001",
        headers={"Authorization": f"Bearer {token}"},
        json={"review_status": "keep", "notes": "Great clip"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["review_status"] == "keep"

    assert data["review_notes"] == "Great clip"

    # The decision lives in the database; rendering's metadata file is not touched.
    meta_file = work_dir / "output" / "clips" / "clip_001.meta.json"
    assert json.loads(meta_file.read_text())["review_status"] == "pending"
    listed = await client.get(
        f"/api/v1/jobs/{job_id}/clips", headers={"Authorization": f"Bearer {token}"}
    )
    assert listed.json()["clips"][0]["review_status"] == "keep"


@pytest.mark.asyncio
async def test_update_clip_no_notes(client, tmp_path):
    token = await login(client)
    clips = {
        "clip_001": {"score": 0.9, "duration": 30.0, "tags": [], "review_status": "pending"},
    }
    job_id, _ = await create_job_with_clips(client, token, tmp_path, clips)

    response = await client.put(
        f"/api/v1/jobs/{job_id}/clips/clip_001",
        headers={"Authorization": f"Bearer {token}"},
        json={"review_status": "reject"},
    )
    assert response.status_code == 200
    assert response.json()["review_status"] == "reject"


@pytest.mark.asyncio
async def test_stream_video(client, tmp_path):
    token = await login(client)
    clips = {
        "clip_001": {"score": 0.9, "duration": 30.0, "tags": [], "review_status": "pending"},
    }
    job_id, _ = await create_job_with_clips(client, token, tmp_path, clips)

    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips/clip_001/video",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert int(response.headers["content-length"]) == 1024
    assert "content-disposition" not in response.headers


@pytest.mark.asyncio
async def test_video_download_param(client, tmp_path):
    """?download=1 asks the browser to save the clip instead of playing it."""
    token = await login(client)
    clips = {
        "clip_001": {"score": 0.9, "duration": 30.0, "tags": [], "review_status": "pending"},
    }
    job_id, _ = await create_job_with_clips(client, token, tmp_path, clips)

    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips/clip_001/video?download=1",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="clip_001.mp4"'


@pytest.mark.asyncio
async def test_download_subtitles(client, tmp_path):
    token = await login(client)
    clips = {
        "clip_001": {"score": 0.9, "duration": 30.0, "tags": [], "review_status": "pending"},
    }
    job_id, work_dir = await create_job_with_clips(client, token, tmp_path, clips)
    (work_dir / "output" / "clips" / "clip_001.srt").write_text("1\n00:00:00,000 --> ...\nhi\n")

    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips/clip_001/subtitles",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="clip_001.srt"'
    assert "00:00:00" in response.text


@pytest.mark.asyncio
async def test_download_subtitles_missing_file(client, tmp_path):
    token = await login(client)
    clips = {
        "clip_001": {"score": 0.9, "duration": 30.0, "tags": [], "review_status": "pending"},
    }
    job_id, _ = await create_job_with_clips(client, token, tmp_path, clips)

    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips/clip_001/subtitles",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404
    assert "Subtitles file not found" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("clip_id", ["clip_..", "clip_%2e%2e", "clip_.%2e"])
async def test_download_subtitles_refuses_bad_ids(client, tmp_path, clip_id):
    """A clip id that could traverse the filesystem is rejected, not resolved.

    A bare ".." segment is normalized away by HTTP clients and routers before
    this route is ever reached, so the attempts that must be handled here are
    ids with dots inside them and percent-encoded ones.
    """
    token = await login(client)
    clips = {
        "clip_001": {"score": 0.9, "duration": 30.0, "tags": [], "review_status": "pending"},
    }
    job_id, _ = await create_job_with_clips(client, token, tmp_path, clips)

    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips/{clip_id}/subtitles",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
    assert "Invalid clip id" in response.json()["detail"]


@pytest.mark.asyncio
async def test_clip_not_found(client, tmp_path):
    token = await login(client)
    job_id, _ = await create_job_with_clips(client, token, tmp_path)

    response = await client.put(
        f"/api/v1/jobs/{job_id}/clips/clip_999",
        headers={"Authorization": f"Bearer {token}"},
        json={"review_status": "keep"},
    )
    assert response.status_code == 404
    assert "Clip not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_video_not_found(client, tmp_path):
    token = await login(client)
    job_id, _ = await create_job_with_clips(client, token, tmp_path)

    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips/clip_999/video",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404
    assert "Video file not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_job_not_found(client):
    token = await login(client)

    response = await client.get(
        "/api/v1/jobs/9999/clips",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_unauthorized_access(client):
    response = await client.get("/api/v1/jobs/1/clips")
    assert response.status_code in (401, 403)


async def _review(client, token, job_id, clip_id, status, notes=None):
    body = {"review_status": status}
    if notes is not None:
        body["notes"] = notes
    response = await client.put(
        f"/api/v1/jobs/{job_id}/clips/{clip_id}",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
    )
    assert response.status_code == 200, response.text
    return response.json()


async def _clips(client, token, job_id):
    response = await client.get(
        f"/api/v1/jobs/{job_id}/clips", headers={"Authorization": f"Bearer {token}"}
    )
    return {c["id"]: c for c in response.json()["clips"]}


CLIP_A = {"score": 0.9, "duration_sec": 60.0, "start_time": 100.0, "end_time": 160.0}


class TestReviewsInDatabase:
    """Review decisions survive re-rendering (they used to live in meta.json)."""

    @pytest.mark.asyncio
    async def test_review_survives_rerender(self, client, tmp_path):
        token = await login(client)
        job_id, work_dir = await create_job_with_clips(
            client, token, tmp_path, {"clip_001": CLIP_A}
        )
        await _review(client, token, job_id, "clip_001", "keep", notes="good hook")

        # Rendering rewrites the metadata file for the same clip.
        meta_file = work_dir / "output" / "clips" / "clip_001.meta.json"
        meta_file.write_text(json.dumps({**CLIP_A, "render_time_sec": 12.0}))

        clip = (await _clips(client, token, job_id))["clip_001"]
        assert clip["review_status"] == "keep"
        assert clip["review_notes"] == "good hook"

    @pytest.mark.asyncio
    async def test_other_content_under_the_same_id_is_pending(self, client, tmp_path):
        token = await login(client)
        job_id, work_dir = await create_job_with_clips(
            client, token, tmp_path, {"clip_001": CLIP_A}
        )
        await _review(client, token, job_id, "clip_001", "reject", notes="no hook")

        # Re-scoring picked another fragment and rendered it as clip_001.
        moved = {**CLIP_A, "start_time": 900.0, "end_time": 950.0}
        (work_dir / "output" / "clips" / "clip_001.meta.json").write_text(json.dumps(moved))

        clip = (await _clips(client, token, job_id))["clip_001"]
        assert clip["review_status"] == "pending"
        assert clip["review_notes"] is None

        updated = await _review(client, token, job_id, "clip_001", "keep")
        assert updated["review_status"] == "keep"
        assert updated["review_notes"] is None  # the old notes were about the old clip

    @pytest.mark.asyncio
    async def test_notes_kept_when_not_sent(self, client, tmp_path):
        token = await login(client)
        job_id, _ = await create_job_with_clips(client, token, tmp_path, {"clip_001": CLIP_A})
        await _review(client, token, job_id, "clip_001", "edit", notes="cut the first line")
        updated = await _review(client, token, job_id, "clip_001", "keep")
        assert updated["review_notes"] == "cut the first line"

    @pytest.mark.asyncio
    async def test_notes_only_preserves_existing_status(self, client, tmp_path):
        """A notes-only PUT must not rewrite the decision already made."""
        token = await login(client)
        job_id, _ = await create_job_with_clips(client, token, tmp_path, {"clip_001": CLIP_A})
        await _review(client, token, job_id, "clip_001", "keep")

        response = await client.put(
            f"/api/v1/jobs/{job_id}/clips/clip_001",
            headers={"Authorization": f"Bearer {token}"},
            json={"notes": "needs a better hook"},
        )
        assert response.status_code == 200
        assert response.json()["review_status"] == "keep"
        assert response.json()["review_notes"] == "needs a better hook"

    @pytest.mark.asyncio
    async def test_notes_on_a_clip_without_a_decision(self, client, tmp_path):
        """Notes typed before any decision keep the clip pending."""
        token = await login(client)
        job_id, _ = await create_job_with_clips(client, token, tmp_path, {"clip_001": CLIP_A})

        response = await client.put(
            f"/api/v1/jobs/{job_id}/clips/clip_001",
            headers={"Authorization": f"Bearer {token}"},
            json={"notes": "maybe start later"},
        )
        assert response.status_code == 200
        assert response.json()["review_status"] == "pending"
        assert response.json()["review_notes"] == "maybe start later"

    @pytest.mark.asyncio
    async def test_decisions_from_meta_json_are_imported_once(self, client, tmp_path):
        token = await login(client)
        legacy = {**CLIP_A, "review_status": "edit", "review_notes": "trim end"}
        job_id, work_dir = await create_job_with_clips(
            client, token, tmp_path, {"clip_001": legacy, "clip_002": CLIP_A}
        )

        clips = await _clips(client, token, job_id)
        assert clips["clip_001"]["review_status"] == "edit"
        assert clips["clip_001"]["review_notes"] == "trim end"
        assert clips["clip_002"]["review_status"] == "pending"

        # A later render drops the old fields; the imported decision stays.
        (work_dir / "output" / "clips" / "clip_001.meta.json").write_text(json.dumps(CLIP_A))
        clips = await _clips(client, token, job_id)
        assert clips["clip_001"]["review_status"] == "edit"
