from collections import namedtuple
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
async def client(db_session, tmp_path):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    from src.api.routes import upload

    original_upload_dir = upload.UPLOAD_DIR
    upload.UPLOAD_DIR = tmp_path / "uploads"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
    upload.UPLOAD_DIR = original_upload_dir


async def _upload(client, token, filename, content, **headers):
    return await client.post(
        "/api/v1/upload/",
        params={"filename": filename},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
            **headers,
        },
        content=content,
    )


@pytest.mark.asyncio
async def test_upload_video(client, tmp_path):
    token = await login(client)
    response = await _upload(client, token, "test_video.mp4", b"\x00" * 1024)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test_video.mp4"
    assert data["size"] == 1024
    assert Path(data["path"]).read_bytes() == b"\x00" * 1024
    assert str(tmp_path / "uploads") in str(Path(data["path"]).resolve())
    assert not list((tmp_path / "uploads").rglob("*.part"))


@pytest.mark.asyncio
async def test_upload_unauthorized(client):
    response = await client.post(
        "/api/v1/upload/", params={"filename": "test.mp4"}, content=b"\x00" * 16
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_upload_unsupported_type(client):
    token = await login(client)
    response = await _upload(client, token, "document.pdf", b"%PDF")
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_allows_various_video_formats(client):
    token = await login(client)
    for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".ts"]:
        response = await _upload(client, token, f"video{ext}", b"\x00" * 64)
        assert response.status_code == 201, f"Failed for extension {ext}"


@pytest.mark.asyncio
async def test_upload_needs_filename(client):
    token = await login(client)
    response = await client.post(
        "/api/v1/upload/",
        headers={"Authorization": f"Bearer {token}"},
        content=b"\x00" * 16,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_empty_body(client):
    token = await login(client)
    response = await _upload(client, token, "empty.mp4", b"")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_strips_directories_from_filename(client, tmp_path):
    token = await login(client)
    response = await _upload(client, token, "../../etc/my_lecture.mp4", b"\x00" * 16)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "my_lecture.mp4"
    assert Path(data["path"]).resolve().is_relative_to((tmp_path / "uploads").resolve())


class TestLimits:
    @pytest.mark.asyncio
    async def test_over_the_limit_is_refused_before_storing(self, client, tmp_path, monkeypatch):
        from src.api.config import settings

        monkeypatch.setattr(settings, "max_upload_gb", 1e-6)  # 1000 bytes
        token = await login(client)
        response = await _upload(client, token, "big.mp4", b"\x00" * 2000)
        assert response.status_code == 413
        assert "API_MAX_UPLOAD_GB" in response.json()["detail"]
        assert not [p for p in (tmp_path / "uploads").rglob("*") if p.is_file()]

    @pytest.mark.asyncio
    async def test_limit_holds_without_content_length(self, client, tmp_path, monkeypatch):
        """A chunked upload has no Content-Length; the stream is counted instead."""
        from src.api.config import settings

        monkeypatch.setattr(settings, "max_upload_gb", 1e-6)
        token = await login(client)

        async def chunks():
            for _ in range(4):
                yield b"\x00" * 600

        response = await _upload(client, token, "big.mp4", chunks())
        assert response.status_code == 413
        assert not [p for p in (tmp_path / "uploads").rglob("*") if p.is_file()]

    @pytest.mark.asyncio
    async def test_not_enough_disk_space(self, client, monkeypatch):
        from src.api.routes import upload

        Usage = namedtuple("Usage", "total used free")
        monkeypatch.setattr(upload.shutil, "disk_usage", lambda _p: Usage(0, 0, 10**9))
        token = await login(client)
        response = await client.get(
            "/api/v1/upload/check",
            params={"filename": "lecture.mp4", "size": 2 * 10**9},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 507
        assert "Not enough disk space" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_check_accepts_a_file_that_fits(self, client):
        token = await login(client)
        response = await client.get(
            "/api/v1/upload/check",
            params={"filename": "lecture.mp4", "size": 1024},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_check_rejects_unsupported_type(self, client):
        token = await login(client)
        response = await client.get(
            "/api/v1/upload/check",
            params={"filename": "notes.pdf", "size": 10},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 400


def test_sizes_in_messages_are_distinguishable():
    from src.api.routes.upload import _gb

    assert _gb(60_000_000) == "60 MB"
    assert _gb(10_040_000_000) == "10.04 GB"
    assert _gb(10_000_000_000) == "10.00 GB"
