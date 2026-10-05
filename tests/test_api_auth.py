"""Access to the API: one built-in user, optional shared password."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.app import app
from src.api.config import APISettings, settings
from src.api.database import Base, get_db
from src.api.models.job import Job
from src.api.services.auth import LOCAL_USER_ID
from tests.api_auth_helpers import TEST_PASSWORD, TEST_SECRET, login, password_mode  # noqa: F401


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
async def test_health_check(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.8.0"


class TestNoPassword:
    """Default: no API_PASSWORD, every request is the built-in user."""

    @pytest.mark.asyncio
    async def test_me_without_token(self, client):
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 200
        assert response.json() == {"username": "local", "password_required": False}

    @pytest.mark.asyncio
    async def test_jobs_work_without_token(self, client, db_session):
        created = await client.post("/api/v1/jobs/", json={"input_source": "/videos/a.mp4"})
        assert created.status_code == 201
        job_id = created.json()["id"]

        assert (await client.get(f"/api/v1/jobs/{job_id}")).status_code == 200
        listed = await client.get("/api/v1/jobs/")
        assert listed.json()["total"] == 1
        job = await db_session.scalar(select(Job).where(Job.id == job_id))
        assert job.user_id == LOCAL_USER_ID

    @pytest.mark.asyncio
    async def test_stale_token_is_ignored(self, client):
        # A browser may still hold a token from a time the password was on.
        response = await client.get(
            "/api/v1/jobs/", headers={"Authorization": "Bearer stale-token"}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_login_is_off(self, client):
        response = await client.post("/api/v1/auth/login", json={"password": "anything"})
        assert response.status_code == 400
        assert "API_PASSWORD" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_registration_is_gone(self, client):
        response = await client.post(
            "/api/v1/auth/register",
            json={"username": "u", "email": "u@example.com", "password": "secret123"},
        )
        assert response.status_code in (404, 405)


@pytest.mark.usefixtures("password_mode")
class TestPassword:
    """API_PASSWORD set: every request needs the session token, localhost too."""

    @pytest.mark.asyncio
    async def test_login_returns_session_token(self, client):
        response = await client.post("/api/v1/auth/login", json={"password": TEST_PASSWORD})
        assert response.status_code == 200
        data = response.json()
        assert data["token_type"] == "bearer"
        assert data["access_token"]

    @pytest.mark.asyncio
    async def test_wrong_password(self, client):
        response = await client.post("/api/v1/auth/login", json={"password": "wrong"})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_requests_without_token_are_refused(self, client):
        assert (await client.get("/api/v1/auth/me")).status_code == 401
        assert (await client.get("/api/v1/jobs/")).status_code == 401
        response = await client.post("/api/v1/jobs/", json={"input_source": "/videos/a.mp4"})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_token(self, client):
        token = await login(client)
        response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json() == {"username": "local", "password_required": True}

    @pytest.mark.asyncio
    async def test_query_token_for_eventsource_and_video(self, client):
        token = await login(client)
        response = await client.get("/api/v1/jobs/", params={"token": token})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_token(self, client):
        response = await client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer not-a-token"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_changing_password_signs_everyone_out(self, client, monkeypatch):
        token = await login(client)
        monkeypatch.setattr(settings, "password", SecretStr("new-password"))
        response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401


class TestSettings:
    def test_password_needs_a_secret(self):
        with pytest.raises(ValidationError, match="API_JWT_SECRET"):
            APISettings(password="pw", jwt_secret=None, _env_file=None)

    def test_error_does_not_print_the_password(self):
        with pytest.raises(ValidationError) as excinfo:
            APISettings(password="very-secret-pw", jwt_secret="short", _env_file=None)
        assert "secret-pw" not in str(excinfo.value)

    def test_placeholder_secret_is_refused(self):
        with pytest.raises(ValidationError, match="API_JWT_SECRET"):
            APISettings(password="pw", jwt_secret="change-me-in-production", _env_file=None)

    def test_no_password_needs_no_secret(self):
        assert not APISettings(jwt_secret=None, _env_file=None).password_required

    def test_reads_env_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("API_PASSWORD", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "GPU_BACKEND=cpu\n"  # non-API keys share the file
            "HF_TOKEN=\n"
            f"API_PASSWORD={TEST_PASSWORD}\n"
            f"API_JWT_SECRET={TEST_SECRET}\n"
            "API_SESSION_DAYS=7\n"
        )
        loaded = APISettings(_env_file=env_file)
        assert loaded.password_required
        assert loaded.session_days == 7

    def test_environment_wins_over_env_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_PASSWORD", "")
        env_file = tmp_path / ".env"
        env_file.write_text(f"API_PASSWORD={TEST_PASSWORD}\n")
        assert not APISettings(_env_file=env_file).password_required

    def test_default_env_file_is_the_project_one(self):
        from src.api.db_base import PROJECT_ROOT

        assert APISettings.model_config["env_file"] == PROJECT_ROOT / ".env"
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_worker_and_api_open_the_same_database(self, tmp_path, monkeypatch):
        """The worker must find the jobs the API queued, .env or not."""
        from src.api.db_base import DatabaseSettings

        monkeypatch.delenv("API_DATABASE_URL", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("API_DATABASE_URL=sqlite+aiosqlite:///./elsewhere/api.db\n")
        api_url = APISettings(_env_file=env_file).database_url
        assert api_url == DatabaseSettings(_env_file=env_file).database_url
        assert api_url.endswith("elsewhere/api.db")


@pytest.mark.asyncio
async def test_jobs_of_old_accounts_move_to_the_local_user(monkeypatch):
    """Databases from the multi-user version keep their jobs visible."""
    import src.api.database as database

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    monkeypatch.setattr(database, "engine", engine)
    await database.create_tables()
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        session.add_all(
            [Job(user_id=uid, input_source="a.mp4", work_dir=f"jobs/{uid}") for uid in (2, 7)]
        )
        await session.commit()

    await database.create_tables()

    async with session_factory() as session:
        owners = (await session.scalars(select(Job.user_id))).all()
    assert owners == [LOCAL_USER_ID, LOCAL_USER_ID]
    await engine.dispose()
