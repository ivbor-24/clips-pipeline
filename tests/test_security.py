"""Tests for src/api/security.py: Host and Origin checks against other web sites."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.security import check_request, is_known_host, origin_allowed, split_host

CORS = ["http://localhost:5173", "http://127.0.0.1:5173"]


class TestHost:
    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "localhost:8080",
            "app.localhost",
            "127.0.0.1:8080",
            "[::1]:8080",
            "192.168.1.20:8080",
            "10.0.0.5",
        ],
    )
    def test_known_without_a_password(self, host):
        assert is_known_host(split_host(host)[0], [])

    @pytest.mark.parametrize(
        "host", ["evil.example", "evil.example:8080", "localhost.evil.example", ""]
    )
    def test_unknown(self, host):
        assert not is_known_host(split_host(host)[0], [])

    def test_allowed_list(self):
        assert is_known_host("studio-pc", ["studio-pc", " nas "])
        assert is_known_host("nas", ["studio-pc", " nas "])


class TestOrigin:
    def test_same_host_and_port(self):
        assert origin_allowed("http://127.0.0.1:8080", "127.0.0.1:8080", [])

    def test_default_port_dropped_by_a_proxy(self):
        assert origin_allowed("https://clips.lan", "clips.lan", [])

    def test_other_port_on_the_same_host(self):
        assert not origin_allowed("http://127.0.0.1:3000", "127.0.0.1:8080", [])

    def test_other_site(self):
        assert not origin_allowed("http://evil.example", "127.0.0.1:8080", [])
        assert not origin_allowed("null", "127.0.0.1:8080", [])

    def test_development_web_ui(self):
        assert origin_allowed("http://127.0.0.1:5173", "127.0.0.1:8000", CORS)


class TestCheckRequest:
    def _check(
        self,
        method="GET",
        path="/api/v1/jobs/",
        host="127.0.0.1:8080",
        origin=None,
        password=False,
        allowed=(),
    ):
        return check_request(method, path, host, origin, password, list(allowed), CORS)

    def test_rebinding_host_refused_without_a_password(self):
        assert "Unknown host name" in self._check(host="evil.example:8080")

    def test_password_mode_leaves_host_to_the_login(self):
        assert self._check(host="evil.example:8080", password=True) is None

    def test_cross_site_post_refused_in_both_modes(self):
        for password in (False, True):
            reason = self._check(
                "POST", "/api/v1/jobs/1/delete", origin="http://evil.example", password=password
            )
            assert "Cross-site" in reason

    def test_same_site_post_and_tools_without_origin(self):
        assert self._check("POST", "/api/v1/jobs/1/delete", origin="http://127.0.0.1:8080") is None
        assert self._check("POST", "/api/v1/jobs/1/delete") is None  # curl, scripts

    def test_cross_site_get_is_not_a_change(self):
        assert self._check("GET", origin="http://evil.example") is None

    def test_health_and_web_pages_are_not_checked(self):
        assert self._check(path="/api/v1/health", host="backend:8000") is None
        assert self._check(path="/index.html", host="evil.example") is None


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    from src.api.app import app
    from src.api.config import settings
    from src.api.database import Base, get_db

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(settings, "allowed_hosts", "")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:8080"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()
    await engine.dispose()


class TestApi:
    @pytest.mark.asyncio
    async def test_local_address_works(self, client):
        assert (await client.get("/api/v1/jobs/")).status_code == 200

    @pytest.mark.asyncio
    async def test_rebound_host_name_refused(self, client):
        response = await client.get("/api/v1/jobs/", headers={"Host": "evil.example:8080"})
        assert response.status_code == 403
        assert "API_ALLOWED_HOSTS" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_allowed_host_name(self, client, monkeypatch):
        from src.api.config import settings

        monkeypatch.setattr(settings, "allowed_hosts", "studio-pc")
        response = await client.get("/api/v1/jobs/", headers={"Host": "studio-pc:8080"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cross_site_delete_refused_before_the_route(self, client):
        response = await client.post(
            "/api/v1/jobs/1/delete", headers={"Origin": "http://evil.example"}
        )
        assert response.status_code == 403
        same_site = await client.post(
            "/api/v1/jobs/1/delete", headers={"Origin": "http://127.0.0.1:8080"}
        )
        assert same_site.status_code == 404  # reached the route: no such job

    @pytest.mark.asyncio
    async def test_health_answers_the_web_ui_container(self, client):
        response = await client.get("/api/v1/health", headers={"Host": "backend:8000"})
        assert response.status_code == 200
