"""Shared helpers for API tests that run with a shared password."""

import pytest
from httpx import AsyncClient
from pydantic import SecretStr

TEST_PASSWORD = "test-password"
TEST_SECRET = "test-secret-that-is-at-least-16-bytes"


@pytest.fixture
def password_mode(monkeypatch):
    """Require the shared password, as on an installation opened to the network."""
    from src.api.config import settings

    monkeypatch.setattr(settings, "password", SecretStr(TEST_PASSWORD))
    monkeypatch.setattr(settings, "jwt_secret", SecretStr(TEST_SECRET))


async def login(client: AsyncClient) -> str:
    """Exchange the shared password for a session token."""
    response = await client.post("/api/v1/auth/login", json={"password": TEST_PASSWORD})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]
