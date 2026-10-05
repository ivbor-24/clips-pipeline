from unittest.mock import patch

import pytest
import pytest_asyncio
import yaml
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.app import app
from src.api.database import Base, get_db
from src.user_settings import user_settings_path
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


@pytest.fixture
def tmp_config(tmp_path):
    config_data = {
        "profiles_dir": "config/profiles",
        "input": {"default_source": "", "cache_dir": "artifacts/cache"},
        "preprocessing": {
            "audio_sample_rate": 16000,
            "audio_channels": 1,
            "video_max_height": 1080,
            "video_codec": "h264",
            "video_crf": 23,
        },
        "transcription": {
            "model": "large-v3-turbo",
            "language": "auto",
            "batch_size": 8,
            "compute_type": "float16",
            "chunk_duration_min": 30,
        },
        "scoring": {
            "min_duration": 45,
            "max_duration": 90,
            "min_score_threshold": 0.55,
            "max_clips_per_video": 5,
            "term_extraction_method": "keybert",
            "llm": {
                "enabled": False,
                "provider": "openai",
                "model": "gpt-4",
                "api_key": None,
                "api_base": None,
                "temperature": 0.7,
                "max_tokens": 1000,
                "prompt_file": "config/prompts/segment_analysis_v1.txt",
                "llm_weight": 0.4,
            },
        },
        "cropping": {
            "output_width": 1080,
            "output_height": 1920,
            "face_confidence_threshold": 0.6,
            "moving_average_window": 5,
            "sample_fps": 1,
        },
        "rendering": {
            "audio_loudnorm_i": -14,
            "audio_loudnorm_tp": -1.5,
            "audio_loudnorm_lra": 11,
            "padding_color": "black",
        },
        "output": {
            "clips_dir": "output/clips",
            "manifest_file": "output/manifest.json",
            "review_file": "artifacts/review.json",
            "isolated": True,
            "output_dir_template": "{video_name}_{timestamp}",
            "timestamp_format": "%Y-%m-%d_%H-%M-%S",
        },
        "logging": {
            "level": "INFO",
            "log_file": "artifacts/logs/pipeline.log",
            "json_format": True,
        },
        "gpu": {"backend": "auto"},
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file, config_data


@pytest.mark.asyncio
async def test_get_config(client, tmp_config):
    config_file, config_data = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.get(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "config" in data
    assert data["config"]["transcription"]["model"] == "large-v3-turbo"
    # The effective config, the defaults (no user settings) and the user
    # settings themselves, with no error when the settings file is empty.
    assert data["defaults"]["scoring"]["max_clips_per_video"] == 5
    assert data["overrides"] == {}
    assert data["error"] is None


@pytest.mark.asyncio
async def test_get_config_unauthorized(client):
    response = await client.get("/api/v1/config/")
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_config_not_found(client, tmp_path):
    token = await login(client)
    fake_path = tmp_path / "nonexistent.yaml"

    with patch("src.api.routes.config.CONFIG_PATH", fake_path):
        response = await client.get(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_config(client, tmp_config):
    """Saving changes only the user settings file; config.yaml is never written."""
    config_file, config_data = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"scoring": {"max_clips_per_video": 10}}},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["config"]["scoring"]["max_clips_per_video"] == 10
    assert data["config"]["scoring"]["min_duration"] == 45
    assert data["overrides"]["scoring"] == {"max_clips_per_video": 10}
    assert data["error"] is None

    with open(config_file) as f:
        assert yaml.safe_load(f)["scoring"]["max_clips_per_video"] == 5
    with open(user_settings_path()) as f:
        saved = yaml.safe_load(f)
    assert saved == {"scoring": {"max_clips_per_video": 10}}
    assert user_settings_path().stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_update_config_drops_default_values(client, tmp_config):
    """A value equal to the default is not stored: the file stays minimal."""
    config_file, _ = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            # 5 is what config.yaml itself has for max_clips_per_video.
            json={"updates": {"scoring": {"max_clips_per_video": 5}}},
        )

    assert response.status_code == 200
    assert response.json()["overrides"] == {}
    with open(user_settings_path()) as f:
        assert yaml.safe_load(f) is None  # an empty settings file


@pytest.mark.asyncio
async def test_update_config_empty_language_normalizes_to_auto(client, tmp_config):
    """Clearing the language field (or an empty JSON-panel value) means auto, not a crash."""
    config_file, _ = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"transcription": {"language": ""}}},
        )

    assert response.status_code == 200
    assert response.json()["config"]["transcription"]["language"] == "auto"


@pytest.mark.asyncio
async def test_update_config_invalid(client, tmp_config):
    """An invalid value in an allowed section is a 400 with the reason."""
    config_file, _ = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"scoring": {"max_clips_per_video": "not-a-number"}}},
        )

    assert response.status_code == 400
    assert "Invalid config" in response.json()["detail"]
    assert not user_settings_path().exists()


@pytest.mark.asyncio
async def test_update_config_refuses_disallowed_section(client, tmp_config):
    config_file, _ = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"gpu": {"backend": "cuda"}}},
        )

    assert response.status_code == 400
    assert "Disallowed config override keys" in response.json()["detail"]
    assert not user_settings_path().exists()


@pytest.mark.asyncio
async def test_reset_config_removes_keys(client, tmp_config):
    config_file, _ = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"scoring": {"max_clips_per_video": 10, "min_duration": 60}}},
        )
        response = await client.post(
            "/api/v1/config/reset",
            headers={"Authorization": f"Bearer {token}"},
            json={"keys": ["scoring.max_clips_per_video"]},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["overrides"] == {"scoring": {"min_duration": 60}}
    assert data["config"]["scoring"]["max_clips_per_video"] == 5


@pytest.mark.asyncio
async def test_reset_config_empty_list_removes_all(client, tmp_config):
    config_file, _ = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"scoring": {"max_clips_per_video": 10}}},
        )
        response = await client.post(
            "/api/v1/config/reset",
            headers={"Authorization": f"Bearer {token}"},
            json={"keys": []},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["overrides"] == {}
    assert data["config"]["scoring"]["max_clips_per_video"] == 5


@pytest.mark.asyncio
async def test_reset_config_unauthorized(client):
    response = await client.post("/api/v1/config/reset", json={"keys": []})
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_config_reports_broken_settings_file(client, tmp_config):
    """A broken settings file does not break GET: it is reported as ``error``."""
    config_file, _ = tmp_config
    user_settings_path().write_text("{not valid yaml")
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.get(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["error"] is not None and "Cannot read" in data["error"]
    assert data["overrides"] == {}
    assert data["config"] == data["defaults"]


@pytest.mark.asyncio
async def test_update_config_replaces_broken_settings_file(client, tmp_config):
    """Saving over a broken settings file replaces it with valid settings."""
    config_file, _ = tmp_config
    user_settings_path().write_text("{not valid yaml")
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"scoring": {"max_clips_per_video": 10}}},
        )

    assert response.status_code == 200
    assert response.json()["error"] is None
    with open(user_settings_path()) as f:
        assert yaml.safe_load(f) == {"scoring": {"max_clips_per_video": 10}}


@pytest.mark.asyncio
async def test_update_config_unauthorized(client, tmp_config):
    config_file, _ = tmp_config
    response = await client.put(
        "/api/v1/config/",
        json={"updates": {"scoring": {"max_clips_per_video": 10}}},
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_list_profiles(client, tmp_path):
    token = await login(client)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "low_vram.yaml").write_text("gpu:\n  backend: cpu\n")
    (profiles_dir / "high_quality.yaml").write_text("rendering:\n  audio_loudnorm_i: -10\n")

    with patch("src.api.routes.config.PROFILES_DIR", profiles_dir):
        response = await client.get(
            "/api/v1/config/profiles",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert sorted(data["profiles"]) == ["high_quality", "low_vram"]


@pytest.mark.asyncio
async def test_list_profiles_empty(client, tmp_path):
    token = await login(client)
    empty_dir = tmp_path / "empty_profiles"

    with patch("src.api.routes.config.PROFILES_DIR", empty_dir):
        response = await client.get(
            "/api/v1/config/profiles",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json()["profiles"] == []


@pytest.mark.asyncio
async def test_list_profiles_unauthorized(client):
    response = await client.get("/api/v1/config/profiles")
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_validate_config_success(client):
    token = await login(client)

    with patch("src.api.routes.config.run_pipeline") as mock_run:
        response = await client.post(
            "/api/v1/config/validate",
            headers={"Authorization": f"Bearer {token}"},
            json={"config": {}, "input_source": "/dev/null"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["errors"] == []
    mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_validate_config_failure(client):
    token = await login(client)

    with patch("src.api.routes.config.run_pipeline", side_effect=Exception("dry-run failed")):
        response = await client.post(
            "/api/v1/config/validate",
            headers={"Authorization": f"Bearer {token}"},
            json={"config": {}, "input_source": "/dev/null"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert "dry-run failed" in data["errors"][0]


@pytest.mark.asyncio
async def test_validate_config_unauthorized(client):
    response = await client.post(
        "/api/v1/config/validate",
        json={"config": {}, "input_source": "/dev/null"},
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_config_redacts_set_api_key(client, tmp_config):
    config_file, config_data = tmp_config
    config_data["scoring"]["llm"]["api_key"] = "sk-real-secret-value"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.get(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["config"]["scoring"]["llm"]["api_key"] == "***"
    assert "sk-real-secret-value" not in response.text


@pytest.mark.asyncio
async def test_get_config_does_not_mask_unset_api_key(client, tmp_config):
    config_file, config_data = tmp_config
    token = await login(client)

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.get(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json()["config"]["scoring"]["llm"]["api_key"] is None


@pytest.mark.asyncio
async def test_update_config_with_placeholder_secret_keeps_real_value(client, tmp_config):
    """PUTting back the "***" a GET returned must not overwrite the real key."""
    config_file, config_data = tmp_config
    token = await login(client)
    # A real key the user saved earlier in the settings file.
    user_settings_path().write_text(
        yaml.safe_dump({"scoring": {"llm": {"api_key": "sk-real-secret-value"}}})
    )

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"scoring": {"llm": {"api_key": "***", "max_tokens": 500}}}},
        )

    assert response.status_code == 200
    assert response.json()["config"]["scoring"]["llm"]["api_key"] == "***"

    with open(user_settings_path()) as f:
        saved = yaml.safe_load(f)
    assert saved["scoring"]["llm"]["api_key"] == "sk-real-secret-value"
    assert saved["scoring"]["llm"]["max_tokens"] == 500
    with open(config_file) as f:
        assert yaml.safe_load(f)["scoring"]["llm"]["api_key"] is None


@pytest.mark.asyncio
async def test_update_config_can_clear_api_key(client, tmp_config):
    config_file, config_data = tmp_config
    token = await login(client)
    user_settings_path().write_text(
        yaml.safe_dump({"scoring": {"llm": {"api_key": "sk-real-secret-value"}}})
    )

    with patch("src.api.routes.config.CONFIG_PATH", config_file):
        response = await client.put(
            "/api/v1/config/",
            headers={"Authorization": f"Bearer {token}"},
            json={"updates": {"scoring": {"llm": {"api_key": ""}}}},
        )

    assert response.status_code == 200
    with open(user_settings_path()) as f:
        saved = yaml.safe_load(f)
    assert saved["scoring"]["llm"]["api_key"] == ""
