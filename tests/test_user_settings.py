"""Tests for src/user_settings.py: user settings in data/settings.yaml."""

from pathlib import Path

import pytest
import yaml

from src.config import Config
from src.user_settings import (
    ALLOWED_OVERRIDE_SECTIONS,
    apply_user_settings,
    load_app_config,
    load_user_settings,
    remove_user_keys,
    save_user_settings,
    user_settings_path,
)


@pytest.fixture
def settings_path(tmp_path, monkeypatch):
    """The user settings file, pointed at tmp_path (as tests/conftest.py does)."""
    path = tmp_path / "settings.yaml"
    monkeypatch.setenv("PIPELINE_USER_SETTINGS", str(path))
    return path


@pytest.fixture
def base_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {"profiles_dir": str(tmp_path / "profiles"), "scoring": {"max_clips_per_video": 5}}
        )
    )
    return str(path)


class TestUserSettingsPath:
    def test_from_environment(self, settings_path):
        assert user_settings_path() == settings_path

    def test_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PIPELINE_USER_SETTINGS", raising=False)
        assert user_settings_path() == Path("data/settings.yaml")


class TestLoadUserSettings:
    def test_missing_file_means_no_settings(self, settings_path):
        assert load_user_settings() == ({}, None)

    def test_empty_file_means_no_settings(self, settings_path):
        settings_path.write_text("")
        assert load_user_settings() == ({}, None)

    def test_roundtrip(self, settings_path):
        save_user_settings({"scoring": {"max_clips_per_video": 8}})
        assert load_user_settings() == ({"scoring": {"max_clips_per_video": 8}}, None)

    def test_invalid_yaml_is_an_error_not_a_crash(self, settings_path):
        settings_path.write_text("{not valid yaml")
        overrides, error = load_user_settings()
        assert overrides == {}
        assert error is not None and "Cannot read" in error

    def test_not_a_mapping_is_an_error(self, settings_path):
        settings_path.write_text("- just\n- a\n- list\n")
        overrides, error = load_user_settings()
        assert overrides == {}
        assert error is not None and "mapping" in error

    def test_disallowed_sections_are_an_error(self, settings_path):
        settings_path.write_text(yaml.safe_dump({"gpu": {"backend": "cuda"}}))
        overrides, error = load_user_settings()
        assert overrides == {}
        assert error is not None and "Disallowed sections" in error

    def test_scalar_section_value_is_an_error(self, settings_path):
        settings_path.write_text(yaml.safe_dump({"scoring": 5}))
        overrides, error = load_user_settings()
        assert overrides == {}
        assert error is not None and "must be mappings" in error

    def test_allowed_sections_are_shared_with_job_overrides(self):
        """The same list gates the Settings page and job config overrides."""
        assert "scoring" in ALLOWED_OVERRIDE_SECTIONS
        assert not {"work_dir", "gpu", "profiles_dir", "hardware_dir"} & ALLOWED_OVERRIDE_SECTIONS


class TestSaveUserSettings:
    def test_creates_parent_directories(self, settings_path):
        save_user_settings({"scoring": {"max_clips_per_video": 8}})
        assert settings_path.parent.exists()

    def test_atomic_and_private(self, settings_path):
        """The file is written whole (temp + rename) with mode 0600, no temp left."""
        save_user_settings({"scoring": {"max_clips_per_video": 8}})
        assert settings_path.stat().st_mode & 0o777 == 0o600
        assert not list(settings_path.parent.glob("*.tmp"))

        save_user_settings({"scoring": {"max_clips_per_video": 9}})
        assert load_user_settings() == ({"scoring": {"max_clips_per_video": 9}}, None)
        assert not list(settings_path.parent.glob("*.tmp"))

    def test_empty_settings_write_an_empty_file(self, settings_path):
        save_user_settings({})
        assert settings_path.exists() and settings_path.read_text() == ""
        assert load_user_settings() == ({}, None)


class TestApplyUserSettings:
    def test_merges_into_the_config(self, settings_path, base_config):
        save_user_settings({"scoring": {"max_clips_per_video": 8}})
        config, error = apply_user_settings(Config.load(base_config))
        assert error is None
        assert config.scoring.max_clips_per_video == 8

    def test_invalid_settings_keep_the_base_config(self, settings_path, base_config):
        save_user_settings({"scoring": {"max_clips_per_video": "not-a-number"}})
        config, error = apply_user_settings(Config.load(base_config))
        assert config.scoring.max_clips_per_video == 5
        assert error is not None and "Invalid user settings" in error

    def test_empty_language_normalizes_to_auto(self, settings_path, base_config):
        save_user_settings({"transcription": {"language": ""}})
        config, error = apply_user_settings(Config.load(base_config))
        assert error is None
        assert config.transcription.language == "auto"

    def test_broken_file_keeps_the_base_config(self, settings_path, base_config):
        settings_path.write_text("{broken")
        config, error = apply_user_settings(Config.load(base_config))
        assert config.scoring.max_clips_per_video == 5
        assert error is not None


class TestLoadAppConfig:
    def test_layers_user_settings_on_the_base_config(self, settings_path, base_config):
        save_user_settings({"scoring": {"max_clips_per_video": 8}})
        config = load_app_config(base_config)
        assert config.scoring.max_clips_per_video == 8

    def test_broken_settings_are_ignored(self, settings_path, base_config):
        settings_path.write_text("{broken")
        config = load_app_config(base_config)
        assert config.scoring.max_clips_per_video == 5

    def test_missing_base_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_app_config(str(tmp_path / "missing.yaml"))


class TestRemoveUserKeys:
    def test_dotted_keys(self, settings_path):
        save_user_settings(
            {"scoring": {"max_clips_per_video": 8, "llm": {"enabled": True}}, "cleanup": {}}
        )
        remove_user_keys(["scoring.max_clips_per_video", "scoring.llm.enabled"])
        # Sections left without keys (scoring.llm, cleanup) are dropped too.
        assert load_user_settings() == ({}, None)

    def test_unknown_keys_are_ignored(self, settings_path):
        save_user_settings({"scoring": {"max_clips_per_video": 8}})
        remove_user_keys(["scoring.min_duration", "rendering.video_encoder"])
        assert load_user_settings() == ({"scoring": {"max_clips_per_video": 8}}, None)

    def test_empty_list_removes_all(self, settings_path):
        save_user_settings({"scoring": {"max_clips_per_video": 8}})
        remove_user_keys([])
        assert load_user_settings() == ({}, None)
