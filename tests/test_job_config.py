"""
Tests for building a job's pipeline config (src/api/services/job.py).
"""

import json
from pathlib import Path

import pytest
import yaml

from src.api.job_config import build_job_config
from src.config import Config


@pytest.fixture
def base_config(tmp_path):
    """Base config with its own profiles dir and one profile."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "fast.yaml").write_text(
        yaml.safe_dump({"scoring": {"min_score_threshold": 0.3, "max_clips_per_video": 9}})
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"profiles_dir": str(profiles_dir), "scoring": {"min_score_threshold": 0.55}}
        )
    )
    return str(config_path)


class TestBuildJobConfig:
    def test_no_overrides_sets_work_dir(self, base_config):
        cfg = build_job_config(None, "jobs/job_1", base_config)

        assert cfg.work_dir == Path("jobs/job_1")
        assert cfg.scoring.min_score_threshold == 0.55

    def test_profile_applied_before_overrides(self, base_config):
        overrides = {"profile": "fast", "scoring": {"min_score_threshold": 0.7}}

        cfg = build_job_config(json.dumps(overrides), "jobs/job_2", base_config)

        assert cfg.scoring.min_score_threshold == 0.7  # user override wins
        assert cfg.scoring.max_clips_per_video == 9  # from the profile
        assert cfg.work_dir == Path("jobs/job_2")

    def test_profile_only(self, base_config):
        cfg = build_job_config(json.dumps({"profile": "fast"}), "jobs/job_3", base_config)

        assert cfg.scoring.min_score_threshold == 0.3

    def test_disallowed_key_raises(self, base_config):
        with pytest.raises(ValueError, match="Disallowed config override keys"):
            build_job_config(json.dumps({"work_dir": "/etc"}), "jobs/job_4", base_config)

    def test_profile_path_traversal_raises(self, base_config):
        with pytest.raises(ValueError, match="Invalid profile name"):
            build_job_config(json.dumps({"profile": "../config"}), "jobs/job_5", base_config)

    def test_user_settings_below_profile_and_overrides(self, base_config, tmp_path, monkeypatch):
        """Order: config.yaml -> hardware profile -> user settings -> job profile -> job overrides."""
        settings = tmp_path / "settings.yaml"
        settings.write_text(yaml.safe_dump({"scoring": {"max_clips_per_video": 10}}))
        monkeypatch.setenv("PIPELINE_USER_SETTINGS", str(settings))

        cfg = build_job_config(None, "jobs/job_6", base_config)
        assert cfg.scoring.max_clips_per_video == 10  # user settings apply

        # A job profile (and its overrides) still win over the user settings.
        overrides = {"profile": "fast", "scoring": {"min_score_threshold": 0.7}}
        cfg = build_job_config(json.dumps(overrides), "jobs/job_7", base_config)
        assert cfg.scoring.max_clips_per_video == 9  # from the profile
        assert cfg.scoring.min_score_threshold == 0.7  # from the job override


class TestHardwareProfile:
    """config/hardware/<backend>.yaml, picked by PIPELINE_HARDWARE (Docker image)."""

    def _write(self, tmp_path):
        hardware = tmp_path / "hardware"
        hardware.mkdir()
        (hardware / "openvino.yaml").write_text(
            "transcription:\n  engine: whisper_cpp\n  whisper_cpp_binary: whisper-cli\n"
        )
        profiles = tmp_path / "profiles"
        profiles.mkdir()
        (profiles / "slow.yaml").write_text("transcription:\n  engine: faster_whisper\n")
        return Config(hardware_dir=str(hardware), profiles_dir=str(profiles))

    def test_applied_from_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIPELINE_HARDWARE", "openvino")
        cfg = self._write(tmp_path).with_hardware_profile()
        assert cfg.transcription.engine == "whisper_cpp"

    def test_no_variable_no_change(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PIPELINE_HARDWARE", raising=False)
        cfg = self._write(tmp_path).with_hardware_profile()
        assert cfg.transcription.engine == "faster_whisper"

    def test_backend_without_file_needs_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIPELINE_HARDWARE", "cuda")
        cfg = self._write(tmp_path).with_hardware_profile()
        assert cfg.transcription.engine == "faster_whisper"

    def test_user_profile_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIPELINE_HARDWARE", "openvino")
        cfg = self._write(tmp_path).with_hardware_profile().load_profile("slow")
        assert cfg.transcription.engine == "faster_whisper"

    def test_path_traversal_refused(self, tmp_path):
        with pytest.raises(ValueError):
            self._write(tmp_path).with_hardware_profile("../profiles/slow")

    def test_shipped_openvino_profile_is_valid(self):
        cfg = Config().with_hardware_profile("openvino")
        assert cfg.transcription.engine == "whisper_cpp"
        assert cfg.transcription.whisper_cpp_model_path.endswith("ggml-large-v3-turbo.bin")
