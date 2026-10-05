"""
Tests for TASK-P1-04: Low VRAM Profile Optimization

Tests profile loading, merging, VRAM detection, and profile suggestion.
"""

from unittest.mock import patch

import pytest

from src.config import Config, _deep_merge
from src.gpu_utils import GPUConfig, get_vram_gb, suggest_profile


class TestDeepMerge:
    """Test the _deep_merge utility function."""

    def test_merge_flat_dicts(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_merge_override_non_dict_with_dict(self):
        base = {"a": 1}
        override = {"a": {"nested": True}}
        _deep_merge(base, override)
        assert base == {"a": {"nested": True}}

    def test_merge_override_dict_with_non_dict(self):
        base = {"a": {"nested": True}}
        override = {"a": "flat"}
        _deep_merge(base, override)
        assert base == {"a": "flat"}

    def test_merge_empty_override(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {})
        assert base == {"a": 1, "b": 2}

    def test_merge_empty_base(self):
        base = {}
        override = {"a": 1}
        _deep_merge(base, override)
        assert base == {"a": 1}


class TestConfigProfileLoading:
    """Test Config.load_profile method."""

    def test_load_profile_low_vram(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        profile_file = profiles_dir / "low_vram.yaml"
        profile_file.write_text(
            "transcription:\n"
            "  model: small\n"
            "  compute_type: int8\n"
            "  batch_size: 2\n"
            "scoring:\n"
            "  max_clips_per_video: 3\n"
        )

        cfg = Config(profiles_dir=str(profiles_dir))
        updated = cfg.load_profile("low_vram")

        assert updated.transcription.model == "small"
        assert updated.transcription.compute_type == "int8"
        assert updated.transcription.batch_size == 2
        assert updated.scoring.max_clips_per_video == 3

    def test_load_profile_preserves_defaults(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        profile_file = profiles_dir / "test.yaml"
        profile_file.write_text("transcription:\n  model: tiny\n")

        cfg = Config(profiles_dir=str(profiles_dir))
        updated = cfg.load_profile("test")

        assert updated.transcription.model == "tiny"
        assert updated.transcription.language == "auto"
        assert updated.scoring.min_duration == 45

    def test_load_profile_not_found(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        cfg = Config(profiles_dir=str(profiles_dir))
        with pytest.raises(FileNotFoundError, match="Profile not found"):
            cfg.load_profile("nonexistent")

    @pytest.mark.parametrize("name", ["../x", "a/b", "../../etc/passwd"])
    def test_load_profile_rejects_path_traversal(self, tmp_path, name):
        """Profile names come from job JSON; they must stay inside profiles_dir."""
        profiles_dir = tmp_path / "profiles"
        (profiles_dir / "a").mkdir(parents=True)
        (profiles_dir / "a" / "b.yaml").write_text("transcription:\n  model: tiny\n")
        (tmp_path / "x.yaml").write_text("transcription:\n  model: tiny\n")

        cfg = Config(profiles_dir=str(profiles_dir))
        with pytest.raises(ValueError, match="Invalid profile name"):
            cfg.load_profile(name)

    def test_load_profile_backward_compatible(self):
        cfg = Config()
        assert cfg.profiles_dir == "config/profiles"
        assert cfg.transcription.model == "large-v3-turbo"

    def test_load_profile_gpu_override(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        profile_file = profiles_dir / "gpu_test.yaml"
        profile_file.write_text("gpu:\n" "  whisper_compute_override: int8\n")

        cfg = Config(profiles_dir=str(profiles_dir))
        updated = cfg.load_profile("gpu_test")

        assert updated.gpu.whisper_compute_override == "int8"
        assert updated.gpu.backend == "auto"

    def test_load_profile_empty_yaml(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        profile_file = profiles_dir / "empty.yaml"
        profile_file.write_text("")

        cfg = Config(profiles_dir=str(profiles_dir))
        updated = cfg.load_profile("empty")

        assert updated.transcription.model == "large-v3-turbo"


class TestGetVramGb:
    """Test get_vram_gb function."""

    def test_vram_gb_with_cuda(self):
        mock_config = GPUConfig(
            backend="cuda",
            whisper_device="cuda",
            whisper_compute_type="float16",
            cleanup_fn=lambda: None,
            vram_total_mb=12288,
            gpu_name="RTX 3060",
        )
        with patch("src.gpu_utils.get_auto_gpu_config", return_value=mock_config):
            result = get_vram_gb()
            assert result == 12.0

    def test_vram_gb_with_8gb_gpu(self):
        mock_config = GPUConfig(
            backend="cuda",
            whisper_device="cuda",
            whisper_compute_type="int8",
            cleanup_fn=lambda: None,
            vram_total_mb=8192,
            gpu_name="RTX 4060",
        )
        with patch("src.gpu_utils.get_auto_gpu_config", return_value=mock_config):
            result = get_vram_gb()
            assert result == 8.0

    def test_vram_gb_no_gpu(self):
        mock_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=lambda: None,
            vram_total_mb=0,
            gpu_name="CPU",
        )
        with patch("src.gpu_utils.get_auto_gpu_config", return_value=mock_config):
            result = get_vram_gb()
            assert result == 0.0

    def test_vram_gb_none_vram(self):
        mock_config = GPUConfig(
            backend="openvino",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=lambda: None,
            vram_total_mb=None,
            gpu_name="Intel UHD",
        )
        with patch("src.gpu_utils.get_auto_gpu_config", return_value=mock_config):
            result = get_vram_gb()
            assert result == 0.0


class TestSuggestProfile:
    """Test suggest_profile function."""

    def test_suggest_low_vram_for_8gb(self):
        assert suggest_profile(8.0) == "low_vram"

    def test_suggest_low_vram_for_6gb(self):
        assert suggest_profile(6.0) == "low_vram"

    def test_suggest_low_vram_for_4gb(self):
        assert suggest_profile(4.0) == "low_vram"

    def test_no_suggestion_for_12gb(self):
        assert suggest_profile(12.0) is None

    def test_no_suggestion_for_10gb(self):
        assert suggest_profile(10.0) is None

    def test_suggest_low_vram_for_9_9gb(self):
        assert suggest_profile(9.9) == "low_vram"

    def test_no_suggestion_for_zero(self):
        assert suggest_profile(0.0) == "low_vram"


class TestConfigProfilesDir:
    """Test Config profiles_dir field."""

    def test_default_profiles_dir(self):
        cfg = Config()
        assert cfg.profiles_dir == "config/profiles"

    def test_custom_profiles_dir(self):
        cfg = Config(profiles_dir="/custom/path")
        assert cfg.profiles_dir == "/custom/path"


class TestCpuFallbackCompatibility:
    """cpu_fallback was removed from Config; old user files must keep loading.

    Config is a plain pydantic model (extra keys are ignored), so a config or
    profile written before the removal still loads.
    """

    def test_config_with_cpu_fallback_still_loads(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("cpu_fallback: auto\n" "transcription:\n" "  model: small\n")

        cfg = Config.load(str(config_file))

        assert cfg.transcription.model == "small"

    def test_profile_with_cpu_fallback_still_loads(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        profile_file = profiles_dir / "low_vram.yaml"
        profile_file.write_text("cpu_fallback: auto\n" "transcription:\n" "  model: small\n")

        cfg = Config(profiles_dir=str(profiles_dir))
        updated = cfg.load_profile("low_vram")

        assert updated.transcription.model == "small"
